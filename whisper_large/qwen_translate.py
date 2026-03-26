import json
import glob
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

INPUT_DIR = Path("outputs")       # whisper outputs
OUTPUT_DIR = Path("outputs_qwen")   # qwen translated outputs
BATCH_SIZE = 2                    # start with 2 on A10G, bump to 4 if ok
MAX_NEW_TOKENS = 256
# ----------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)
model.eval()


def build_prompt(text: str) -> str:
    return (
        "You are a professional translator.\n"
        "Translate the following English sentence into natural Simplified Chinese.\n"
        "Only output the Chinese translation.\n\n"
        f"English: {text}\nChinese:"
    )


def translate_batch(texts):
    prompts = [build_prompt(t) for t in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # extract portion after "Chinese:"
    cleaned = []
    for out in decoded:
        if "Chinese:" in out:
            cleaned.append(out.split("Chinese:", 1)[-1].strip())
        else:
            cleaned.append(out.strip())
    return cleaned


latency = {
    "stage": "mt_qwen",
    "model": MODEL_NAME,
    "device": DEVICE,
    "dtype": str(DTYPE),
    "batch_size": BATCH_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "files": [],
}

t_total0 = time.time()

files = sorted(glob.glob(str(INPUT_DIR / "pred_*.jsonl")))

for file in files:
    output_path = OUTPUT_DIR / Path(file).name
    if output_path.exists():
        print(f"[skip] {output_path} exists")
        continue

    print(f"Processing {file}")
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    t0 = time.time()
    out_rows = []

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]

        # Whisper ASR output is stored in pred_zh field in your current files (english text)
        english_texts = [item["pred_zh"] for item in batch]
        zh_texts = translate_batch(english_texts)

        for item, zh in zip(batch, zh_texts):
            item["pred_zh"] = zh
            out_rows.append(item)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    file_time = time.time() - t0
    avg = file_time / max(1, len(data))

    with open(output_path, "w", encoding="utf-8") as f:
        for item in out_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    latency["files"].append(
        {"file": Path(file).name, "count": len(data), "time_sec": file_time, "avg_sec_per_ex": avg}
    )
    print(f"[saved] {output_path} | file_time={file_time:.2f}s | avg={avg:.4f}s/ex")

latency["total_time_sec"] = time.time() - t_total0
with open(OUTPUT_DIR / "latency_summary.json", "w", encoding="utf-8") as f:
    json.dump(latency, f, ensure_ascii=False, indent=2)

print("Done. Qwen latency summary written to outputs_mt/latency_summary.json")