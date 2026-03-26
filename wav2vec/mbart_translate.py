# mt_mbart.py
import json
import glob
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG ----------------
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CPU 上建议 fp32；如果你在新 Intel CPU 上想试试 bf16，可改成 torch.bfloat16（不保证每环境都支持）
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

INPUT_DIR = Path("outputs")       # whisper outputs
OUTPUT_DIR = Path("outputs_mt")   # mt outputs
BATCH_SIZE = 8 if DEVICE == "cuda" else 2  # CPU 先从 2 开始
MAX_NEW_TOKENS = 128
# ----------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading mBART model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
).to(DEVICE)
model.eval()

# mBART-50 language codes
SRC_LANG = "en_XX"
TGT_LANG = "zh_CN"

# set source language for tokenizer
tokenizer.src_lang = SRC_LANG
forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

def translate_batch(texts):
    # mBART expects language codes; we set tokenizer.src_lang and force BOS to target lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,  # English sentence length cap; adjust if needed
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.strip() for d in decoded]


latency = {
    "stage": "mt_mbart",
    "model": MODEL_NAME,
    "device": DEVICE,
    "dtype": str(DTYPE),
    "batch_size": BATCH_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "src_lang": SRC_LANG,
    "tgt_lang": TGT_LANG,
    "files": [],
}

t_total0 = time.time()

files = sorted(glob.glob(str(INPUT_DIR / "pred_*.jsonl")))
if not files:
    raise FileNotFoundError(f"No pred_*.jsonl found under {INPUT_DIR.resolve()}")

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

    # sanity check
    if data and "pred_zh" not in data[0]:
        print("[warn] src_en not found in file rows. Available keys:", list(data[0].keys()))

    t0 = time.time()
    out_rows = []

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]

        english_texts = [item.get("pred_zh", "") for item in batch]
        zh_texts = translate_batch(english_texts)

        for item, zh in zip(batch, zh_texts):
            item["pred_zh"] = zh  # keep whisper pred_zh intact
            out_rows.append(item)

        # CPU 不需要 empty_cache；GPU 可以保留
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

print("Done. mBART latency summary written to outputs_mt/latency_summary.json")