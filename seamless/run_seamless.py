import os, json, time
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

MODEL_NAME = "facebook/seamless-m4t-v2-large"
TGT_LANG = "cmn"          # 简体中文
SPLIT = "test"
OUT_DIR = "outputs"
SHARD_SIZE = 500          # 每 500 条一个 jsonl，断点续跑友好
BATCH_SIZE = 2           # A10G 通常 8；OOM 就改 4
MAX_NEW_TOKENS = 256

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading dataset...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split=SPLIT)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
model.eval()

# def translate_batch(batch):
#     arrays = [a["array"] for a in batch["audio"]]
#     inputs = processor(audio=arrays, sampling_rate=16000, return_tensors="pt", padding=True)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         out = model.generate(**inputs, tgt_lang=TGT_LANG, max_new_tokens=MAX_NEW_TOKENS)
#     preds = processor.batch_decode(out, skip_special_tokens=True)
#     return {"pred_zh": preds}

MAX_AUDIO_SECONDS = 30  # 你可以改成 25/20 更稳，30 一般足够

def translate_batch(batch):
    arrays = []
    max_len = int(MAX_AUDIO_SECONDS * 16000)  # 16000Hz

    for a in batch["audio"]:
        arr = a["array"]
        if len(arr) > max_len:
            arr = arr[:max_len]   # 截断到最大长度
        arrays.append(arr)

    inputs = processor(audio=arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, tgt_lang=TGT_LANG, max_new_tokens=MAX_NEW_TOKENS)

    preds = processor.batch_decode(out, skip_special_tokens=True)
    return {"pred_zh": preds}

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

n = len(ds)
print(f"Total examples: {n}")

# latency stats
total_examples_done = 0
total_infer_seconds = 0.0

# Warmup（更稳定的 latency）
warm = ds.select(range(min(8, n)))
_ = warm.map(translate_batch, batched=True, batch_size=min(BATCH_SIZE, 8))
torch.cuda.synchronize() if device == "cuda" else None

for start in range(0, n, SHARD_SIZE):
    end = min(start + SHARD_SIZE, n)
    shard_path = os.path.join(OUT_DIR, f"pred_{start:05d}_{end-1:05d}.jsonl")
    if os.path.exists(shard_path):
        print(f"[skip] {shard_path} exists")
        continue

    sub = ds.select(range(start, end))
    print(f"Translating {start}..{end-1} (size={len(sub)})")

    t0 = time.time()
    pred_sub = sub.map(translate_batch, batched=True, batch_size=BATCH_SIZE)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()

    infer_sec = t1 - t0
    total_infer_seconds += infer_sec
    total_examples_done += len(sub)

    rows = []
    for ex in pred_sub:
        rows.append({
            "id": ex.get("id", None),
            "sentence_en": ex.get("sentence", None),
            "ref_zh": ex.get("translation", None),
            "pred_zh": ex.get("pred_zh", None),
        })
    save_jsonl(shard_path, rows)

    print(f"[saved] {shard_path} | shard_time={infer_sec:.2f}s | shard_avg={infer_sec/len(sub):.4f}s/ex")

summary = {
    "model": MODEL_NAME,
    "split": SPLIT,
    "tgt_lang": TGT_LANG,
    "batch_size": BATCH_SIZE,
    "shard_size": SHARD_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "device": device,
    "dtype": str(dtype),
    "examples_done": total_examples_done,
    "infer_seconds_total": total_infer_seconds,
    "avg_seconds_per_example": (total_infer_seconds / total_examples_done) if total_examples_done else None,
    "avg_examples_per_second": (total_examples_done / total_infer_seconds) if total_infer_seconds else None,
}

with open(os.path.join(OUT_DIR, "latency_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done. Latency summary written to outputs/latency_summary.json")