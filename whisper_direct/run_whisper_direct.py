import os, json, time
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, WhisperForConditionalGeneration

# -------------------- Config --------------------
MODEL_NAME = "openai/whisper-large-v3"
DATASET_NAME = "fixie-ai/covost2"
CONFIG_NAME = "en_zh-CN"
SPLIT = "test"
OUT_DIR = "outputs"
SHARD_SIZE = 500          
BATCH_SIZE = 1            
MAX_NEW_TOKENS = 128
MAX_AUDIO_SECONDS = 30 
SUBSET_N = 1000           # 保持 1000 条的测试规模

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading dataset...")
ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
if SUBSET_N:
    ds = ds.select(range(min(SUBSET_N, len(ds))))

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
model.eval()

# 配置翻译任务：英文音频 -> 中文文本
forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

def translate_batch(batch):
    arrays = []
    max_len = int(MAX_AUDIO_SECONDS * 16000)
    for a in batch["audio"]:
        arr = a["array"]
        arrays.append(arr[:max_len] if len(arr) > max_len else arr)

    inputs = processor(audio=arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    # 统一输入张量的设备与精度
    inputs = {k: v.to(device, dtype=dtype if k=="input_features" else torch.long) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, forced_decoder_ids=forced_ids, max_new_tokens=MAX_NEW_TOKENS)

    preds = processor.batch_decode(out, skip_special_tokens=True)
    return {"pred_zh": preds}

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

n = len(ds)
print(f"Total examples: {n}")

total_examples_done = 0
total_infer_seconds = 0.0

# 迭代处理 Shards
for start in range(0, n, SHARD_SIZE):
    end = min(start + SHARD_SIZE, n)
    shard_path = os.path.join(OUT_DIR, f"pred_whisper_{start:05d}_{end-1:05d}.jsonl")
    
    if os.path.exists(shard_path):
        print(f"[skip] {shard_path} exists")
        continue

    sub = ds.select(range(start, end))
    print(f"Translating {start}..{end-1} (size={len(sub)})")

    t0 = time.time()
    pred_sub = sub.map(translate_batch, batched=True, batch_size=BATCH_SIZE)
    # 同步 GPU 确保 latency 准确
    torch.cuda.synchronize() if device == "cuda" else None
    infer_sec = time.time() - t0

    total_infer_seconds += infer_sec
    total_examples_done += len(sub)

    rows = []
    for ex in pred_sub:
        rows.append({
            "id": ex.get("client_id", None), # CoVoST2 默认是 client_id
            "sentence_en": ex.get("sentence", None),
            "ref_zh": ex.get("translation", None),
            "pred_zh": ex.get("pred_zh", None),
        })
    save_jsonl(shard_path, rows)
    print(f"[saved] {shard_path} | shard_avg={infer_sec/len(sub):.4f}s/ex")

# 生成统一的 Summary
summary = {
    "model": MODEL_NAME,
    "device": device,
    "examples_done": total_examples_done,
    "infer_seconds_total": total_infer_seconds,
    "avg_seconds_per_example": total_infer_seconds / total_examples_done if total_examples_done else 0,
}

with open(os.path.join(OUT_DIR, "latency_whisper.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("All tasks finished.")