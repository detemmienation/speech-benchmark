import json
import time
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# -------------------- Config --------------------
# 使用 base-960h，它是针对英文 ASR 优化的标准模型
MODEL_NAME = "facebook/wav2vec2-base-960h"
DATASET_NAME = "fixie-ai/covost2"
CONFIG_NAME = "en_zh-CN"
SPLIT = "test"

OUT_DIR = Path("outputs")
SHARD_SIZE = 250

# Wav2Vec 2.0 比较轻量，Batch Size 可以适当调大
BATCH_SIZE = 8                 
MAX_AUDIO_SECONDS = 30.0
SUBSET_N = 1000                
DO_SHUFFLE = False             

# Mac 环境自动识别 mps，服务器识别 cuda
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Wav2Vec 2.0 某些算子在半精度下可能不稳，建议优先使用 float32
DTYPE = torch.float32 

def out_path(start: int, end: int) -> Path:
    # 命名为 pred_ 前缀，方便区分级联的中间产物
    return OUT_DIR / f"pred_{start:05d}_{end:05d}.jsonl"

def safe_str(x) -> str:
    return "" if x is None else str(x)

def truncate_audio(array, sr: int, max_sec: float):
    if array is None: return array
    max_len = int(sr * max_sec)
    if len(array) > max_len:
        return array[:max_len]
    return array

def guess_fields(ex: dict):
    src = ex.get("sentence") or ex.get("src_text") or ex.get("text") or ""
    ref = ex.get("translation") or ex.get("tgt_text") or ex.get("target") or ""
    return safe_str(src), safe_str(ref)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset on {DEVICE}...")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    # Wav2Vec 2.0 强制要求 16kHz 采样率
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    if SUBSET_N is not None:
        ds = ds.select(range(min(SUBSET_N, len(ds))))

    n = len(ds)
    print(f"Total examples: {n}")

    print(f"Loading model: {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
    model.eval()

    latency = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "batch_size": BATCH_SIZE,
        "total_examples": n,
        "shards": [],
    }
    t_total0 = time.time()

    for start in range(0, n, SHARD_SIZE):
        end = min(n - 1, start + SHARD_SIZE - 1)
        fp = out_path(start, end)

        if fp.exists():
            print(f"[skip] {fp} exists")
            continue

        sub = ds.select(range(start, end + 1))
        print(f"Transcribing {start}..{end}")

        def asr_batch(batch):
            arrays = [truncate_audio(a["array"], 16000, MAX_AUDIO_SECONDS) for a in batch["audio"]]
            
            inputs = processor(
                arrays, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).to(DEVICE, dtype=DTYPE)

            with torch.no_grad():
                logits = model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            # Wav2Vec 2.0 输出通常是大写，需要 lowercase 适配后续翻译模型
            transcriptions = [t.lower() for t in processor.batch_decode(predicted_ids)]

            return {"asr_text": transcriptions}

        t0 = time.time()
        # 执行 ASR
        processed_sub = sub.map(asr_batch, batched=True, batch_size=BATCH_SIZE)

        with open(fp, "w", encoding="utf-8") as f:
            for k in range(len(sub)):
                ex = sub[k]
                src_en_ref, ref_zh = guess_fields(ex)
                f.write(json.dumps({
                    "idx": start + k,
                    "src_en": src_en_ref, # 原始英文参考
                    "ref_zh": ref_zh,         # 最终中文参考
                    "pred_zh": processed_sub[k]["asr_text"], # ASR 识别出的英文 (喂给 mBART)
                }, ensure_ascii=False) + "\n")

        shard_time = time.time() - t0
        avg = shard_time / len(sub)
        latency["shards"].append({
            "start": start, 
            "end": end, 
            "count": len(sub), 
            "shard_time_sec": shard_time, 
            "avg_sec_per_ex": avg
        })
        print(f"[saved] {fp} | shard_avg={avg:.4f}s/ex")

    latency["total_time_sec"] = time.time() - t_total0
    with open(OUT_DIR / "latency_summary_wav2vec.json", "w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)

    print(f"Done. Summary written to {OUT_DIR}/latency_summary_wav2vec.json")

if __name__ == "__main__":
    main()