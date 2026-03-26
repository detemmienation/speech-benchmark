import json
import time
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration

# -------------------- Config --------------------
MODEL_NAME = "openai/whisper-large-v3"   # or "openai/whisper-large-v2"
DATASET_NAME = "fixie-ai/covost2"
CONFIG_NAME = "en_zh-CN"
SPLIT = "test"

OUT_DIR = Path("outputs")
SHARD_SIZE = 250

BATCH_SIZE = 1                 # A10G: start with 1 for large-v3
MAX_AUDIO_SECONDS = 30.0       # truncate long clips to reduce OOM risk
MAX_NEW_TOKENS = 128

SUBSET_N = 1000     # 只跑前 1000 条
SUBSET_SEED = 42    # 固定随机种子，保证每次抽样一致
DO_SHUFFLE = False   # True=随机抽样1000条；False=取前1000条（按原顺序）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def out_path(start: int, end: int) -> Path:
    return OUT_DIR / f"pred_{start:05d}_{end:05d}.jsonl"


def safe_str(x) -> str:
    return "" if x is None else str(x)


def truncate_audio(array, sr: int, max_sec: float):
    if array is None:
        return array
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

    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    if DO_SHUFFLE:
        ds = ds.shuffle(seed=SUBSET_SEED)
    if SUBSET_N is not None:
        ds = ds.select(range(min(SUBSET_N, len(ds))))

    n = len(ds)
    print(f"Total examples: {n}")

    ex0 = ds[0]["audio"]
    print("Audio example type:", type(ex0), "keys:", list(ex0.keys()) if isinstance(ex0, dict) else ex0)

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    # make sure feature extractor returns attention_mask
    if hasattr(processor, "feature_extractor"):
        try:
            processor.feature_extractor.return_attention_mask = True
        except Exception:
            pass

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    # translate to Chinese
    forced_ids = processor.get_decoder_prompt_ids(language="zh", task="translate")

    latency = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "batch_size": BATCH_SIZE,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "shard_size": SHARD_SIZE,
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
        sub = sub.cast_column("audio", Audio(sampling_rate=16000))

        print(f"Translating {start}..{end} (size={len(sub)})")

        def translate_batch(batch):
            arrays = []
            for a in batch["audio"]:
                arr = truncate_audio(a["array"], a["sampling_rate"], MAX_AUDIO_SECONDS)
                arrays.append(arr)

            inputs = processor(
                audio=arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,   # important for warning
            )

            # ---- FIX: move to device + align dtype with model ----
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(device=DEVICE, dtype=DTYPE)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(device=DEVICE)
            # some versions might return other tensors
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and k not in ("input_features", "attention_mask"):
                    inputs[k] = v.to(device=DEVICE)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    forced_decoder_ids=forced_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

            preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

            del inputs, gen_ids
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            return {"pred_zh": preds}

        t0 = time.time()

        pred_sub = sub.map(
            translate_batch,
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=[c for c in sub.column_names if c != "audio"],
        )

        shard_time = time.time() - t0
        avg = shard_time / len(sub)

        with open(fp, "w", encoding="utf-8") as f:
            for k in range(len(sub)):
                ex = sub[k]
                src, ref = guess_fields(ex)
                f.write(
                    json.dumps(
                        {
                            "idx": start + k,
                            "src_en": src,
                            "ref_zh": ref,
                            "pred_zh": pred_sub[k]["pred_zh"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        latency["shards"].append(
            {"start": start, "end": end, "count": len(sub), "shard_time_sec": shard_time, "avg_sec_per_ex": avg}
        )
        print(f"[saved] {fp} | shard_time={shard_time:.2f}s | shard_avg={avg:.4f}s/ex")

    latency["total_time_sec"] = time.time() - t_total0
    with open(OUT_DIR / "latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)

    print("Done. Latency summary written to outputs/latency_summary.json")


if __name__ == "__main__":
    main()