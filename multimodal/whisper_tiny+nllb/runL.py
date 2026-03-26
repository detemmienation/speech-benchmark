import torch
import os
import json
import time
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# -----------------------
# 配置区
# -----------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "whisper_tiny_nllb"
NUM_SAMPLES = 100

device = 0 if torch.cuda.is_available() else -1  # pipeline 用 0 / -1

# -----------------------
# 主函数
# -----------------------
def run_cascade():

    print(f"\n>>> Starting Cascade: Whisper-Tiny + NLLB")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 1. 加载数据
    print(f"Loading CoVoST2 ({NUM_SAMPLES} samples)...")
    ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test").select(range(NUM_SAMPLES))
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # 2. 加载 ASR
    print("Loading Whisper-Tiny...")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device
    )

    # 3. 加载 MT
    print("Loading NLLB-600M...")
    mt_model_path = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(mt_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_path)

    if torch.cuda.is_available():
        model = model.to("cuda")

    results = []

    total_asr_time = 0.0
    total_mt_time = 0.0
    total_time = 0.0
    examples_done = 0

    # -----------------------
    # 推理循环
    # -----------------------
    for ex in tqdm(ds, desc="Cascade Inference"):

        audio_array = ex["audio"]["array"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_total = time.time()

        # -------- ASR --------
        start_asr = time.time()
        asr_out = asr_pipe(audio_array)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_asr = time.time()

        pred_en = asr_out["text"].strip()

        # -------- MT --------
        start_mt = time.time()

        inputs = tokenizer(pred_en, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        tgt_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")

        out = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=128
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_mt = time.time()

        pred_zh = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

        end_total = time.time()

        # -------- 统计 --------
        asr_time = end_asr - start_asr
        mt_time = end_mt - start_mt
        total_sample_time = end_total - start_total

        total_asr_time += asr_time
        total_mt_time += mt_time
        total_time += total_sample_time
        examples_done += 1

        results.append({
            "id": ex.get("id", ""),
            "sentence_en_gt": ex["sentence"],
            "pred_en": pred_en,
            "ref_zh": ex["translation"],
            "pred_zh": pred_zh
        })

    # -----------------------
    # 保存预测
    # -----------------------
    save_path = f"{OUT_DIR}/pred_{MODEL_NAME}_results for latency.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {save_path}")

    # -----------------------
    # 保存 latency
    # -----------------------
    avg_total = total_time / examples_done
    avg_asr = total_asr_time / examples_done
    avg_mt = total_mt_time / examples_done

    latency_stats = {
        "model": MODEL_NAME,
        "split": "test",
        "tgt_lang": "cmn",
        "batch_size": 1,
        "examples_done": examples_done,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "total_infer_seconds": total_time,
        "avg_seconds_per_example": avg_total,
        "avg_asr_seconds": avg_asr,
        "avg_mt_seconds": avg_mt,
        "examples_per_second": 1.0 / avg_total
    }

    latency_path = f"{OUT_DIR}/latency_{MODEL_NAME}.json"
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_stats, f, indent=4)

    print(f"Latency saved to {latency_path}")


if __name__ == "__main__":
    run_cascade()