import os
import json
import time
import torch
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# -----------------------
# Config
# -----------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATASET_ID = "fixie-ai/covost2"
CONFIG_NAME = "en_zh-CN"
SPLIT = "test"

NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 256

PREFIXES_TO_REMOVE = [
    "The Chinese translation of the English audio is:",
    "The translation is:",
    "The audio says:",
    "Here is the translation:",
    "The audio translates to:",  # important for your current output pattern
]

DEBUG_FIRST_N = 3  # print audio stats for first N examples


def move_inputs_to_model_device(inputs, model):
    """
    When using device_map="auto", model is sharded across devices.
    A safe approach:
      - Move input_ids/attention_mask to model.device
      - For remaining tensors, move to model.device as well
    """
    target_device = model.device

    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(target_device)
    return inputs


def clean_prediction(text: str) -> str:
    if text is None:
        return ""
    pred = text.strip()
    for p in PREFIXES_TO_REMOVE:
        if pred.startswith(p):
            pred = pred[len(p):].strip()
    pred = pred.strip(" '\"‘’“”")
    return pred


def run_qwen2_audio():
    print(f"\n>>> Starting Inference: {MODEL_ID}")

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # -----------------------
    # Load dataset
    # -----------------------
    print(f"Loading {DATASET_ID} ({NUM_SAMPLES} samples)...")
    ds = load_dataset(
        DATASET_ID,
        CONFIG_NAME,
        split=SPLIT,
        trust_remote_code=True
    ).select(range(NUM_SAMPLES))

    # Force resample to 16k (important for many speech models)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # -----------------------
    # Load model & processor
    # -----------------------
    print("Loading processor & model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",   # keep your choice
    )
    model.eval()

    results = []
    infer_seconds_total = 0.0
    examples_done = 0

    # -----------------------
    # Inference loop
    # -----------------------
    for idx, ex in enumerate(tqdm(ds, desc="Qwen2-Audio Inference")):
        audio_array = ex["audio"]["array"]
        sampling_rate = ex["audio"]["sampling_rate"]

        # Debug audio sanity check
        if idx < DEBUG_FIRST_N:
            mean_abs = float(torch.tensor(audio_array).abs().mean())
            print(f"\n[DEBUG] idx={idx} audio_len={len(audio_array)} sr={sampling_rate} mean_abs={mean_abs:.6f}")

        # Minimal prompt (reduce instruction echo)
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": "Translate the audio to Chinese."}
            ]}
        ]

        chat_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # IMPORTANT: use audio= (NOT audios=)
        inputs = processor(
            text=chat_text,
            audio=[audio_array],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )

        # Move tensors to model.device safely
        inputs = move_inputs_to_model_device(inputs, model)

        # Timing
        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        if use_cuda:
            torch.cuda.synchronize()
        end_time = time.time()

        infer_seconds_total += (end_time - start_time)
        examples_done += 1

        # Decode only newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = generate_ids[:, prompt_len:]

        pred_zh_raw = processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        pred_zh = clean_prediction(pred_zh_raw)

        results.append({
            "id": ex.get("client_id", str(examples_done)),
            "sentence_en_gt": ex["sentence"],
            "pred_en": "",
            "ref_zh": ex["translation"],
            "pred_zh": pred_zh
        })

        # Optional: quick peek
        if idx < DEBUG_FIRST_N:
            print(f"[DEBUG] raw_pred: {pred_zh_raw}")
            print(f"[DEBUG] clean_pred: {pred_zh}")

    # -----------------------
    # Save JSONL
    # -----------------------
    name = "qwen2_audio"
    save_path = os.path.join(OUT_DIR, f"pred_{name}_results.jsonl")
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {save_path}")

    # -----------------------
    # Latency stats
    # -----------------------
    avg_sec_per_ex = infer_seconds_total / examples_done if examples_done > 0 else 0.0
    avg_ex_per_sec = 1.0 / avg_sec_per_ex if avg_sec_per_ex > 0 else 0.0

    latency_stats = {
        "model": MODEL_ID,
        "split": SPLIT,
        "tgt_lang": "cmn",
        "batch_size": 1,
        "shard_size": 1000,
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": "cuda" if use_cuda else "cpu",
        "dtype": str(dtype),
        "examples_done": examples_done,
        "infer_seconds_total": infer_seconds_total,
        "avg_seconds_per_example": avg_sec_per_ex,
        "avg_examples_per_second": avg_ex_per_sec
    }

    latency_path = os.path.join(OUT_DIR, f"latency_{name}.json")
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_stats, f, indent=4, ensure_ascii=False)
    print(f"Latency stats saved to {latency_path}")


if __name__ == "__main__":
    run_qwen2_audio()