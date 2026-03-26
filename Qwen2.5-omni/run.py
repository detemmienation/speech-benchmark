import os
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

# -----------------------
# Config
# -----------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
NAME = "qwen2_5_omni_7b"

NUM_SAMPLES = 1000          # 先小跑可改成 10
MAX_NEW_TOKENS = 128       # 关键：避免生成一堆“Human: …”对话

PREFIXES_TO_REMOVE = [
    "The audio translates to:",
    "The Chinese translation of the English audio is:",
    "The translation is:",
    "The audio says:",
    "Here is the translation:",
]

# 如果模型开始自问自答，常见会出现这些标记，直接截断
STOP_STRINGS = ["Human:", "Assistant:", "\nHuman:", "\nAssistant:"]


def clean_prediction(text: str) -> str:
    if text is None:
        return ""

    # 1) 先截断到第一个 stop string 之前
    for s in STOP_STRINGS:
        if s in text:
            text = text.split(s, 1)[0]

    pred = text.strip()

    # 2) 去掉常见废话前缀
    for p in PREFIXES_TO_REMOVE:
        if pred.startswith(p):
            pred = pred[len(p):].strip()

    # 3) 去掉首尾引号
    pred = pred.strip(" '\"‘’“”")
    return pred


def move_inputs_to_model_device(inputs, model):
    # device_map="auto" 时，稳妥做法：把 inputs tensor 移到 model.device
    target = model.device
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(target)
    return inputs


def run():
    print(f"\n>>> Starting Inference: {MODEL_ID}", flush=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32
    print("device:", "cuda" if use_cuda else "cpu", "dtype:", dtype, flush=True)

    # 1) Dataset (强制重采样到 16k)
    print(f"Loading CoVoST2 en_zh-CN (first {NUM_SAMPLES} samples)...", flush=True)
    ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test", trust_remote_code=True).select(range(NUM_SAMPLES))
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # 2) Processor & Model (text-only thinker)
    print("Loading processor/model (Thinker: text output only)...", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    print("model.device:", model.device, flush=True)

    # 用 eos 作为 pad，避免某些情况下 warning / 停不下来
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    results = []
    infer_seconds_total = 0.0

    for idx, ex in enumerate(tqdm(ds, desc=f"{NAME} Inference")):
        audio_array = np.asarray(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])

        # 3) Prompt（尽量简短，减少模型“发挥”）
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": "Translate the audio to Chinese."}
            ]}
        ]
        chat_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        # ✅ 关键：audio= 不是 audios=
        inputs = processor(
            text=chat_text,
            audio=[audio_array],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = move_inputs_to_model_device(inputs, model)

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id
            )

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        infer_seconds_total += (t1 - t0)

        # 只解码新生成部分
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = out_ids[:, prompt_len:]

        pred_raw = processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        pred_zh = clean_prediction(pred_raw)

        results.append({
            "id": ex.get("client_id", str(idx)),
            "sentence_en_gt": ex["sentence"],
            "pred_en": "",
            "ref_zh": ex["translation"],
            "pred_zh": pred_zh
        })

        # debug 前两条看一下效果
        if idx < 2:
            print("\n[DEBUG] raw:", pred_raw, flush=True)
            print("[DEBUG] cleaned:", pred_zh, flush=True)

    # 4) Save JSONL
    save_path = os.path.join(OUT_DIR, f"pred_{NAME}_results.jsonl")
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {save_path}", flush=True)

    # 5) Latency stats
    examples_done = len(results)
    avg_sec = infer_seconds_total / examples_done if examples_done else 0.0
    latency_stats = {
        "model": MODEL_ID,
        "split": "test",
        "tgt_lang": "cmn",
        "batch_size": 1,
        "shard_size": 1000,
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": "cuda" if use_cuda else "cpu",
        "dtype": str(dtype).replace("torch.", ""),
        "examples_done": examples_done,
        "infer_seconds_total": infer_seconds_total,
        "avg_seconds_per_example": avg_sec,
        "avg_examples_per_second": (1.0 / avg_sec) if avg_sec > 0 else 0.0
    }

    latency_path = os.path.join(OUT_DIR, f"latency_{NAME}.json")
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_stats, f, indent=4, ensure_ascii=False)
    print(f"Latency stats saved to {latency_path}", flush=True)


if __name__ == "__main__":
    run()