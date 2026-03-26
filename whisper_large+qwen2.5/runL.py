import os
import json
import time
import torch
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# -----------------------
# Config
# -----------------------
ASR_MODEL = "openai/whisper-large-v3"
MT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

NAME = "whisper_large_qwen"
NUM_SAMPLES = 100
MAX_NEW_TOKENS = 128

use_cuda = torch.cuda.is_available()
asr_device = 0 if use_cuda else -1  # pipeline device: 0/-1
print("CUDA:", use_cuda)

# -----------------------
# Load dataset
# -----------------------
print(f"Loading CoVoST2 English-Chinese dataset ({NUM_SAMPLES} samples)...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test").select(range(NUM_SAMPLES))
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# -----------------------
# Init ASR
# -----------------------
print(f"Initializing ASR: {ASR_MODEL}")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=ASR_MODEL,
    device=asr_device,
    model_kwargs={"torch_dtype": torch.float16} if use_cuda else None
)

# -----------------------
# Init MT (4-bit)
# -----------------------
print(f"Initializing MT: {MT_MODEL} (4-bit nf4)")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MT_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MT_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

model.eval()

# 生成时用 eos 当 pad，避免 warning
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

# -----------------------
# Helper: move inputs
# -----------------------
def move_inputs_to_model_device(inputs):
    # device_map="auto" 情况下，通常 model.device 是主设备（一般 cuda:0）
    target = model.device
    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(target)
    return inputs

# -----------------------
# Inference
# -----------------------
results = []

total_asr_time = 0.0
total_mt_time = 0.0
total_time = 0.0
examples_done = 0

for ex in tqdm(ds, desc="Heavy Cascade Inference"):
    audio_array = ex["audio"]["array"]
    ref_zh = ex["translation"]

    if use_cuda:
        torch.cuda.synchronize()
    t_total0 = time.time()

    # ---- ASR: Audio -> English ----
    t_asr0 = time.time()
    asr_out = asr_pipe(audio_array, return_timestamps=True)
    if use_cuda:
        torch.cuda.synchronize()
    t_asr1 = time.time()

    pred_en = (asr_out.get("text") or "").strip()

    # ---- MT: English -> Chinese ----
    t_mt0 = time.time()

    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the following English speech transcript to natural Chinese."
        },
        {"role": "user", "content": f"English: {pred_en}\nChinese:"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = move_inputs_to_model_device(inputs)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id
        )

    if use_cuda:
        torch.cuda.synchronize()
    t_mt1 = time.time()

    # 只取生成部分（裁掉 prompt）
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = out[:, prompt_len:]
    pred_zh = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

    t_total1 = time.time()

    # ---- latency accumulate ----
    asr_time = t_asr1 - t_asr0
    mt_time = t_mt1 - t_mt0
    tot_time = t_total1 - t_total0

    total_asr_time += asr_time
    total_mt_time += mt_time
    total_time += tot_time
    examples_done += 1

    results.append({
        "id": ex.get("id", ""),
        "sentence_en_gt": ex["sentence"],
        "pred_en": pred_en,
        "ref_zh": ref_zh,
        "pred_zh": pred_zh
    })

# -----------------------
# Save predictions
# -----------------------
save_path = f"{OUT_DIR}/pred_{NAME}_results_for_latency.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone! Results saved to {save_path}")

# -----------------------
# Save latency stats
# -----------------------
avg_total = total_time / examples_done if examples_done else 0.0
avg_asr = total_asr_time / examples_done if examples_done else 0.0
avg_mt = total_mt_time / examples_done if examples_done else 0.0

latency_stats = {
    "model": NAME,
    "asr_model": ASR_MODEL,
    "mt_model": MT_MODEL,
    "split": "test",
    "tgt_lang": "cmn",
    "batch_size": 1,
    "examples_done": examples_done,
    "device": "cuda" if use_cuda else "cpu",
    "max_new_tokens": MAX_NEW_TOKENS,
    "total_infer_seconds": total_time,
    "avg_seconds_per_example": avg_total,
    "avg_asr_seconds": avg_asr,
    "avg_mt_seconds": avg_mt,
    "avg_examples_per_second": (1.0 / avg_total) if avg_total > 0 else 0.0
}

latency_path = f"{OUT_DIR}/latency_{NAME}.json"
with open(latency_path, "w", encoding="utf-8") as f:
    json.dump(latency_stats, f, indent=4, ensure_ascii=False)

print(f"Latency saved to {latency_path}")