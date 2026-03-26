import os, json, time
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --- 配置区 ---
MODELS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    # "llama": "meta-llama/Meta-Llama-3-8B-Instruct"
}

# 是否保存每条样本的耗时（会比较大）
SAVE_PER_EX_LATENCY = False

print("Loading CoVoST 2 English texts...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test").select(range(1000))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

def _sync_if_cuda(device: str):
    # GPU 计时必须同步，否则会低估时间
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def run_model(name, path):
    print(f"\n>>> Starting {name} ({path})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_str = "torch.float16" if device == "cuda" else "torch.float32"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- 模型加载 ---
    tokenizer = AutoTokenizer.from_pretrained(path)

    if name in ["nllb", "mbart"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            use_safetensors=True
        )
        if device == "cpu":
            model = model.to("cpu")
    else:
        # Qwen/Llama: 量化一般需要 GPU；如果你在 CPU 跑，可能会报错/很慢
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=quant_config if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to("cpu")

    model.eval()

    results = []
    per_ex_latency = []  # 可选：逐条耗时

    # --- latency 统计：只统计 inference（不包含写文件/释放显存）---
    infer_t0 = time.perf_counter()

    for ex in tqdm(ds, desc=f"Inference {name}"):
        en_text = ex["sentence"]

        # 单条计时（可选）
        if SAVE_PER_EX_LATENCY:
            _sync_if_cuda(device)
            ex_t0 = time.perf_counter()

        # --- 推理 ---
        if name == "nllb":
            inputs = tokenizer(en_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            tgt_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=128
                )
            pred = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

        elif name == "mbart":
            inputs = tokenizer(en_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # mBART 的中文代码
            if hasattr(tokenizer, "lang_code_to_id") and "zh_CN" in tokenizer.lang_code_to_id:
                tgt_lang_id = tokenizer.lang_code_to_id["zh_CN"]
            else:
                tgt_lang_id = tokenizer.convert_tokens_to_ids("zh_CN")

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=128
                )
            pred = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

        elif name in ["qwen", "llama"]:
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate English to Chinese."},
                {"role": "user", "content": f"English: {en_text}\nChinese:"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            pred = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        else:
            raise ValueError(f"Unknown model key: {name}")

        if SAVE_PER_EX_LATENCY:
            _sync_if_cuda(device)
            ex_t1 = time.perf_counter()
            per_ex_latency.append(ex_t1 - ex_t0)

        results.append({
            "id": ex.get("id", ""),
            "sentence_en": en_text,
            "ref_zh": ex["translation"],
            "pred_zh": pred.strip()
        })

    _sync_if_cuda(device)
    infer_t1 = time.perf_counter()

    infer_seconds_total = infer_t1 - infer_t0
    examples_done = len(results)
    avg_seconds_per_example = infer_seconds_total / max(1, examples_done)
    avg_examples_per_second = examples_done / max(1e-9, infer_seconds_total)

    # --- 保存预测 ---
    save_path = f"pred_{name}_results.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 保存 latency 汇总 ---
    latency_path = f"latency_{name}.json"
    latency_obj = {
        "stage": "unimodal_mt",
        "model": path,
        "split": "test",
        "src_lang": "en",
        "tgt_lang": "zh",
        "device": device,
        "dtype": str(dtype_str),
        "batch_size": 1,  # 你当前是逐条推理
        "max_new_tokens": 128,
        "examples_done": examples_done,
        "infer_seconds_total": infer_seconds_total,
        "avg_seconds_per_example": avg_seconds_per_example,
        "avg_examples_per_second": avg_examples_per_second
    }
    if SAVE_PER_EX_LATENCY:
        latency_obj["per_example_seconds"] = per_ex_latency

    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_obj, f, ensure_ascii=False, indent=2)

    print(f"\nSaved predictions to: {save_path}")
    print(f"Saved latency to: {latency_path}")
    print(json.dumps(latency_obj, ensure_ascii=False, indent=2))

    # --- 释放显存 ---
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- 依次执行 ---
for name, path in MODELS.items():
    run_model(name, path)