import torch
import os, json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- 配置区 ---
MODELS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    # "llama": "meta-llama/Meta-Llama-3-8B-Instruct"
}
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 加载数据
print("Loading CoVoST 2 English texts...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test").select(range(1000))

# 量化配置 (用于 7B/8B 模型)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

def run_model(name, path):
    print(f"\n>>> Starting {name} ({path})")
    
    # --- 模型加载逻辑 ---
    if name in ["nllb", "mbart"]:
        # Seq2Seq 架构
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path, 
            device_map="auto", 
            dtype=torch.float16,  # 这里顺便修正了之前的 deprecated 警告
            use_safetensors=True   # 显式指定使用 safetensors
        )
    else:
        # CausalLM 架构 (Qwen/Llama)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=quant_config, device_map="auto"
        )
    
    results = []
    for ex in tqdm(ds, desc=f"Inference {name}"):
        en_text = ex["sentence"]
        
        # --- 针对不同模型的 Prompt/Generation 策略 ---
        if name == "nllb":
            inputs = tokenizer(en_text, return_tensors="pt").to("cuda")
            
            # 修复：使用 convert_tokens_to_ids 获取中文 ID
            tgt_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")
            
            out = model.generate(
                **inputs, 
                forced_bos_token_id=tgt_lang_id, 
                max_length=128
            )
            pred = tokenizer.batch_decode(out, skip_special_tokens=True)[0] 
        elif name == "mbart":
            inputs = tokenizer(en_text, return_tensors="pt").to("cuda")
            # mBART 的中文代码通常是 zh_CN
            tgt_lang_id = tokenizer.lang_code_to_id["zh_CN"] # mBART 通常支持这个属性
            # 如果 mBART 也报错，就改成下面这行：
            # tgt_lang_id = tokenizer.convert_tokens_to_ids("zh_CN")
            
            out = model.generate(**inputs, forced_bos_token_id=tgt_lang_id, max_length=128)
            pred = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        elif name in ["qwen", "llama"]:
            # 使用 Chat Template 提高翻译准确性
            messages = [
                {"role": "system", "content": "You are a professional translator. Translate English to Chinese."},
                {"role": "user", "content": f"English: {en_text}\nChinese:"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            # 裁剪掉输入部分
            pred = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        results.append({
            "id": ex.get("id", ""),
            "sentence_en": en_text,
            "ref_zh": ex["translation"],
            "pred_zh": pred.strip()
        })

    # 保存文件
    save_path = f"{OUT_DIR}/pred_{name}_results.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # 释放显存，防止跑下一个模型时 OOM
    del model, tokenizer
    torch.cuda.empty_cache()

# --- 依次执行 ---
for name, path in MODELS.items():
    run_model(name, path)