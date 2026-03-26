import torch
import os, json
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)

# --- 配置区 ---
ASR_MODEL = "openai/whisper-large-v3"
MT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载数据 (只取前 1000 条)
print("Loading CoVoST 2 English-Chinese dataset (1000 samples)...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test").select(range(1000))
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# 2. 初始化 ASR (Whisper Large)
print(f"Initializing ASR: {ASR_MODEL}")
asr_pipe = pipeline(
    "automatic-speech-recognition", 
    model=ASR_MODEL, 
    device=device,
    model_kwargs={"torch_dtype": torch.float16} # 使用半精度减少显存
)

# 3. 初始化 MT (Qwen 2.5 4-bit 量化)
print(f"Initializing MT: {MT_MODEL}")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MT_MODEL, 
    quantization_config=quant_config, 
    device_map="auto"
)

results = []

# 4. 级联推理
for ex in tqdm(ds, desc="Heavy Cascade Inference"):
    audio_array = ex["audio"]["array"]
    ref_zh = ex["translation"]
    
    # 第一棒: Audio -> English Text (加上长音频处理)
    asr_out = asr_pipe(audio_array, return_timestamps=True)
    pred_en = asr_out["text"]
    
    # 第二棒: English Text -> Chinese Text (使用 Chat Template)
    messages = [
        {"role": "system", "content": "You are a professional translator. Translate the following English speech transcript to natural Chinese."},
        {"role": "user", "content": f"English: {pred_en}\nChinese:"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    # 裁剪掉 Prompt，只留翻译结果
    pred_zh = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    results.append({
        "id": ex.get("id", ""),
        "sentence_en_gt": ex["sentence"],
        "pred_en": pred_en,
        "ref_zh": ref_zh,
        "pred_zh": pred_zh.strip()
    })

# 5. 保存结果
save_path = f"{OUT_DIR}/pred_whisper_large_qwen_results.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone! Heavy results saved to {save_path}")