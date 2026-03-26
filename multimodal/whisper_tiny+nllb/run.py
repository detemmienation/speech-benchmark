import torch
import os, json
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# --- 配置区 ---
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. 加载数据 (必须 cast Audio 才能读音频文件)
print("Loading CoVoST 2 (1000 samples)...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split="test").select(range(1000))
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_cascade():
    name = "whisper_tiny_nllb"
    print(f"\n>>> Starting Cascade: Whisper-Tiny + NLLB")
    
    # --- 加载 ASR 和 MT ---
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
    
    # 这里使用 NLLB 600M 
    mt_model_path = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(mt_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_path).to(device)
    
    results = []
    for ex in tqdm(ds, desc="Cascade Inference"):
        # ASR 阶段: Audio -> English Text
        audio_array = ex["audio"]["array"]
        asr_out = asr_pipe(audio_array, return_timestamps=True)
        pred_en = asr_out["text"]
        
        # MT 阶段: English Text -> Chinese Text
        inputs = tokenizer(pred_en, return_tensors="pt").to(device)
        # NLLB 简体中文代码是 zho_Hans
        tgt_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")
        
        out = model.generate(
            **inputs, 
            forced_bos_token_id=tgt_lang_id, 
            max_length=128
        )
        pred_zh = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

        results.append({
            "id": ex.get("id", ""),
            "sentence_en_gt": ex["sentence"],
            "pred_en": pred_en,
            "ref_zh": ex["translation"],
            "pred_zh": pred_zh.strip()
        })

    # 保存文件
    save_path = f"{OUT_DIR}/pred_{name}_results.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run_cascade()