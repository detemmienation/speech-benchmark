import json
from comet import download_model, load_from_checkpoint

def run_comet_eval():
    file_path = "outputs/pred_qwen2_audio_results.jsonl"
    print(f"Evaluating COMET for: {file_path}")

    model_input = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # 根据 run.py 中定义的字段提取
            src = r.get("sentence_en_gt") 
            mt = r.get("pred_zh")
            ref = r.get("ref_zh")
            
            if not mt or not ref or not src:
                continue
                
            model_input.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })

    if not model_input:
        print("Error: No data found.")
        return

    # 加载 Unbabel 推荐模型
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    # 预测并输出
    model_output = model.predict(model_input, batch_size=8, gpus=1)
    
    print("=" * 30)
    print(f"Qwen2-Audio COMET System Score: {model_output.system_score:.4f}")
    print("=" * 30)

if __name__ == "__main__":
    run_comet_eval()