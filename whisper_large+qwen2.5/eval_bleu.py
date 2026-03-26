import glob, json
import evaluate

# 加载评测指标 (sacrebleu 会自动处理中文分词)
bleu = evaluate.load("sacrebleu")

preds, refs = [], []
# 匹配你刚才 run_heavy.py 生成的文件名
file_path = "outputs/pred_whisper_large_qwen_results.jsonl"

print(f"Evaluating: {file_path}")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mt = r.get("pred_zh")
            ref = r.get("ref_zh")
            if not mt or not ref:
                continue
            preds.append(mt)
            refs.append([ref])  # sacrebleu 要求每个参考答案是一个列表
except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please run the inference script first.")

if preds:
    # 重点：sacrebleu 在 compute 时，如果检测到是中文，会自动应用其内部的中文处理逻辑
    res = bleu.compute(predictions=preds, references=refs, tokenize="zh")
    print("-" * 30)
    print(f"Whisper-Large + Qwen-2.5 BLEU: {res['score']:.2f}")
    print("-" * 30)
else:
    print("No predictions to evaluate.")