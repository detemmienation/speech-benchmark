import glob, json
import evaluate

# 1. 加载评测指标 (sacrebleu 会自动处理中文分词)
bleu = evaluate.load("sacrebleu")

preds, refs = [], []
# 指向 Qwen2-Audio 的结果文件
file_path = "outputs/pred_qwen2_audio_results.jsonl"

print(f"Evaluating Qwen2-Audio: {file_path}")

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        mt = r.get("pred_zh")
        ref = r.get("ref_zh")
        if not mt or not ref:
            continue
        preds.append(mt)
        refs.append([ref])  # sacrebleu 要求每个参考答案是一个列表

# 2. 计算分数
if preds:
    # 直接计算，不手动分词，让 sacrebleu 自己处理
    res = bleu.compute(predictions=preds, references=refs, tokenize="zh")
    print("=" * 30)
    print(f"Qwen2-Audio BLEU: {res['score']:.2f}")
    print("=" * 30)
else:
    print("Error: No data found.")