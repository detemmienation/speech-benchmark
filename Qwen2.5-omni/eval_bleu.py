import json
import evaluate

# 加载 sacreBLEU 指标
bleu = evaluate.load("sacrebleu")

preds, refs = [], []

# 修改成你当前 Qwen2.5-Omni 输出文件名
file_path = "outputs/pred_qwen2_5_omni_7b_results.jsonl"

print(f"Evaluating: {file_path}")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mt = r.get("pred_zh")
            ref = r.get("ref_zh")

            # 过滤空值
            if not mt or not ref:
                continue

            preds.append(mt.strip())
            refs.append([ref.strip()])  # sacrebleu 要求每个 reference 是 list

except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please run inference first.")

if preds:
    # tokenize="zh" 对中文 BLEU 很重要
    res = bleu.compute(
        predictions=preds,
        references=refs,
        tokenize="zh"
    )

    print("-" * 40)
    print(f"Qwen2.5-Omni-7B BLEU: {res['score']:.2f}")
    print("-" * 40)
else:
    print("No predictions to evaluate.")