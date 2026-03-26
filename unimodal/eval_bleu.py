import glob, json
import evaluate

bleu = evaluate.load("sacrebleu")

# 示例调用
for model_name in ["nllb", "mbart", "qwen"]:
    refs = []
    preds = []
    with open(f"pred_{model_name}_results.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            refs.append(data['ref'])
            preds.append(data['pred'])
    
    res = bleu.compute(predictions=preds, references=refs, tokenize="zh")
    print(f"BLEU for {model_name}:", res["score"])
    print(res)
