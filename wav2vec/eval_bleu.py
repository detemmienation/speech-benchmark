import glob, json
import evaluate

bleu = evaluate.load("sacrebleu")

preds, refs = [], []
for fp in sorted(glob.glob("outputs_mt/pred_*.jsonl")):    
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mt = r.get("pred_zh")
            ref = r.get("ref_zh")
            if not mt or not ref:
                continue
            preds.append(mt)
            refs.append([ref])  # sacrebleu expects list of references per example

res = bleu.compute(predictions=preds, references=refs, tokenize="zh")
print("BLEU:", res["score"])
print(res)