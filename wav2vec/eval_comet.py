import glob, json, torch
from comet import download_model, load_from_checkpoint

MODEL = "Unbabel/wmt22-comet-da"  # reference-based

model_path = download_model(MODEL)
model = load_from_checkpoint(model_path)
model.eval()

data = []
for fp in sorted(glob.glob("outputs_mt/pred_*.jsonl")):
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r.get("src_en")
            mt  = r.get("pred_zh")
            ref = r.get("ref_zh")
            if not (src and mt and ref):
                continue
            data.append({"src": src, "mt": mt, "ref": ref})

gpus = 1 if torch.cuda.is_available() else 0
out = model.predict(data, batch_size=16, gpus=gpus)
print("COMET(system_score):", out.system_score)