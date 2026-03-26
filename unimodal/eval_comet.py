import glob, json, torch
from comet import download_model, load_from_checkpoint

MODEL = "Unbabel/wmt22-comet-da"  # reference-based

model_path = download_model(MODEL)
model = load_from_checkpoint(model_path)
model.eval()

data = []
for model_name in ["nllb", "mbart", "qwen"]:
    with open(f"pred_{model_name}_results.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            src = r.get("en")
            mt  = r.get("pred")
            ref = r.get("ref")
            if not (src and mt and ref):
                continue
            data.append({"src": src, "mt": mt, "ref": ref})

    gpus = 1 if torch.cuda.is_available() else 0
    out = model.predict(data, batch_size=16, gpus=gpus)
    print(f"COMET(system_score) for {model_name}:", out.system_score)