import json
from comet import download_model, load_from_checkpoint

file_path = "outputs/pred_whisper_tiny_nllb_results.jsonl"
out_scores_path = "outputs/comet_scores_whisper_tiny_nllb.jsonl"

print(f"Evaluating COMET: {file_path}")

data = []
try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            src = r.get("sentence_en_gt")
            mt = r.get("pred_zh")
            ref = r.get("ref_zh")

            if not src or not mt or not ref:
                continue

            data.append({
                "src": src.strip(),
                "mt": mt.strip(),
                "ref": ref.strip()
            })

except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please run inference first.")

if data:
    print("Loading COMET model (first run will download)...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    outputs = model.predict(
        data,
        batch_size=8,
        gpus=1  # 没 GPU 就改成 0
    )

    print("-" * 50)
    print(f"Whisper-Tiny + NLLB-600M COMET: {outputs.system_score:.4f}")
    print("-" * 50)

    with open(out_scores_path, "w", encoding="utf-8") as f:
        for ex, s in zip(data, outputs.scores):
            rec = {**ex, "comet": float(s)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Per-sentence COMET scores saved to: {out_scores_path}")

else:
    print("No predictions to evaluate.")