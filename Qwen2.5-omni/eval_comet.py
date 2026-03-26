import json
from comet import download_model, load_from_checkpoint

# 修改成你当前 Qwen2.5-Omni 输出文件名
file_path = "outputs/pred_qwen2_5_omni_7b_results.jsonl"

print(f"Evaluating COMET: {file_path}")

data = []

try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            src = r.get("sentence_en_gt")
            mt = r.get("pred_zh")
            ref = r.get("ref_zh")

            # COMET 需要 src + mt + ref
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
    print("Loading COMET model (first time will download)...")

    # 推荐用这个稳定版本
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    # gpus=1 表示用一张 GPU；如果想用 CPU 改成 gpus=0
    outputs = model.predict(
        data,
        batch_size=8,
        gpus=1
    )

    print("-" * 40)
    print(f"Qwen2.5-Omni-7B COMET: {outputs.system_score:.4f}")
    print("-" * 40)

else:
    print("No predictions to evaluate.")