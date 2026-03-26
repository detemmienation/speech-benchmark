import json
from comet import download_model, load_from_checkpoint

# 1. 下载并加载标准的 COMET 模型
# 建议使用这个轻量但准确的模型：wmt22-comet-da
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

file_path = "outputs/pred_whisper_large_qwen_results.jsonl"
data = []

print(f"Loading data from {file_path}...")
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        # COMET 需要三个部分：源文本(src), 模型翻译(mt), 参考答案(ref)
        data.append({
            "src": r.get("sentence_en_gt", ""), # ASR识别的 pred_en 也可以，但通常用 gt 作为源
            "mt": r.get("pred_zh", ""),
            "ref": r.get("ref_zh", "")
        })

# 2. 计算分数
print("Computing COMET scores (this may take a few minutes on GPU)...")
model_output = model.predict(data, batch_size=8, gpus=1)

print("-" * 40)
print(f"Whisper-Large + Qwen-2.5 COMET Score: {model_output.system_score:.4f}")
print("-" * 40)