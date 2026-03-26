import os, json, time
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
import torch
from datasets import load_dataset, Audio
from tqdm import tqdm

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SPLIT = "test"
OUT_DIR = "outputs"
SHARD_SIZE = 500
BATCH_SIZE = 1                 # 先用1最稳；显存足再尝试2
MAX_NEW_TOKENS = 256
MAX_AUDIO_SECONDS = 30         # 跟你 seamless 一致
USE_AUDIO_IN_VIDEO = True      # 参照模型卡
# 只跑子集的话可打开：
SUBSET_N = 1000                # e.g., 1000
SUBSET_SEED = 42
DO_SHUFFLE = False
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

print("Loading dataset...")
ds = load_dataset("fixie-ai/covost2", "en_zh-CN", split=SPLIT)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

if SUBSET_N is not None:
    if DO_SHUFFLE:
        ds = ds.shuffle(seed=SUBSET_SEED)
    ds = ds.select(range(min(SUBSET_N, len(ds))))

n = len(ds)
print(f"Total examples: {n}")

print("Loading model + processor...")
offload_dir = Path("./offload_qwen3_omni")
offload_dir.mkdir(parents=True, exist_ok=True)

# 你这台机器 RAM 只有 15GiB，所以 CPU 内存预算别写太激进
max_memory = {
    0: "22GiB",      # A10G 24GB 留余量
    "cpu": "12GiB",  # 给系统留 2-3GiB，不然容易炸
}

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
    max_memory=max_memory,
    offload_folder=str(offload_dir),
    offload_state_dict=True,
)
model.disable_talker()
model.eval()


processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_NAME)

def truncate_audio(arr, sr, max_sec):
    max_len = int(sr * max_sec)
    return arr[:max_len] if arr is not None and len(arr) > max_len else arr

def build_conversation(audio_path: str):
    # 中文指令：只输出中文译文，别啰嗦
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "请把这段英文语音翻译成自然的简体中文。只输出中文翻译，不要解释。"},
            ],
        }
    ]

def infer_one(audio_path: str) -> str:
    conversation = build_conversation(audio_path)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # 兼容返回结构：取 text token
    if hasattr(out, "sequences"):
        sequences = out.sequences
    elif isinstance(out, tuple) and hasattr(out[0], "sequences"):
        sequences = out[0].sequences
    elif isinstance(out, tuple):
        sequences = out[0]
    else:
        sequences = out

    pred = processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return pred

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# latency stats（对齐你的 seamless summary 字段）
total_examples_done = 0
total_infer_seconds = 0.0

# Warmup（可选）
warm_n = min(2, n)
if device == "cuda" and warm_n > 0:
    print("Warmup...")
    with TemporaryDirectory() as tmpdir:
        for i in range(warm_n):
            ex = ds[i]["audio"]
            arr = truncate_audio(ex["array"], ex["sampling_rate"], MAX_AUDIO_SECONDS)
            wav_path = str(Path(tmpdir) / f"warm_{i}.wav")
            sf.write(wav_path, arr, 16000)
            _ = infer_one(wav_path)
    torch.cuda.synchronize()

for start in range(0, n, SHARD_SIZE):
    end = min(start + SHARD_SIZE, n)
    shard_path = os.path.join(OUT_DIR, f"pred_{start:05d}_{end-1:05d}.jsonl")
    if os.path.exists(shard_path):
        print(f"[skip] {shard_path} exists")
        continue

    print(f"Translating {start}..{end-1} (size={end-start})")

    t0 = time.time()
    rows = []

    with TemporaryDirectory() as tmpdir:
        # 为了“输出一致 + 简单稳定”，这里逐条跑（BATCH_SIZE=1）
        # 如果你确认显存很富余，可以改成批处理，但需要更复杂的多音频输入组织
        for i in tqdm(range(start, end), desc=f"shard {start}-{end-1}"):
            ex = ds[i]
            audio = ex["audio"]
            arr = truncate_audio(audio["array"], audio["sampling_rate"], MAX_AUDIO_SECONDS)

            wav_path = str(Path(tmpdir) / f"{i:06d}.wav")
            sf.write(wav_path, arr, 16000)

            pred = infer_one(wav_path)

            rows.append(
                {
                    "id": ex.get("id", None),
                    "sentence_en": ex.get("sentence", None),
                    "ref_zh": ex.get("translation", None),
                    "pred_zh": pred,  # 注意：字段名对齐 seamless
                }
            )

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    infer_sec = t1 - t0
    total_infer_seconds += infer_sec
    total_examples_done += (end - start)

    save_jsonl(shard_path, rows)
    print(f"[saved] {shard_path} | shard_time={infer_sec:.2f}s | shard_avg={infer_sec/(end-start):.4f}s/ex")

summary = {
    "model": MODEL_NAME,
    "split": SPLIT,
    "tgt_lang": "zh",  # Qwen3-Omni 用 prompt 控制；这里写 zh
    "batch_size": BATCH_SIZE,
    "shard_size": SHARD_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "max_audio_seconds": MAX_AUDIO_SECONDS,
    "device": device,
    "dtype": "auto",
    "examples_done": total_examples_done,
    "infer_seconds_total": total_infer_seconds,
    "avg_seconds_per_example": (total_infer_seconds / total_examples_done) if total_examples_done else None,
    "avg_examples_per_second": (total_examples_done / total_infer_seconds) if total_infer_seconds else None,
}

with open(os.path.join(OUT_DIR, "latency_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Done. Latency summary written to outputs/latency_summary.json")