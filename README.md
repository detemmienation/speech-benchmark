# speech-benchmark
speech-benchmark for 11-777 project

## overview
This project benchmarks speech-to-text translation models on:
- Dataset: CoVoST2 (en → zh-CN, test split)
-	Task: English speech → Chinese text
- Metrics: BLEU, COMET, Latency

### Current Models
- seamless_large/ → facebook/seamless-m4t-v2-large
- whisper_large/

### Future models (planned):
-	qwen3_omni/

Each model has its own folder containing:
```
run.py
eval_bleu.py
eval_comet.py
outputs/
```
## Benchmark Enviroment(at root repository)
```
cd ~
python3 -m venv benchmark-env
source benchmark-env/bin/activate
pip install -U pip

# download requirements
pip install torch==2.5.1 torchaudio==2.5.1
pip install transformers>=4.41.0 accelerate>=0.30.0
pip install datasets==3.6.0
pip install evaluate sacrebleu comet-ml unbabel-comet
pip install soundfile librosa pandas tqdm

# system requirements
sudo apt update
sudo apt install -y libsndfile1 ffmpeg
```

## How to Run (Example: Seamless)
```
cd seamless

python3 -m venv seamless-env
source seamless-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_seamless.py
```



## Evaluation

BLEU:

```python eval_bleu.py```

COMET:

```python eval_comet.py```

Latency results are saved to:

```outputs/latency_summary.json```


## Environment

Tested on:
- GPU: NVIDIA A10G (24GB)
- PyTorch 2.5.1
- Transformers 4.41+
