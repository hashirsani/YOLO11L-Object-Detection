# Video Indexing at Speed (YOLO11l)

Indexes FineVideo Parquet shards to detect five target objects (`person`, `car`, `dog`, `laptop`, `cell phone`) and count visible people per second using Ultralytics `yolo11l.pt`.

## Repository Contents
- `run.py` — CLI entry point (`--input <parquet_dir> --output <results.jsonl>`) with multi-GPU/multiprocess execution and crash/error isolation.
- `pipeline.py` — Core video decoding (`PyAV`), rotation handling, frame sampling (`SAMPLE_K=4`), batched `YOLO11l` inference (`imgsz=640`, `conf=0.4`, `FP16`), and per-second aggregation.
- `prepare.py` — One-time preparation script that downloads `yolo11l.pt` next to the code for offline execution.
- `requirements.txt` — Exact Python dependencies.
- `results.jsonl.gz` — Gzip-compressed results for the first 50 FineVideo shards (`train-00000` through `train-00049`).
- `HARDWARE.md` — Hardware specification and full 50-file runtime breakdown.
- `WRITEUP.md` — Answers to the take-home questions.
- `ai_chats/` — Full AI chat history for this task.

## 1. System Packages (Optional)
PyAV (`av==19.0.2`) bundles its own FFmpeg libraries in standard Linux wheels, so no extra `apt-get` packages are strictly required on Ubuntu 22.04. If your environment needs system video/GL libraries:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0
```

## 2. Python Installation
Create and activate a Python virtual environment, then install exact dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: If installing CUDA-enabled PyTorch wheels on a fresh machine requires the PyTorch CUDA 12 index, run `pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128` followed by `pip install -r requirements.txt`.)*

## 3. Preparation Command (Requires Internet, Run Once Before Timing)
Download the official `yolo11l.pt` weights into the repository directory so `run.py` runs completely offline:

```bash
python prepare.py
```

## 4. Running Inference (Offline)
Run the indexer on a folder of `.parquet` files (works with internet disabled once `yolo11l.pt` is present):

```bash
python run.py --input /path/to/parquet_folder --output results.jsonl
```

### Optional Environment Variables
- `SAMPLE_K` — Frames sampled per second (`default: 4`; set `0` for all frames)
- `YOLO_HALF` — `"1"` for FP16 (`default: 1`), `"0"` for FP32
- `YOLO_BATCH` — Frames per GPU batch (`default: 32`)
- `OBJ_FRAC` — Presence fraction threshold per second (`default: 0.5`)
- `WORKERS` — Worker processes (`default: min(cpu_count, 6)`; set `0` for single-process)