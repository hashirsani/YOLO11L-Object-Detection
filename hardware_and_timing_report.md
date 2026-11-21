# Hardware & Full Run Timings

## Environment
- **Platform:** Kaggle Notebooks (GPU T4 x2)
- **GPU:** 2 × NVIDIA Tesla T4 (15,360 MiB VRAM each; 30,720 MiB total)
- **CPU:** Intel(R) Xeon(R) CPU @ 2.00GHz (4 vCPUs)
- **RAM:** 31 GB
- **Disk:** SSD-backed `/kaggle/working` container volume (20 GB allocated)
- **Operating System:** Ubuntu 24.04.1 LTS (Python 3.13.15, CUDA 12.8)

## Full 50-File Run Timings
- **Dataset Scope:** First 50 FineVideo shards (`train-00000-of-01357.parquet` through `train-00049-of-01357.parquet`), totaling **1,650 videos** and **461,757 seconds (~128.3 hours)** of footage.
- **Shard Download Time (untimed setup):** `101 seconds`
- **Inference & Indexing Runtime (`run_info.json`):**
  - First 25 shards (`half1`): `6,509 seconds` (~1h 48m 29s)
  - Second 25 shards (`half2`): `6,211 seconds` (~1h 43m 31s)
  - **Total Inference Run Time:** **`12,720 seconds` (3 hours 32 minutes 00 seconds)**
- **Total End-to-End Session Wall Time (including package install, `yolo11l.pt` download, shard download, inference, gzip output, and verification):** **`~12,860 seconds` (~3 hours 34 minutes 20 seconds)**