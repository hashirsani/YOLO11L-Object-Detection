"""Core processing for run.py.

Defaults are the FINAL settings (what the graders will run):
  SAMPLE_K=4 frames per second, FP16.
For the all-frames FP32 reference use: SAMPLE_K=0 YOLO_HALF=0 WORKERS=0

Env knobs (all optional):
  SAMPLE_K    frames kept per second (0 = every frame)            default 4
  YOLO_HALF   "1" = FP16, "0" = FP32                               default 1
  YOLO_BATCH  frames per GPU batch                                  default 32
  OBJ_FRAC    object present if hits >= OBJ_FRAC * sampled frames   default 0.5
  WORKERS     worker processes (0 = single process)                 default min(cpus, 6)
  ROW_LIMIT   only the first N videos of each parquet (0 = all)
  ROW_LIST    text file of "parquet_name:row" lines; only those videos run
"""
import io
import json
import math
import os
from fractions import Fraction
from pathlib import Path

CLASS_NAMES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
ORDER = ["person", "car", "dog", "laptop", "cell phone"]
CONF = 0.4
IMGSZ = 640
BATCH = int(os.environ.get("YOLO_BATCH", "32"))
HALF = os.environ.get("YOLO_HALF", "1") == "1"
SAMPLE_K = int(os.environ.get("SAMPLE_K", "4"))
OBJ_FRAC = float(os.environ.get("OBJ_FRAC", "0.5"))
ROW_LIMIT = int(os.environ.get("ROW_LIMIT", "0"))
_rl = os.environ.get("ROW_LIST")
ROW_SET = set(Path(_rl).read_text().split()) if _rl else None

_model = None
_device = None  # set per worker process


def get_model():
    global _model
    if _model is None:
        import logging
        from ultralytics import YOLO
        logging.getLogger("ultralytics").setLevel(logging.ERROR)  # silence 'half' warning
        local = Path(__file__).with_name("yolo11l.pt")
        weights = os.environ.get("YOLO_WEIGHTS", str(local) if local.exists() else "yolo11l.pt")
        _model = YOLO(weights)
    return _model


def count_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def init_worker(counter, n_gpus):
    """Runs once in each worker process: pick a GPU, limit CPU threads, load the model."""
    global _device
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    _device = f"cuda:{idx % n_gpus}" if n_gpus else "cpu"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    get_model()


def detect(frames):
    """frames: list of BGR uint8 arrays. Returns [(set_of_names, people_count), ...]."""
    kw = {"device": _device} if _device else {}
    results = get_model().predict(
        frames, imgsz=IMGSZ, conf=CONF, classes=list(CLASS_NAMES),
        half=HALF, verbose=False, **kw,
    )
    out = []
    for r in results:
        cls = r.boxes.cls.int().tolist()
        out.append(({CLASS_NAMES[c] for c in cls}, cls.count(0)))
    return out


def summarize(stats):
    """stats: per-frame [(set_of_names, people), ...] for ONE second -> (objects, people).

    object present = found in at least OBJ_FRAC of the sampled frames (0.5 = half)
    people         = median of per-frame counts, floored when halfway
    """
    n = len(stats)
    if n == 0:
        return [], 0
    need = OBJ_FRAC * n - 1e-9
    objects = [o for o in ORDER if sum(o in s for s, _ in stats) >= need]
    counts = sorted(p for _, p in stats)
    if n % 2:
        people = counts[n // 2]
    else:
        people = (counts[n // 2 - 1] + counts[n // 2]) // 2
    return objects, people


def slot_for(t, sec, k):
    """Which of the k equal slots of second `sec` the timestamp t falls in."""
    frac = t - sec
    if frac < 0:
        frac = 0
    return min(k - 1, int(frac * k))


def get_rotation(stream):
    """Rotation tag in degrees (0/90/180/270)."""
    try:
        r = stream.metadata.get("rotate")
        if r is not None:
            return int(float(r)) % 360
    except Exception:
        pass
    return 0


def process_video(mp4_bytes):
    """Yield (second, objects, people) for every second, in order."""
    import av
    import numpy as np

    container = av.open(io.BytesIO(mp4_bytes))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        k = (-(get_rotation(stream)) // 90) % 4  # clockwise tag -> np.rot90 steps
        fps = stream.average_rate or Fraction(30)

        pts0 = None
        n_frames = 0
        max_sec = -1
        next_emit = 0
        per = {}          # second -> list of per-frame stats
        imgs, secs = [], []
        cur_sec, taken = -1, set()

        def run_batch():
            for sec, st in zip(secs, detect(imgs)):
                per.setdefault(sec, []).append(st)
            imgs.clear()
            secs.clear()

        for frame in container.decode(stream):
            if frame.pts is not None:
                if pts0 is None:
                    pts0 = frame.pts
                t = (frame.pts - pts0) * stream.time_base
            else:
                t = Fraction(n_frames) / fps
            n_frames += 1
            sec = max(max_sec, 0, math.floor(t))   # keep seconds monotonic
            max_sec = sec

            if SAMPLE_K:                      # keep the first frame of each slot
                if sec != cur_sec:
                    cur_sec, taken = sec, set()
                slot = slot_for(t, sec, SAMPLE_K)
                if slot in taken:
                    continue                  # skip: no colour conversion, no GPU
                taken.add(slot)

            img = frame.to_ndarray(format="bgr24")
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
            imgs.append(img)
            secs.append(sec)

            if len(imgs) >= BATCH:
                run_batch()
                while next_emit < max_sec:       # seconds before max_sec are complete
                    yield (next_emit, *summarize(per.pop(next_emit, [])))
                    next_emit += 1

        if n_frames == 0:
            raise ValueError("no frames decoded")
        if imgs:
            run_batch()
        while next_emit <= max_sec:
            yield (next_emit, *summarize(per.pop(next_emit, [])))
            next_emit += 1
    finally:
        container.close()


def video_lines(video, mp4):
    """One video -> list of JSON strings. Never raises (errors become an error line)."""
    out = []
    try:
        if not mp4:
            raise ValueError("empty video bytes")
        for sec, objects, people in process_video(mp4):
            out.append(json.dumps({"video": video, "second": sec,
                                   "objects": objects, "people": people}))
    except Exception as e:  # keep partial lines, add one error line
        out.append(json.dumps({"video": video, "error": str(e)[:200]}))
    return out


def iter_videos(path):
    """Yield (video_id, mp4_bytes, error_or_None) for each wanted video in a parquet file."""
    import pyarrow.parquet as pq

    name = Path(path).name
    row_idx = 0
    try:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8, columns=["mp4"]):
            for mp4 in batch.column("mp4").to_pylist():
                if ROW_LIMIT and row_idx >= ROW_LIMIT:
                    return
                video = f"{name}:{row_idx}"
                row_idx += 1
                if ROW_SET is not None and video not in ROW_SET:
                    continue
                yield video, mp4, None
    except Exception as e:  # damaged parquet file
        yield f"{name}:{row_idx}", None, f"parquet read failed: {str(e)[:150]}"


def process_parquet(path):
    """Single-process version: yield JSON lines for every video in one parquet file."""
    for video, mp4, err in iter_videos(path):
        if err:
            yield json.dumps({"video": video, "error": err})
        else:
            yield from video_lines(video, mp4)
