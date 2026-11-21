"""python run.py --input <folder of .parquet files> --output results.jsonl"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pipeline


def make_pool(workers, n_gpus):
    ctx = mp.get_context("spawn")      # never fork after CUDA/pyarrow threads exist
    counter = ctx.Value("i", 0)
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                               initializer=pipeline.init_worker,
                               initargs=(counter, n_gpus))


def err_line(video, msg):
    return json.dumps({"video": video, "error": msg})


def run_parallel(files, out, workers):
    n_gpus = pipeline.count_gpus()
    holder = {"pool": make_pool(workers, n_gpus)}
    pending = {}        # future -> (video, mp4, pool it was submitted to)
    suspects = []       # videos that were in flight when a worker died

    def submit(video, mp4):
        pool = holder["pool"]
        pending[pool.submit(pipeline.video_lines, video, mp4)] = (video, mp4, pool)

    def resolve_suspects():
        # A crash fails every in-flight video, but only one is the culprit.
        # Re-run each suspect alone in a throwaway 1-worker pool to find out which.
        while suspects:
            video, mp4 = suspects.pop()
            solo = make_pool(1, n_gpus)
            try:
                lines = solo.submit(pipeline.video_lines, video, mp4).result()
            except BrokenProcessPool:
                lines = [err_line(video, "worker crashed while decoding")]
            except Exception as e:
                lines = [err_line(video, f"worker failed: {str(e)[:150]}")]
            finally:
                solo.shutdown(wait=False, cancel_futures=True)
            for ln in lines:
                out.write(ln + "\n")

    def collect():
        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        for fut in done:
            video, mp4, pool = pending.pop(fut)
            try:
                lines = fut.result()
            except BrokenProcessPool:
                if pool is holder["pool"]:             # first sight of this crash
                    holder["pool"] = make_pool(workers, n_gpus)
                    pool.shutdown(wait=False, cancel_futures=True)
                suspects.append((video, mp4))
                continue
            except Exception as e:
                lines = [err_line(video, f"worker failed: {str(e)[:150]}")]
            for ln in lines:
                out.write(ln + "\n")
        resolve_suspects()

    for f in files:
        for video, mp4, err in pipeline.iter_videos(f):
            if err:
                out.write(err_line(video, err) + "\n")
                continue
            while len(pending) >= window_size(workers):
                collect()
            submit(video, mp4)
        out.flush()
        print(f"{f.name}: read", file=sys.stderr)
    while pending:
        collect()
    holder["pool"].shutdown()


def window_size(workers):
    return workers * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl file")
    args = ap.parse_args()

    files = sorted(Path(args.input).glob("*.parquet"))
    if not files:
        print(f"no .parquet files in {args.input}", file=sys.stderr)

    workers = int(os.environ.get("WORKERS", min(os.cpu_count() or 1, 6)))
    with open(args.output, "w", encoding="utf-8") as out:
        if workers <= 0:
            for f in files:
                n = 0
                for line in pipeline.process_parquet(f):
                    out.write(line + "\n")
                    n += 1
                out.flush()
                print(f"{f.name}: {n} lines", file=sys.stderr)
        else:
            run_parallel(files, out, workers)


if __name__ == "__main__":
    main()
