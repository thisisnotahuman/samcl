"""
Build WebDataset/CSV pair index and teacher embedding caches only (no CLIP training).

Use a CPU-only Slurm job if you want to avoid occupying a GPU for hours; for much faster
image encoding (ResNet), use ``--device cuda`` on a GPU node (see
``scripts/slurm_prep_semantic_wds_cache_gpu.SBATCH``).

Example::

    python -m open_clip_train.prep_semantic_cache --semantic-sampling \\
      --train-data /path/to/cc3m_wds --semantic-wds-shard-glob 'cc3m-train-*.tar' \\
      --device cpu --semantic-sampler-compute-device cpu \\
      --cache-dir /path/cache --teacher-cache-dir /path/cache/cc3m_shared_teacher \\
      --teacher-text-model sentence-transformers/all-MiniLM-L6-v2 \\
      --teacher-image-arch resnet50

Add ``--prep-debug-timings`` for phase timing logs (dataset load, text embed, WDS decode vs encode).
"""
from __future__ import annotations

import os

# Before any HF tokenizers use + subprocess (nvidia-smi) in this process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import multiprocessing
import subprocess
import sys
import threading
import time
from typing import Any, Optional, Tuple

import torch

from open_clip_train.data_semantic import build_semantic_teacher_caches_only, coco_semantic_image_root, semantic_uses_coco_dataset
from open_clip_train.params import parse_args


def _cuda_device_index(device_str: str) -> int:
    s = str(device_str).strip().lower()
    if s == "cuda":
        return 0
    if s.startswith("cuda:"):
        return int(s.split(":", 1)[1])
    return 0


def _prep_gpu_util_boost_worker(
    device_index: int,
    matmul_size: int,
    stop_event: Any,
) -> None:
    """Separate process: continuous GEMM so the GPU shows high utilization during I/O gaps."""
    import torch as th

    th.cuda.set_device(device_index)
    dev = th.device(f"cuda:{device_index}")
    a = th.empty(matmul_size, matmul_size, device=dev, dtype=th.float32)
    b = th.empty(matmul_size, matmul_size, device=dev, dtype=th.float32)
    c = th.empty(matmul_size, matmul_size, device=dev, dtype=th.float32)
    while not stop_event.is_set():
        th.randn(matmul_size, matmul_size, out=a)
        th.randn(matmul_size, matmul_size, out=b)
        th.mm(a, b, out=c)


def _nvidia_smi_query_stats(device_index: int) -> Optional[Tuple[float, float, float, float, float]]:
    """Returns (sm_util%, mem_ctrl%, mem_used_MiB, mem_total_MiB, temp_C) or None."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        line = (proc.stdout or "").strip().splitlines()
        if proc.returncode != 0 or not line:
            return None
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) < 5:
            return None
        return (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def _prep_gpu_util_log_worker(
    device_index: int,
    interval_sec: float,
    subsample_sec: float,
    stop: threading.Event,
) -> None:
    """Background thread: each log line is the mean over the previous ``interval_sec`` window."""
    subsample = max(0.2, min(float(subsample_sec), float(interval_sec)))

    while True:
        deadline = time.monotonic() + float(interval_sec)
        sm_vals: list[float] = []
        mc_vals: list[float] = []
        mu_vals: list[float] = []
        mt_vals: list[float] = []
        temp_vals: list[float] = []

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            step = min(subsample, remaining)
            if stop.wait(timeout=step):
                deadline = time.monotonic()
                break
            row = _nvidia_smi_query_stats(device_index)
            if row is None:
                continue
            sm, mc, mu, mt, te = row
            sm_vals.append(sm)
            mc_vals.append(mc)
            mu_vals.append(mu)
            mt_vals.append(mt)
            temp_vals.append(te)

        if sm_vals:
            logging.info(
                "prep_gpu_stats_avg [cuda:%s] window=%.0fs n=%d | sm_mean%% mem_ctrl_mean%% "
                "mem_used_mean_MiB mem_total_mean_MiB temp_mean_C | %.1f %.1f %.0f %.0f %.1f",
                device_index,
                interval_sec,
                len(sm_vals),
                sum(sm_vals) / len(sm_vals),
                sum(mc_vals) / len(mc_vals),
                sum(mu_vals) / len(mu_vals),
                sum(mt_vals) / len(mt_vals),
                sum(temp_vals) / len(temp_vals),
            )

        if stop.is_set():
            break


def main() -> int:
    args = parse_args(sys.argv[1:])
    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    if not getattr(args, "semantic_sampling", False):
        logging.error("Add --semantic-sampling and data paths (WDS directory, CSV/TSV, or COCO JSON + image root).")
        return 1
    if semantic_uses_coco_dataset(args):
        try:
            coco_semantic_image_root(args)
        except ValueError as e:
            logging.error("COCO prep: %s", e)
            return 1
    elif not args.train_data:
        logging.error("--train-data is required (unless using --coco-captions-json with --coco-images-dir).")
        return 1

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    boost = bool(getattr(args, "prep_gpu_util_boost", False))
    boost_size = int(getattr(args, "prep_gpu_util_boost_size", 4096))
    proc: Optional[multiprocessing.Process] = None
    stop_event: Any = None

    if boost:
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            logging.error("--prep-gpu-util-boost requires --device cuda and a visible CUDA device.")
            return 1
        if boost_size < 512:
            logging.error("--prep-gpu-util-boost-size must be >= 512.")
            return 1

    log_interval = float(getattr(args, "prep_gpu_log_util_interval", 120.0))
    log_sub = float(getattr(args, "prep_gpu_log_util_subsample_sec", 5.0))
    util_log_stop: Optional[threading.Event] = None
    util_log_thread: Optional[threading.Thread] = None
    if (
        str(args.device).startswith("cuda")
        and torch.cuda.is_available()
        and log_interval > 0
    ):
        util_log_stop = threading.Event()
        idx_log = _cuda_device_index(args.device)
        util_log_thread = threading.Thread(
            target=_prep_gpu_util_log_worker,
            args=(idx_log, log_interval, log_sub, util_log_stop),
            name="prep-gpu-util-log",
            daemon=True,
        )
        util_log_thread.start()
        logging.info(
            "prep_gpu_stats_avg: cuda:%s every %.0fs (mean over window, subsample %.2fs; interval 0 disables)",
            idx_log,
            log_interval,
            log_sub,
        )

    if boost:
        ctx = multiprocessing.get_context("spawn")
        stop_event = ctx.Event()
        idx = _cuda_device_index(args.device)
        proc = ctx.Process(
            target=_prep_gpu_util_boost_worker,
            args=(idx, boost_size, stop_event),
            name="prep-gpu-util-boost",
            daemon=True,
        )
        proc.start()
        logging.warning(
            "prep-gpu-util-boost: side GEMM process on cuda:%s (size=%dx%d). "
            "Stops when prep finishes; may compete with teacher encoding.",
            idx,
            boost_size,
            boost_size,
        )

    try:
        build_semantic_teacher_caches_only(args)
    finally:
        if util_log_stop is not None:
            util_log_stop.set()
        if util_log_thread is not None:
            util_log_thread.join(timeout=float(log_interval) + 30.0)
        if proc is not None and stop_event is not None:
            stop_event.set()
            proc.join(timeout=15.0)
            if proc.is_alive():
                logging.warning("prep-gpu-util-boost: terminating side process.")
                proc.terminate()
                proc.join(timeout=5.0)

    logging.info("prep_semantic_cache: finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
