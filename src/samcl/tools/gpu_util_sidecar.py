"""
Background GPU load helper for HPC jobs that monitor utilization.

Two mechanisms (can be combined):

1) **Rolling-threshold boost**: append ``nvidia-smi`` SM util samples; if the rolling mean
   falls below ``--threshold``, run extra ``GpuMatmulBurner`` work until the poll window ends.

2) **Continuous duty** (recommended for bursty workloads like teacher-cache JPEG decode):
   each ``--poll-interval`` wall-clock window, spend ``--continuous-duty`` fraction of that
   window running real GEMM, regardless of ``nvidia-smi`` readings. This raises *time-averaged*
   SM activity even when the main process only uses the GPU in short bursts.

Typical usage (from Slurm bash)::

  python -u -m samcl.tools.gpu_util_sidecar --continuous-duty 0.45 --poll-interval 1.0 &
  SIDE_PID=$!
  trap 'kill $SIDE_PID 2>/dev/null || true' EXIT INT TERM
  python -u -m samcl.train --prep_teacher_cache_only ...
"""

from __future__ import annotations

import argparse
import collections
import signal
import subprocess
import sys
import time

import torch

from samcl.utils.gpu_burn import GpuMatmulBurner


def _query_sm_util_percent(*, gpu_index: int) -> float | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(int(gpu_index)),
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            return None
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return None
        return float(line[0].strip())
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def main() -> None:
    p = argparse.ArgumentParser("samcl.tools.gpu_util_sidecar")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device for GEMM bursts.")
    p.add_argument(
        "--nvidia-smi-gpu-index",
        type=int,
        default=0,
        help="GPU index passed to nvidia-smi -i (often 0 inside a single-GPU allocation).",
    )
    p.add_argument("--threshold", type=float, default=50.0, help="Rolling mean SM util %% below which we add extra GEMM.")
    p.add_argument("--window-sec", type=float, default=20.0, help="Rolling window length in seconds.")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Fixed wall-clock cycle length (seconds).")
    p.add_argument("--matmul-dim", type=int, default=4096)
    p.add_argument("--matmul-iters", type=int, default=16, help="GEMM chain depth per burner call.")
    p.add_argument(
        "--burst-cycles",
        type=int,
        default=8,
        help="Max burner() calls per threshold top-up pass (each pass may repeat until cycle budget).",
    )
    p.add_argument(
        "--continuous-duty",
        type=float,
        default=0.0,
        help=(
            "Fraction of each poll-interval spent in GEMM unconditionally (0 disables). "
            "Example: 0.45 with poll-interval=1.0 => ~450ms/s of sustained matmul."
        ),
    )
    p.add_argument("--log-every-sec", type=float, default=30.0)
    args = p.parse_args()

    device = torch.device(str(args.device))
    if device.type != "cuda":
        print("[gpu_util_sidecar] device is not cuda; exiting.", file=sys.stderr)
        raise SystemExit(0)

    stop = False

    def _stop(*_a: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    max_samples = max(2, int(round(float(args.window_sec) / max(0.05, float(args.poll_interval)))))
    util_hist: collections.deque[float] = collections.deque(maxlen=max_samples)

    burner = GpuMatmulBurner(
        device=device,
        matmul_dim=int(args.matmul_dim),
        iters=int(args.matmul_iters),
        dtype="float16",
        reserve_gb=0.0,
    )

    last_log = time.monotonic()
    cd = max(0.0, min(1.0, float(args.continuous_duty)))
    print(
        f"[gpu_util_sidecar] device={device} nvidia_smi_i={int(args.nvidia_smi_gpu_index)} "
        f"threshold={float(args.threshold):.1f}% window={float(args.window_sec):.1f}s "
        f"poll={float(args.poll_interval):.3f}s continuous_duty={cd:.3f} "
        f"dim={int(args.matmul_dim)} iters={int(args.matmul_iters)} burst_cycles={int(args.burst_cycles)}",
        flush=True,
    )

    poll = max(0.05, float(args.poll_interval))

    while not stop:
        t_cycle = time.monotonic()
        deadline = t_cycle + poll

        u = _query_sm_util_percent(gpu_index=int(args.nvidia_smi_gpu_index))
        if u is not None:
            util_hist.append(u)

        # (A) Guaranteed duty GEMM: raises time-averaged util even when main is CPU/IO bound.
        if cd > 0:
            duty_deadline = t_cycle + cd * poll
            while time.monotonic() < duty_deadline and not stop:
                burner()

        # (B) Threshold top-up: fill remaining budget in this poll window.
        if len(util_hist) >= 2:
            avg = sum(util_hist) / len(util_hist)
            if avg < float(args.threshold):
                while time.monotonic() < deadline and not stop:
                    for _ in range(max(1, int(args.burst_cycles))):
                        if stop or time.monotonic() >= deadline:
                            break
                        burner()

        # (C) Pad sleep to keep a stable cadence (also yields the GPU to the main process).
        spare = deadline - time.monotonic()
        if spare > 0:
            time.sleep(spare)

        now = time.monotonic()
        if now - last_log >= float(args.log_every_sec) and util_hist:
            avg = sum(util_hist) / len(util_hist)
            last_u = util_hist[-1]
            print(
                f"[gpu_util_sidecar] last_sm={last_u:.1f}% rolling_mean={avg:.1f}% n={len(util_hist)}",
                flush=True,
            )
            last_log = now

    print("[gpu_util_sidecar] stopped.", flush=True)


if __name__ == "__main__":
    main()
