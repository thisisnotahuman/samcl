import json
import logging
import math
import os
import time
from typing import Optional
from collections import deque
import subprocess
import threading

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from open_clip_train.four_afc_eval import run_four_afc_object_categories_eval
from open_clip_train.samcl_retrieval_eval import run_samcl_retrieval_eval


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _nvidia_smi_query(device_index: int):
    """Return (sm_util, mem_ctrl, mem_used, mem_total, temp_c) floats or None."""
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if p.returncode != 0:
            return None
        line = (p.stdout or "").strip().splitlines()
        if not line:
            return None
        parts = [x.strip() for x in line[0].split(",")]
        if len(parts) < 5:
            return None
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except Exception:
        return None


def _gpu_util_sampler(
    device_index: int,
    subsample_sec: float,
    window_sec: float,
    stop: threading.Event,
    out_deque,
    lock: threading.Lock,
) -> None:
    """Background thread sampling nvidia-smi into a time-stamped deque."""
    subsample = max(0.5, float(subsample_sec))
    window = max(1.0, float(window_sec))
    while not stop.wait(timeout=subsample):
        row = _nvidia_smi_query(device_index)
        if row is None:
            continue
        now = time.monotonic()
        with lock:
            out_deque.append((now, row))
            cutoff = now - window * 1.2  # keep a bit extra for stable averaging
            while out_deque and out_deque[0][0] < cutoff:
                out_deque.popleft()


def _alloc_extra_gpu_buffers(device: torch.device, reserve_mib: int):
    """Allocate two big FP32 matrices and return (A, B) or None.

    This is a hack for clusters that cancel low GPU util / low VRAM jobs.
    """
    if device.type != "cuda" or reserve_mib <= 0 or not torch.cuda.is_available():
        return None
    half = max(1, int(reserve_mib) // 2)  # MiB per matrix
    n_elems = (half * 1024 * 1024) // 4  # float32
    side = int(math.isqrt(n_elems))
    if side < 256:
        return None
    side = (side // 256) * 256  # align for matmul kernels
    try:
        a = torch.randn(side, side, device=device, dtype=torch.float32)
        b = torch.empty(side, side, device=device, dtype=torch.float32)
        mib_actual = 2 * (side * side * 4) / (1024 * 1024)
        logging.warning(
            "extra_gpu_reserve_mib=%d: allocated %.1f MiB (two %dx%d FP32) for utilization padding",
            reserve_mib,
            mib_actual,
            side,
            side,
        )
        return a, b
    except Exception as e:
        logging.warning("extra_gpu_reserve_mib=%d: buffer alloc failed (%s); disabling", reserve_mib, e)
        return None


@torch.no_grad()
def _step_extra_gpu_load(buffers) -> None:
    """Run one heavy matmul to keep GPU busy."""
    if buffers is None:
        return
    a, b = buffers
    torch.matmul(a, a, out=b)
    a.copy_(b)


def _extra_gpu_boost_worker(
    device_index: int,
    buffers,
    stop: threading.Event,
    sleep_ms: int,
    matmul_iters: int,
    duty: Optional[float],
    idle_gate: Optional[threading.Event],
) -> None:
    """Continuous matmuls on a side CUDA stream to keep GPU busy during dataloader stalls.

    Must not share the same buffer tensors with the training thread: concurrent matmul/copy on
    the same tensors is undefined behavior and often kills the worker silently.

    If idle_gate is set: only run bursts while idle_gate.is_set() (main thread blocked in DataLoader).
    Main clears idle_gate while running forward/backward.
    """
    sleep_s = max(0.0, float(sleep_ms) / 1000.0)
    iters = max(1, int(matmul_iters))
    duty_f: Optional[float] = None
    if duty is not None:
        try:
            d = float(duty)
            if 0.0 < d < 1.0:
                duty_f = d
        except (TypeError, ValueError):
            duty_f = None
    cuda_dev = torch.device("cuda", device_index)
    try:
        torch.cuda.set_device(cuda_dev)
    except Exception as e:
        logging.error("extra_gpu_util_boost: set_device(%s) failed: %s", cuda_dev, e)
        return
    stream = torch.cuda.Stream(device=cuda_dev)
    while not stop.is_set():
        if idle_gate is not None:
            while not idle_gate.is_set() and not stop.is_set():
                stop.wait(timeout=0.02)
            if stop.is_set():
                break
        try:
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(iters):
                    _step_extra_gpu_load(buffers)
            stream.synchronize()
            burst_s = time.perf_counter() - t0
        except Exception:
            logging.exception("extra_gpu_util_boost worker exiting after error")
            break
        if duty_f is not None:
            idle = burst_s * (1.0 / duty_f - 1.0)
            if idle > 0:
                stop.wait(timeout=idle)
        elif sleep_s > 0:
            stop.wait(timeout=sleep_s)


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


_LATEST_CKPT_NAME = "epoch_latest.pt"


def _save_latest_training_ckpt(
    args,
    model,
    optimizer,
    scaler,
    *,
    epoch_next: int,
    global_step_next: int,
) -> None:
    """Atomic write to checkpoints/epoch_latest.pt (master only)."""
    if not getattr(args, "save_logs", False) or not is_master(args):
        return
    if not getattr(args, "save_most_recent", False):
        return
    ckpt_dir = getattr(args, "checkpoint_path", None)
    if not ckpt_dir:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_dict = {
        "epoch": int(epoch_next),
        "name": args.name,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step_next),
    }
    if getattr(args, "scheduler_total_steps", None) is not None:
        checkpoint_dict["scheduler_total_steps"] = int(args.scheduler_total_steps)
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()
    tmp_save_path = os.path.join(ckpt_dir, "tmp.pt")
    latest_save_path = os.path.join(ckpt_dir, _LATEST_CKPT_NAME)
    torch.save(checkpoint_dict, tmp_save_path)
    os.replace(tmp_save_path, latest_save_path)


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


class _IdleBoostBatchIterator:
    """Set idle_gate while blocked in DataLoader __next__ so GPU boost may run."""

    __slots__ = ("_inner", "_idle_gate")

    def __init__(self, inner_iter, idle_gate: threading.Event):
        self._inner = inner_iter
        self._idle_gate = idle_gate

    def __iter__(self):
        return self

    def __next__(self):
        self._idle_gate.set()
        return next(self._inner)


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    tb_writer=None,
    preprocess_eval=None,
    tokenizer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    epoch_step_base = int(getattr(args, "_scheduler_step_offset", 0) or 0)
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    extra_buffers = _alloc_extra_gpu_buffers(device, int(getattr(args, "extra_gpu_reserve_mib", 0)))
    boost_on = bool(getattr(args, "extra_gpu_util_boost", False))
    boost_idle_only = bool(getattr(args, "extra_gpu_boost_idle_only", False))
    boost_stop = None
    boost_thread = None
    cuda_dev_idx = int(torch.cuda.current_device()) if device.type == "cuda" and torch.cuda.is_available() else -1
    boost_idle_gate = None
    if (
        boost_on
        and boost_idle_only
        and extra_buffers is not None
        and cuda_dev_idx >= 0
    ):
        boost_idle_gate = threading.Event()
    if (
        extra_buffers is not None
        and boost_on
        and cuda_dev_idx >= 0
    ):
        boost_stop = threading.Event()
        boost_thread = threading.Thread(
            target=_extra_gpu_boost_worker,
            args=(
                cuda_dev_idx,
                extra_buffers,
                boost_stop,
                int(getattr(args, "extra_gpu_util_boost_sleep_ms", 0)),
                int(getattr(args, "extra_gpu_boost_matmul_iters", 32)),
                getattr(args, "extra_gpu_boost_duty", None),
                boost_idle_gate,
            ),
            name="extra-gpu-util-boost",
            daemon=True,
        )
        boost_thread.start()
        duty_log = getattr(args, "extra_gpu_boost_duty", None)
        logging.warning(
            "extra_gpu_util_boost enabled (background stream, %d matmuls/loop, duty=%s, idle_only=%s). "
            "Per-step extra matmul on main thread is disabled to avoid buffer races. May slow training.",
            int(getattr(args, "extra_gpu_boost_matmul_iters", 32)),
            f"{duty_log:.3f}" if isinstance(duty_log, (int, float)) else str(duty_log),
            boost_idle_only,
        )
    elif boost_on and extra_buffers is None:
        logging.warning(
            "extra_gpu_util_boost requested but no padding buffers (set --extra-gpu-reserve-mib > 0); boost disabled."
        )
    util_window = float(getattr(args, "log_gpu_util_window_sec", 120.0))
    util_sub = float(getattr(args, "log_gpu_util_subsample_sec", 5.0))
    util_stop = None
    util_thread = None
    util_buf = None
    util_lock = threading.Lock()
    if device.type == "cuda" and torch.cuda.is_available() and util_window > 0:
        dev_idx = cuda_dev_idx if cuda_dev_idx >= 0 else int(torch.cuda.current_device())
        util_stop = threading.Event()
        util_buf = deque()
        util_thread = threading.Thread(
            target=_gpu_util_sampler,
            args=(dev_idx, util_sub, util_window, util_stop, util_buf, util_lock),
            name="gpu-util-sampler",
            daemon=True,
        )
        util_thread.start()
        logging.info(
            "gpu_util_avg enabled: window=%.0fs subsample=%.1fs (disable with --log-gpu-util-window-sec 0)",
            util_window,
            util_sub,
        )

    train_iter = iter(dataloader)
    if boost_thread is not None and boost_idle_gate is not None:
        train_iter = _IdleBoostBatchIterator(train_iter, boost_idle_gate)

    args._stop_training_after_max_steps = False
    try:
        for i, batch in enumerate(train_iter):
            if boost_idle_gate is not None:
                boost_idle_gate.clear()

            i_accum = i // args.accum_freq
            step = epoch_step_base + i_accum

            if not args.skip_scheduler:
                scheduler(step)

            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            data_time_m.update(time.time() - end)
            optimizer.zero_grad()

            # Optional: per-step matmul on reserved buffers (no background boost only).
            # When --extra-gpu-util-boost is on, the background thread owns these tensors; do not race.
            if extra_buffers is not None and not boost_on:
                _step_extra_gpu_load(extra_buffers)

            if args.accum_freq == 1:
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out["logit_scale"]
                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    losses = loss(**model_out, output_dict=True)

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
            else:
                # First, cache the features without any gradient tracking.
                with torch.no_grad():
                    with autocast():
                        model_out = model(images, texts)

                        for f in ("logit_scale", "logit_bias"):
                            model_out.pop(f, None)

                        for key, val in model_out.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                    accum_images.append(images)
                    accum_texts.append(texts)

                # If (i + 1) % accum_freq is not zero, move on to the next batch.
                if ((i + 1) % args.accum_freq) > 0:
                    # FIXME this makes data time logging unreliable when accumulating
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                optimizer.zero_grad()
                for j in range(args.accum_freq):
                    images = accum_images[j]
                    texts = accum_texts[j]
                    with autocast():
                        model_out = model(images, texts)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                        losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        del inputs
                        del inputs_no_accum
                        total_loss = sum(losses.values())
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    gn_every = int(getattr(args, "log_grad_norm_every_n_steps", 0) or 0)
                    want_clip = args.grad_clip_norm is not None
                    want_log = gn_every > 0 and is_master(args) and (step % gn_every == 0)
                    if want_clip or want_log:
                        scaler.unscale_(optimizer)
                    if want_log:
                        gnv = float(
                            torch.nn.utils.clip_grad_norm_(
                                unwrap_model(model).parameters(), float("inf")
                            )
                        )
                        logging.info("Grad norm (L2, pre-step): %.6e step=%d", gnv, int(step))
                    if want_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()

            acc_d = max(1, int(args.accum_freq))
            global_opt_steps = epoch_step_base + (i + 1) // acc_d
            ms = getattr(args, "max_steps", None)
            if ms is not None and int(ms) > 0 and global_opt_steps >= int(ms):
                args._scheduler_step_offset = int(global_opt_steps)
                args._stop_training_after_max_steps = True
                break

            # reset gradient accum, if enabled
            if args.accum_freq > 1:
                accum_images, accum_texts, accum_features = [], [], []

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

            save_every = int(getattr(args, "save_most_recent_every_n_steps", 0) or 0)
            if (
                save_every > 0
                and getattr(args, "save_most_recent", False)
                and args.save_logs
                and is_master(args)
                and optimizer is not None
                and (step + 1) > 0
                and ((step + 1) % save_every) == 0
            ):
                _save_latest_training_ckpt(
                    args,
                    model,
                    optimizer,
                    scaler,
                    epoch_next=int(epoch),
                    global_step_next=int(step + 1),
                )
                logging.info(
                    "Saved %s (intra-epoch) epoch=%d global_step_next=%d",
                    _LATEST_CKPT_NAME,
                    int(epoch),
                    int(step + 1),
                )
                if args.distributed:
                    if args.horovod:
                        import horovod.torch as hvd

                        hvd.join()
                    else:
                        torch.distributed.barrier()

            _four_every = int(getattr(args, "four_afc_eval_every_n_steps", 0) or 0)
            _four_json = getattr(args, "four_afc_eval_json", None)
            eval_type = str(getattr(args, "train_eval_type", "retrieval")).lower().strip()

            _ret_every = int(getattr(args, "samcl_retrieval_eval_every_n_steps", 0) or 0)
            if (
                eval_type in ("retrieval", "both")
                and _ret_every > 0
                and preprocess_eval is not None
                and tokenizer is not None
                and step > 0
                and (step % _ret_every) == 0
            ):
                if is_master(args):
                    was_training = model.training
                    model.eval()
                    metrics = run_samcl_retrieval_eval(
                        model=model,
                        preprocess=preprocess_eval,
                        tokenizer=tokenizer,
                        device=device,
                        args=args,
                        step=int(step),
                        epoch=int(epoch),
                        tb_writer=tb_writer,
                    )
                    if was_training:
                        model.train()
                    if metrics is not None and args.wandb and wandb is not None:
                        wandb.log(
                            {
                                "train/retrieval_i2t_r1": float(metrics.i2t_r1),
                                "train/retrieval_i2t_r5": float(metrics.i2t_r5),
                                "train/retrieval_i2t_r10": float(metrics.i2t_r10),
                                "train/retrieval_t2i_r1": float(metrics.t2i_r1),
                                "train/retrieval_t2i_r5": float(metrics.t2i_r5),
                                "train/retrieval_t2i_r10": float(metrics.t2i_r10),
                            },
                            step=int(step),
                        )
                if args.distributed:
                    if args.horovod:
                        import horovod.torch as hvd

                        hvd.join()
                    else:
                        torch.distributed.barrier()

            if (
                eval_type in ("four_afc", "both")
                and _four_every > 0
                and _four_json
                and preprocess_eval is not None
                and tokenizer is not None
                and step > 0
                and (step % _four_every) == 0
            ):
                if is_master(args):
                    was_training = model.training
                    model.eval()
                    acc = run_four_afc_object_categories_eval(
                        model,
                        preprocess_eval,
                        tokenizer,
                        device,
                        args,
                        step=step,
                        epoch=epoch,
                        tb_writer=tb_writer,
                    )
                    if was_training:
                        model.train()
                    if acc is not None and args.wandb and wandb is not None:
                        wandb.log({"train/four_afc_acc": float(acc)}, step=int(step))
                if args.distributed:
                    if args.horovod:
                        import horovod.torch as hvd

                        hvd.join()
                    else:
                        torch.distributed.barrier()

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count_in_epoch = i_accum + 1
            global_step_count = epoch_step_base + i_accum + 1
            if is_master(args) and (
                i_accum % args.log_every_n_steps == 0 or batch_count_in_epoch == num_batches_per_epoch
            ):
                batch_size = len(images)
                # Log epoch progress consistently: samples within this epoch vs samples_per_epoch.
                # (global_step_count is still used for tb/wandb step + scheduler.)
                num_samples = batch_count_in_epoch * batch_size * args.accum_freq * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count_in_epoch / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                logit_scale_scalar = logit_scale.item()
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                        for loss_name, loss_m in losses_m.items()
                    ]
                )
                _bt = (
                    max(batch_time_m.avg, 1e-9)
                    if batch_time_m.count > 0
                    else max(batch_time_m.val, 1e-9)
                )
                samples_per_second = args.accum_freq * args.batch_size * args.world_size / _bt
                samples_per_second_per_gpu = args.accum_freq * args.batch_size / _bt
                gpu_util_log = ""
                if util_buf is not None and util_window > 0:
                    with util_lock:
                        now = time.monotonic()
                        cutoff = now - util_window
                        rows = [r for (t, r) in util_buf if t >= cutoff]
                    if rows:
                        sm = sum(r[0] for r in rows) / len(rows)
                        mc = sum(r[1] for r in rows) / len(rows)
                        mu = sum(r[2] for r in rows) / len(rows)
                        mt = sum(r[3] for r in rows) / len(rows)
                        te = sum(r[4] for r in rows) / len(rows)
                        gpu_util_log = (
                            f" GPU(util_mean%={sm:.1f} mem_ctrl%={mc:.1f} "
                            f"vram_mean_MiB={mu:.0f}/{mt:.0f} temp_mean_C={te:.1f})"
                        )
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log + gpu_util_log
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log_data.update({name: val.val for name, val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}

                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)

                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
        # end for
        if not getattr(args, "_stop_training_after_max_steps", False):
            args._scheduler_step_offset = int(epoch_step_base + num_batches_per_epoch)
    finally:
        if boost_stop is not None:
            boost_stop.set()
        if boost_thread is not None:
            boost_thread.join(timeout=5.0)
        if util_stop is not None:
            util_stop.set()
        if util_thread is not None:
            util_thread.join(timeout=float(util_sub) + 5.0)


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # unwrap DDP for single process eval
        if args.distributed and not args.horovod:
            model = model.module
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
