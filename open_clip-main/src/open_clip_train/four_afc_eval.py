"""
Object-categories 4-alternative forced choice (4AFC) evaluation for OpenCLIP-style models.

JSON schema matches mcl ``object_categories_data_module`` eval metadata:
each trial contains ``target_img_filename``, ``foil_img_filenames`` (length 3),
``target_category``, and ``foil_categories`` (length 3).

**text** mode (default): one target image + four text options (target + 3 foils).
Correct iff the target text (index 0) has highest CLIP similarity to the image.

**image** mode: four images (target + 3 foils) + target category as text.
Correct iff the target image (index 0) has highest similarity to that text.

Reference: ``mcl/object_categories_eval.py::eval_1_out_of_4``.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _load_trials(json_path: str) -> list[dict[str, Any]]:
    path = Path(json_path).expanduser()
    if not path.is_file():
        logging.warning("four_afc_eval: JSON not found: %s", path)
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data")
    if not isinstance(data, list):
        logging.warning("four_afc_eval: expected top-level key 'data' as a list in %s", path)
        return []
    return data


def run_four_afc_object_categories_eval(
    model: torch.nn.Module,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    args: Any,
    *,
    step: int,
    epoch: int,
    tb_writer: Optional[Any] = None,
) -> Optional[float]:
    """
    Returns overall accuracy in [0, 1], or None if skipped / no trials.
    """
    json_path = getattr(args, "four_afc_eval_json", None) or ""
    if not str(json_path).strip():
        return None

    trials = _load_trials(str(json_path))
    if not trials:
        return None

    max_trials = int(getattr(args, "four_afc_eval_max_trials", 0) or 0)
    rng = random.Random(int(getattr(args, "seed", 0)) + int(epoch) * 1000003 + int(step))

    indices = list(range(len(trials)))
    if max_trials > 0 and len(indices) > max_trials:
        rng.shuffle(indices)
        indices = indices[:max_trials]

    eval_type = str(getattr(args, "four_afc_eval_type", "text")).lower().strip()
    if eval_type in ("t", "txt"):
        eval_type = "text"
    if eval_type in ("i", "img"):
        eval_type = "image"

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    um = _unwrap_model(model)

    correct = 0
    total = 0
    skipped = 0

    for idx in indices:
        trial = trials[int(idx)]
        try:
            if eval_type == "text":
                img_path = trial["target_img_filename"]
                labels = [trial["target_category"]] + list(trial["foil_categories"])
                pil = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(pil).unsqueeze(0)
                if input_dtype is not None:
                    image_tensor = image_tensor.to(device=device, dtype=input_dtype, non_blocking=True)
                else:
                    image_tensor = image_tensor.to(device=device, non_blocking=True)
                texts = tokenizer(labels)
                texts = texts.to(device=device, non_blocking=True)
                with torch.inference_mode():
                    with autocast():
                        image_features = um.encode_image(image_tensor, normalize=True)
                        text_features = um.encode_text(texts, normalize=True)
                        logit_scale = um.logit_scale.exp()
                        logits = logit_scale * (image_features @ text_features.T)
                pred = int(torch.argmax(logits, dim=-1).item())
                ok = pred == 0
            elif eval_type == "image":
                paths = [trial["target_img_filename"]] + list(trial["foil_img_filenames"])
                pils = [Image.open(p).convert("RGB") for p in paths]
                image_tensor = torch.stack([preprocess(p) for p in pils])
                if input_dtype is not None:
                    image_tensor = image_tensor.to(device=device, dtype=input_dtype, non_blocking=True)
                else:
                    image_tensor = image_tensor.to(device=device, non_blocking=True)
                cap = trial["target_category"]
                texts = tokenizer([cap])
                texts = texts.to(device=device, non_blocking=True)
                with torch.inference_mode():
                    with autocast():
                        image_features = um.encode_image(image_tensor, normalize=True)
                        text_features = um.encode_text(texts, normalize=True)
                        logit_scale = um.logit_scale.exp()
                        logits = logit_scale * (text_features @ image_features.T)
                pred = int(torch.argmax(logits, dim=-1).item())
                ok = pred == 0
            else:
                logging.error("four_afc_eval: unknown --four-afc-eval-type %r", eval_type)
                return None
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                logging.warning("four_afc_eval: skip trial %s: %s", idx, e)
            continue

        total += 1
        if ok:
            correct += 1

    if total == 0:
        logging.warning("four_afc_eval: no successful trials (skipped=%d)", skipped)
        return None

    acc = correct / total
    logging.info(
        "4AFC eval epoch=%d step=%d type=%s trials=%d/%d acc=%.4f skipped_files=%d json=%s",
        epoch,
        step,
        eval_type,
        total,
        len(indices),
        acc,
        skipped,
        json_path,
    )
    if tb_writer is not None:
        tb_writer.add_scalar("train/four_afc_acc", acc, step)
    return acc
