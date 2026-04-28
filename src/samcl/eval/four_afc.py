"""
4-alternative forced choice (4-AFC) on naturalistic object categories.

Ported in minimal form from mcl/object_categories_eval.py + mcl/object_categories_data_module.py:
same JSON schema and image/text trial layouts; scoring uses cosine similarity on the SAMCL
dual encoder (encode_image / encode_text) instead of mcl MultiModalModel logits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from tqdm import tqdm


FourAfcSubtype = Literal["image", "text"]


def default_four_afc_metadata_path() -> Path:
    """Default location next to packaged eval JSON (see samcl/data/object_categories_eval/)."""
    return Path(__file__).resolve().parent.parent / "data" / "object_categories_eval" / "eval_object_categories_4150.json"


def load_four_afc_trials(metadata_path: Path) -> list[dict[str, Any]]:
    with open(metadata_path, encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["data"])


def _resolve_image_path(raw: str, image_root: Path | None) -> Path:
    p = Path(raw)
    if p.is_file():
        return p
    if image_root is not None:
        q = image_root / raw
        if q.is_file():
            return q
    raise FileNotFoundError(f"4-AFC image not found: {raw!r} (image_root={image_root!r})")


@dataclass(frozen=True)
class FourAfcMetrics:
    accuracy: float
    n_trials: int
    subtype: str


@torch.no_grad()
def evaluate_four_afc(
    *,
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    metadata_path: Path,
    device: torch.device,
    subtype: FourAfcSubtype = "image",
    max_text_len: int = 77,
    max_trials: int | None = None,
    image_root: Path | None = None,
    show_progress: bool = False,
) -> FourAfcMetrics:
    """
    subtype:
      - image: 4 images (target + 3 foils), 1 target category string; pick image best matching text (GT index 0).
      - text: 1 target image, 4 category strings; pick text best matching image (GT index 0).
    """
    trials = load_four_afc_trials(metadata_path)
    n_limit = len(trials) if max_trials is None else min(int(max_trials), len(trials))
    correct = 0
    it = range(n_limit)
    if show_progress:
        it = tqdm(it, desc="4afc", leave=False)

    for i in it:
        trial = trials[i]
        if subtype == "image":
            paths = [trial["target_img_filename"], *trial["foil_img_filenames"]]
            if len(paths) != 4:
                raise ValueError(f"trial {i}: expected 4 image paths, got {len(paths)}")
            pil_images = [Image.open(_resolve_image_path(p, image_root)).convert("RGB") for p in paths]
            pv = image_processor(images=pil_images, return_tensors="pt")["pixel_values"].to(device)
            cat = str(trial["target_category"])
            text_inputs = tokenizer(
                cat,
                padding=True,
                truncation=True,
                max_length=max_text_len,
                return_tensors="pt",
            )
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            z_i = model.encode_image(pv)
            z_t = model.encode_text(input_ids, attention_mask)
            sim = (z_i @ z_t.T).squeeze(-1)
            pred = int(sim.argmax(dim=-1).item())
        else:
            path = trial["target_img_filename"]
            pil = Image.open(_resolve_image_path(path, image_root)).convert("RGB")
            pv = image_processor(images=[pil], return_tensors="pt")["pixel_values"].to(device)
            labels = [str(trial["target_category"]), *[str(x) for x in trial["foil_categories"]]]
            if len(labels) != 4:
                raise ValueError(f"trial {i}: expected 4 labels, got {len(labels)}")
            text_inputs = tokenizer(
                labels,
                padding=True,
                truncation=True,
                max_length=max_text_len,
                return_tensors="pt",
            )
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            z_i = model.encode_image(pv)
            z_t = model.encode_text(input_ids, attention_mask)
            sim = (z_i @ z_t.T).squeeze(0)
            pred = int(sim.argmax(dim=-1).item())

        if pred == 0:
            correct += 1

    acc = correct / max(1, n_limit)
    return FourAfcMetrics(accuracy=float(acc), n_trials=int(n_limit), subtype=str(subtype))
