"""
SayCam (image, caption) dataset with COCO-like interface.

Metadata JSON: list of {utterance, frame_filenames, video_filename, ...} per utterance.
Image root: directory under which each video has a subdir (e.g. .../5fps/S_20130417_0600_01/);
  only entries with video_filename starting with "S" are used.

Two modes (hyperparameter):
  - one_frame: one sample per utterance, each __getitem__ randomly picks one frame (IID-like).
  - all_frames: one sample per (frame, utterance); all frames flattened then shuffled IID.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SayCamPair:
    """One (image, caption) pair; interface aligned with CocoPair for samplers/cache."""
    image_id: int
    caption_id: int
    caption: str
    image_path: str


def _load_saycam_metadata(metadata_json: str | Path) -> list[dict[str, Any]]:
    path = Path(metadata_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"SayCam metadata must be a JSON list, got {type(data)}")
    return data


def _frame_to_video_dir(frame_filename: str) -> str:
    """From frame filename like S_20130417_0600_01_1.80.jpg -> video dir S_20130417_0600_01."""
    stem = Path(frame_filename).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _frame_full_path(images_root: Path, frame_filename: str) -> Path:
    """Full path: images_root / video_dir / frame_filename (one level under root is video name)."""
    video_dir = _frame_to_video_dir(frame_filename)
    return images_root / video_dir / frame_filename


class SayCamPairsDataset(Dataset):
    """
    COCO-like (image, caption) dataset over SayCam.

    - one_frame: len = num_utterances; each sample = one random frame from that utterance.
    - all_frames: len = total frames; each sample = (frame, utterance) with all frames kept.
    """

    def __init__(
        self,
        images_root: str | Path,
        metadata_json: str | Path,
        *,
        mode: Literal["one_frame", "all_frames"] = "one_frame",
        max_pairs: int | None = None,
        seed: int = 0,
    ) -> None:
        self.images_root = Path(images_root)
        self.metadata_json = Path(metadata_json)
        self.mode = str(mode).strip().lower()
        if self.mode not in ("one_frame", "all_frames"):
            raise ValueError(f"SayCam mode must be one_frame | all_frames, got {mode!r}")
        self._seed = int(seed)
        self._epoch = 0

        raw = _load_saycam_metadata(self.metadata_json)
        # Only keep S-prefix videos (directory names under 5fps)
        raw = [
            item
            for item in raw
            if str(item.get("video_filename", "")).strip().startswith("S")
        ]
        # Build per-utterance: utterance text + list of frame filenames
        self._utterances: list[tuple[str, list[str]]] = []
        for item in raw:
            u = (str(item.get("utterance", "")).strip(), list(item.get("frame_filenames", [])))
            if u[1]:
                self._utterances.append(u)

        if max_pairs is not None and max_pairs <= 0:
            self._utterances = []

        if self.mode == "one_frame":
            if max_pairs is not None:
                self._utterances = self._utterances[: int(max_pairs)]
            # Pairs: one per utterance; image_id = caption_id = utterance_idx, image_path = first frame (for cache)
            self._pairs: list[SayCamPair] = []
            self._image_to_caption_ids: dict[int, list[int]] = {}
            self._caption_to_image_id: dict[int, int] = {}
            for i, (caption, frame_filenames) in enumerate(self._utterances):
                first_path = str(_frame_full_path(self.images_root, frame_filenames[0]))
                self._pairs.append(
                    SayCamPair(image_id=i, caption_id=i, caption=caption, image_path=first_path)
                )
                self._image_to_caption_ids.setdefault(i, []).append(i)
                self._caption_to_image_id[i] = i
        else:
            # all_frames: flatten to (frame_path, caption, utterance_idx) for every frame
            flat: list[tuple[str, str, int]] = []
            for u_idx, (caption, frame_filenames) in enumerate(self._utterances):
                for fn in frame_filenames:
                    flat.append((str(_frame_full_path(self.images_root, fn)), caption, u_idx))
            if max_pairs is not None:
                flat = flat[: int(max_pairs)]
            self._flat_triples = flat
            self._pairs = []
            self._image_to_caption_ids = {}
            self._caption_to_image_id: dict[int, int] = {}
            seen_caption: set[int] = set()
            for linear_idx, (image_path, caption, u_idx) in enumerate(flat):
                self._pairs.append(
                    SayCamPair(
                        image_id=linear_idx,
                        caption_id=u_idx,
                        caption=caption,
                        image_path=image_path,
                    )
                )
                self._image_to_caption_ids.setdefault(linear_idx, []).append(u_idx)
                if u_idx not in seen_caption:
                    self._caption_to_image_id[u_idx] = linear_idx
                    seen_caption.add(u_idx)
            # Fallback for caption_id without any frame in flat (e.g. after max_pairs truncation)
            for u_idx in range(len(self._utterances)):
                self._caption_to_image_id.setdefault(u_idx, 0)

    def set_epoch(self, epoch: int) -> None:
        """Used in one_frame mode so each epoch gets different random frames."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        meta: dict[str, Any] | None = None
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, meta = idx
        idx = int(idx)

        p = self._pairs[idx]
        if self.mode == "one_frame":
            _, frame_filenames = self._utterances[idx]
            rng = random.Random(self._seed + self._epoch * 1977 + idx)
            fn = rng.choice(frame_filenames)
            image_path = str(_frame_full_path(self.images_root, fn))
            image = Image.open(image_path).convert("RGB")
        else:
            image_path = p.image_path
            image = Image.open(image_path).convert("RGB")

        out: dict[str, Any] = {
            "image": image,
            "caption": p.caption,
            "image_id": p.image_id,
            "caption_id": p.caption_id,
            "image_path": image_path,
            "index": idx,
        }
        if meta is not None:
            out["sampling_meta"] = meta
        return out

    @property
    def pairs(self) -> list[SayCamPair]:
        return self._pairs

    @property
    def image_ids(self) -> list[int]:
        return list(self._image_to_caption_ids.keys())

    def caption_ids_for_image(self, image_id: int) -> list[int]:
        return self._image_to_caption_ids.get(int(image_id), [])

    def image_id_for_caption(self, caption_id: int) -> int:
        return int(self._caption_to_image_id.get(int(caption_id), 0))
