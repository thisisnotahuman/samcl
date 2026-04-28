from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CocoPair:
    image_id: int
    caption_id: int  # COCO annotation id for the caption
    caption: str
    image_path: str


class CocoPairsDataset(Dataset):
    """
    Each item is a *positive* COCO (image, caption) pair:
      - image_id: COCO image id
      - caption_id: COCO annotation id
      - caption: raw caption text
      - image: PIL.Image (RGB)

    This dataset does not implement any sampling logic. Sampling is done via BatchSampler.
    """

    def __init__(
        self,
        coco_images_dir: str | Path,
        coco_captions_json: str | Path,
        *,
        max_pairs: int | None = None,
    ) -> None:
        self.coco_images_dir = Path(coco_images_dir)
        self.coco_captions_json = Path(coco_captions_json)
        self.coco = COCO(str(self.coco_captions_json))

        ann_ids = list(self.coco.anns.keys())
        if max_pairs is not None:
            ann_ids = ann_ids[: int(max_pairs)]

        self._pairs: list[CocoPair] = []
        self._image_to_caption_ids: dict[int, list[int]] = {}
        self._caption_to_image_id: dict[int, int] = {}

        for ann_id in ann_ids:
            ann = self.coco.anns[ann_id]
            image_id = int(ann["image_id"])
            caption_id = int(ann["id"])
            caption = str(ann["caption"])

            img = self.coco.imgs[image_id]
            file_name = img["file_name"]
            image_path = str(self.coco_images_dir / file_name)

            self._pairs.append(
                CocoPair(
                    image_id=image_id,
                    caption_id=caption_id,
                    caption=caption,
                    image_path=image_path,
                )
            )
            self._caption_to_image_id[caption_id] = image_id
            self._image_to_caption_ids.setdefault(image_id, []).append(caption_id)

        # Stable iteration order
        for k in list(self._image_to_caption_ids.keys()):
            self._image_to_caption_ids[k].sort()

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        """
        Supports either:
          - idx: int
          - idx: (int, meta_dict) where meta_dict is produced by a sampler
        """
        meta: dict[str, Any] | None = None
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, meta = idx
        idx = int(idx)

        p = self._pairs[idx]
        image = Image.open(p.image_path).convert("RGB")
        out = {
            "image": image,
            "caption": p.caption,
            "image_id": p.image_id,
            "caption_id": p.caption_id,
            "image_path": p.image_path,
            "index": idx,
        }
        if meta is not None:
            out["sampling_meta"] = meta
        return out

    @property
    def pairs(self) -> list[CocoPair]:
        return self._pairs

    @property
    def image_ids(self) -> list[int]:
        # Unique image ids present in this dataset slice
        return list(self._image_to_caption_ids.keys())

    def caption_ids_for_image(self, image_id: int) -> list[int]:
        return self._image_to_caption_ids.get(int(image_id), [])

    def image_id_for_caption(self, caption_id: int) -> int:
        return int(self._caption_to_image_id[int(caption_id)])

