from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from samcl.data.coco_pairs import CocoPairsDataset
from samcl.data.saycam_pairs import SayCamPairsDataset
from samcl.data.wds_pairs import WdsPairsDataset, load_pil_rgb_image_path
from samcl.teachers.image_teacher import FrozenImageTeacher
from samcl.teachers.text_teacher import FrozenTextTeacher


@dataclass(frozen=True)
class TeacherCachePaths:
    root: Path

    @property
    def meta_json(self) -> Path:
        return self.root / "meta.json"

    @property
    def caption_emb_pt(self) -> Path:
        return self.root / "caption_teacher_emb.pt"

    @property
    def image_emb_pt(self) -> Path:
        return self.root / "image_teacher_emb.pt"


class TeacherEmbeddingCache:
    """
    Builds and loads teacher embeddings:
      - caption embeddings for every caption_id in dataset
      - image embeddings for every unique image_id in dataset

    Embeddings are L2-normalized, so cosine similarity == dot product.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        dataset: CocoPairsDataset | SayCamPairsDataset | WdsPairsDataset,
        text_teacher: FrozenTextTeacher | object,
        image_teacher: FrozenImageTeacher | object,
        expected_teacher_tag: str | None = None,
    ) -> None:
        self.paths = TeacherCachePaths(Path(cache_dir))
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.text_teacher = text_teacher
        self.image_teacher = image_teacher
        self._expected_teacher_tag = expected_teacher_tag

        self.caption_ids: torch.Tensor | None = None
        self.caption_emb: torch.Tensor | None = None
        self.caption_id_to_row: dict[int, int] | None = None

        self.image_ids: torch.Tensor | None = None
        self.image_emb: torch.Tensor | None = None
        self.image_id_to_row: dict[int, int] | None = None

    def _save_meta(self, meta: dict) -> None:
        with self.paths.meta_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _teacher_tag_matches_disk(self) -> bool:
        expected = self._expected_teacher_tag
        if not self.paths.meta_json.is_file():
            if expected is None:
                return True
            return False

        with self.paths.meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tag = meta.get("teacher_tag")

        if expected is None:
            if tag is not None and str(tag).startswith("student_mirrored"):
                return False
            return True
        return tag == expected

    def load_if_exists(self) -> bool:
        if not self.paths.caption_emb_pt.exists() or not self.paths.image_emb_pt.exists():
            return False
        if not self._teacher_tag_matches_disk():
            return False

        cap = torch.load(self.paths.caption_emb_pt, map_location="cpu")
        img = torch.load(self.paths.image_emb_pt, map_location="cpu")

        self.caption_ids = cap["ids"].long()
        self.caption_emb = cap["emb"]
        self.caption_id_to_row = {int(k): int(v) for k, v in cap["id_to_row"].items()}

        self.image_ids = img["ids"].long()
        self.image_emb = img["emb"]
        self.image_id_to_row = {int(k): int(v) for k, v in img["id_to_row"].items()}
        return True

    def build(self) -> None:
        # Caption embeddings (unique by caption_id)
        caption_ids: list[int] = []
        captions: list[str] = []
        for p in self.dataset.pairs:
            caption_ids.append(int(p.caption_id))
            captions.append(p.caption)

        # COCO caption annotation ids are unique already, but keep stable order anyway.
        self.caption_ids = torch.tensor(caption_ids, dtype=torch.long)
        self.caption_id_to_row = {int(cid): i for i, cid in enumerate(caption_ids)}

        # Chunked encoding to avoid materializing a huge embedding tensor on GPU at once.
        cap_chunks: list[torch.Tensor] = []
        chunk_size = 4096
        for s in tqdm(range(0, len(captions), chunk_size), desc="teacher_text_embed", leave=False):
            e = self.text_teacher.encode(captions[s : s + chunk_size]).detach().cpu()
            cap_chunks.append(e)
        cap_emb = torch.cat(cap_chunks, dim=0) if cap_chunks else torch.empty((0, 0))
        self.caption_emb = cap_emb.to(dtype=torch.float16)

        torch.save(
            {
                "ids": self.caption_ids,
                "emb": self.caption_emb,
                "id_to_row": self.caption_id_to_row,
            },
            self.paths.caption_emb_pt,
        )

        # Image embeddings (unique images)
        image_ids = self.dataset.image_ids
        self.image_ids = torch.tensor(image_ids, dtype=torch.long)
        self.image_id_to_row = {int(iid): i for i, iid in enumerate(image_ids)}

        # Compute image embeddings by loading one representative path per image_id.
        # We select the first pair that references the image.
        image_id_to_path: dict[int, str] = {}
        for p in self.dataset.pairs:
            if p.image_id not in image_id_to_path:
                image_id_to_path[int(p.image_id)] = p.image_path

        embs: list[torch.Tensor] = []
        bs = max(1, int(self.image_teacher.cfg.batch_size))
        decode_workers = int(os.environ.get("SAMCL_TEACHER_CACHE_DECODE_WORKERS", "0"))
        n_img = len(image_ids)
        n_steps = math.ceil(n_img / bs) if n_img > 0 else 0

        def _load_one(path: str):
            return load_pil_rgb_image_path(path)

        for start in tqdm(range(0, n_img, bs), total=n_steps, desc="teacher_image_embed_batches", leave=False):
            chunk_ids = image_ids[start : start + bs]
            paths = [image_id_to_path[int(iid)] for iid in chunk_ids]
            if decode_workers > 1:
                nw = min(int(decode_workers), len(paths))
                with ThreadPoolExecutor(max_workers=max(1, nw)) as ex:
                    batch = list(ex.map(_load_one, paths))
            else:
                batch = [_load_one(p) for p in paths]

            e = self.image_teacher.encode_images(batch).detach().cpu()
            embs.append(e)

        self.image_emb = (torch.cat(embs, dim=0) if embs else torch.empty((0, 0))).to(dtype=torch.float16)

        torch.save(
            {
                "ids": self.image_ids,
                "emb": self.image_emb,
                "id_to_row": self.image_id_to_row,
            },
            self.paths.image_emb_pt,
        )

        meta: dict = {
            "num_pairs": len(self.dataset),
            "num_captions": int(self.caption_ids.numel()),
            "num_images": int(self.image_ids.numel()),
        }
        if self._expected_teacher_tag is not None:
            meta["teacher_tag"] = self._expected_teacher_tag
        self._save_meta(meta)

    def ensure_built(self) -> None:
        if self.load_if_exists():
            return
        self.build()

    def get_caption_emb(self, caption_id: int) -> torch.Tensor:
        assert self.caption_emb is not None and self.caption_id_to_row is not None
        row = self.caption_id_to_row[int(caption_id)]
        return self.caption_emb[row]

    def get_image_emb(self, image_id: int) -> torch.Tensor:
        assert self.image_emb is not None and self.image_id_to_row is not None
        row = self.image_id_to_row[int(image_id)]
        return self.image_emb[row]

