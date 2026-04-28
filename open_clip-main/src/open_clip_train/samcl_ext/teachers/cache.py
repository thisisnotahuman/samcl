from __future__ import annotations

import io
import json
import logging
import math
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from tqdm import tqdm

from open_clip_train.samcl_ext.coco_pairs import CocoPairsDataset
from open_clip_train.samcl_ext.pairs_dataset import CsvPairsDataset, WdsPairsDataset
from open_clip_train.samcl_ext.teachers.image_teacher import FrozenImageTeacher
from open_clip_train.samcl_ext.teachers.text_teacher import FrozenTextTeacher


def _pil_rgb_from_bytes(img_bytes: bytes) -> Any:
    """Module-level for ThreadPoolExecutor (JPEG decode often releases GIL)."""
    from PIL import Image

    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


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
    Builds and loads teacher embeddings for semantic batching (SAMCL-style).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        dataset: Union[CsvPairsDataset, WdsPairsDataset, CocoPairsDataset],
        text_teacher: FrozenTextTeacher,
        image_teacher: FrozenImageTeacher,
        prep_debug_timings: bool = False,
        prep_wds_decode_workers: int = 0,
    ) -> None:
        self.paths = TeacherCachePaths(Path(cache_dir))
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.text_teacher = text_teacher
        self.image_teacher = image_teacher
        self.prep_debug_timings = bool(prep_debug_timings)
        self.prep_wds_decode_workers = max(0, int(prep_wds_decode_workers))

        self.caption_ids: torch.Tensor | None = None
        self.caption_emb: torch.Tensor | None = None
        self.caption_id_to_row: dict[int, int] | None = None

        self.image_ids: torch.Tensor | None = None
        self.image_emb: torch.Tensor | None = None
        self.image_id_to_row: dict[int, int] | None = None

    def _save_meta(self, meta: dict) -> None:
        with self.paths.meta_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _tlog(self, label: str, t0: float) -> None:
        if self.prep_debug_timings:
            logging.info("[prep_timing] %s: %.2fs", label, time.perf_counter() - t0)

    def load_if_exists(self) -> bool:
        if not self.paths.caption_emb_pt.exists() or not self.paths.image_emb_pt.exists():
            return False

        cap = torch.load(self.paths.caption_emb_pt, map_location="cpu")
        img = torch.load(self.paths.image_emb_pt, map_location="cpu")

        self.caption_ids = cap["ids"].long()
        self.caption_emb = cap["emb"]
        self.caption_id_to_row = {int(k): int(v) for k, v in cap["id_to_row"].items()}

        self.image_ids = img["ids"].long()
        self.image_emb = img["emb"]
        self.image_id_to_row = {int(k): int(v) for k, v in img["id_to_row"].items()}
        if self.prep_debug_timings:
            logging.info("[prep_timing] teacher_cache: skipped build (loaded existing .pt)")
        return True

    def build(self) -> None:
        t_build0 = time.perf_counter()
        caption_ids: list[int] = []
        captions: list[str] = []
        t0 = time.perf_counter()
        for p in self.dataset.pairs:
            caption_ids.append(int(p.caption_id))
            captions.append(p.caption)
        self._tlog("teacher_cache scan_pairs_for_captions", t0)

        self.caption_ids = torch.tensor(caption_ids, dtype=torch.long)
        self.caption_id_to_row = {int(cid): i for i, cid in enumerate(caption_ids)}

        cap_chunks: list[torch.Tensor] = []
        chunk_size = 4096
        t0 = time.perf_counter()
        for s in tqdm(range(0, len(captions), chunk_size), desc="teacher_text_embed", leave=False):
            e = self.text_teacher.encode(captions[s : s + chunk_size]).detach().cpu()
            cap_chunks.append(e)
        cap_emb = torch.cat(cap_chunks, dim=0) if cap_chunks else torch.empty((0, 0))
        self.caption_emb = cap_emb.to(dtype=torch.float16)
        self._tlog("teacher_cache text_teacher_encode", t0)

        t0 = time.perf_counter()
        torch.save(
            {
                "ids": self.caption_ids,
                "emb": self.caption_emb,
                "id_to_row": self.caption_id_to_row,
            },
            self.paths.caption_emb_pt,
        )
        self._tlog("teacher_cache write_caption_teacher_pt", t0)

        image_ids = self.dataset.image_ids
        self.image_ids = torch.tensor(image_ids, dtype=torch.long)
        self.image_id_to_row = {int(iid): i for i, iid in enumerate(image_ids)}

        t0 = time.perf_counter()
        image_id_to_path: dict[int, str] = {}
        for p in self.dataset.pairs:
            if p.image_id not in image_id_to_path:
                image_id_to_path[int(p.image_id)] = p.image_path

        from PIL import Image

        from open_clip_train.samcl_ext.pairs_dataset import decode_wds_image_ref

        bs = int(self.image_teacher.cfg.batch_size)
        wds_by_tar: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        regular_items: list[tuple[int, str]] = []
        for iid in image_ids:
            pth = image_id_to_path[int(iid)]
            dec = decode_wds_image_ref(pth)
            if dec is not None:
                tar_abs, stem, img_ext = dec
                wds_by_tar[tar_abs].append((int(iid), stem, img_ext))
            else:
                regular_items.append((int(iid), pth))
        self._tlog("teacher_cache image_paths_and_wds_buckets", t0)

        id_to_vec: dict[int, torch.Tensor] = {}

        batch_iids: list[int] = []
        batch_pils: list[Any] = []

        def _flush_regular() -> None:
            nonlocal batch_iids, batch_pils
            if not batch_pils:
                return
            e = self.image_teacher.encode_images(batch_pils).detach().cpu()
            for j, iid in enumerate(batch_iids):
                id_to_vec[iid] = e[j]
            batch_iids = []
            batch_pils = []

        t0 = time.perf_counter()
        for iid, pth in tqdm(regular_items, desc="teacher_image_embed_fs", leave=False):
            im = Image.open(pth).convert("RGB")
            batch_iids.append(iid)
            batch_pils.append(im)
            if len(batch_pils) >= bs:
                _flush_regular()
        _flush_regular()
        self._tlog("teacher_cache image_teacher_fs_paths (decode+encode)", t0)

        total_wds_batches = sum(math.ceil(len(items) / bs) for items in wds_by_tar.values())
        if total_wds_batches > 0:
            dev = getattr(self.image_teacher, "device", None)
            hint = "GPU is much faster than CPU for this step." if (
                dev is not None and str(dev).startswith("cuda")
            ) else "On CPU this step can take many hours for full CC3M."
            logging.info(
                "teacher_image_embed_wds: %d shards, ~%d ResNet batches (batch_size=%d). %s",
                len(wds_by_tar),
                total_wds_batches,
                bs,
                hint,
            )
        wds_decode_s = 0.0
        wds_encode_s = 0.0
        t0 = time.perf_counter()
        n_dec_workers = self.prep_wds_decode_workers
        if n_dec_workers > 1 and total_wds_batches > 0:
            logging.info(
                "teacher_image_embed_wds: parallel JPEG decode with %d threads (tar read stays sequential)",
                n_dec_workers,
            )
        with tqdm(total=total_wds_batches, desc="teacher_image_embed_wds", leave=True) as pbar:
            executor: ThreadPoolExecutor | None = None
            if n_dec_workers > 1:
                executor = ThreadPoolExecutor(max_workers=n_dec_workers)
            try:
                for tar_abs in sorted(wds_by_tar.keys()):
                    items = wds_by_tar[tar_abs]
                    with tarfile.open(tar_abs, "r:*") as tar:
                        for s in range(0, len(items), bs):
                            chunk = items[s : s + bs]
                            batch_iids = []
                            raws: list[bytes] = []
                            t_dec = time.perf_counter()
                            for iid, stem, img_ext in chunk:
                                member = f"{stem}{img_ext}"
                                m = tar.getmember(member)
                                raw = tar.extractfile(m).read()
                                raws.append(raw)
                                batch_iids.append(iid)
                            if executor is not None and len(raws) > 1:
                                batch_pils = list(executor.map(_pil_rgb_from_bytes, raws))
                            else:
                                batch_pils = [
                                    Image.open(io.BytesIO(raw)).convert("RGB") for raw in raws
                                ]
                            wds_decode_s += time.perf_counter() - t_dec
                            t_enc = time.perf_counter()
                            e = self.image_teacher.encode_images(batch_pils).detach().cpu()
                            wds_encode_s += time.perf_counter() - t_enc
                            for j, iid in enumerate(batch_iids):
                                id_to_vec[iid] = e[j]
                            pbar.update(1)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
        self._tlog("teacher_cache image_teacher_wds total (tar read + decode + encode)", t0)
        if self.prep_debug_timings and total_wds_batches > 0:
            tot = wds_decode_s + wds_encode_s
            pct_d = 100.0 * wds_decode_s / tot if tot > 0 else 0.0
            pct_e = 100.0 * wds_encode_s / tot if tot > 0 else 0.0
            logging.info(
                "[prep_timing] teacher_cache image_teacher_wds split: decode %.2fs (%.0f%%) | "
                "encode_images %.2fs (%.0f%%) | batches=%d",
                wds_decode_s,
                pct_d,
                wds_encode_s,
                pct_e,
                total_wds_batches,
            )

        t0 = time.perf_counter()
        row_tensors = [id_to_vec[int(iid)] for iid in image_ids]
        self.image_emb = torch.stack(row_tensors, dim=0).to(dtype=torch.float16)

        torch.save(
            {
                "ids": self.image_ids,
                "emb": self.image_emb,
                "id_to_row": self.image_id_to_row,
            },
            self.paths.image_emb_pt,
        )
        self._tlog("teacher_cache image_emb_stack_order_write_pt", t0)

        self._save_meta(
            {
                "num_pairs": len(self.dataset),
                "num_captions": int(self.caption_ids.numel()),
                "num_images": int(self.image_ids.numel()),
            }
        )
        if self.prep_debug_timings:
            logging.info("[prep_timing] teacher_cache build() total: %.2fs", time.perf_counter() - t_build0)

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
