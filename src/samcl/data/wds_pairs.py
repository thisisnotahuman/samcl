from __future__ import annotations

import glob
import hashlib
import io
import os
import pickle
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

_WDS_SEP = "\x1f"
_WDS_PREFIX = "wdsref:"

_IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
}


@dataclass(frozen=True)
class WdsImageCaptionPair:
    image_id: int
    caption_id: int
    caption: str
    image_path: str


def encode_wds_image_ref(tar_path: str, stem: str, img_ext: str) -> str:
    """Synthetic path for an image inside a tar shard (WebDataset). img_ext includes the dot, e.g. '.jpg'."""
    return f"{_WDS_PREFIX}{tar_path}{_WDS_SEP}{stem}{_WDS_SEP}{img_ext}"


def decode_wds_image_ref(path: str) -> tuple[str, str, str] | None:
    if not str(path).startswith(_WDS_PREFIX):
        return None
    rest = str(path)[len(_WDS_PREFIX) :]
    parts = rest.split(_WDS_SEP, 2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def load_pil_rgb_image_path(path: str) -> Image.Image:
    """Load RGB PIL image from a filesystem path or a wdsref: synthetic path."""
    dec = decode_wds_image_ref(path)
    if dec is None:
        return Image.open(path).convert("RGB")
    tar_path, stem, img_ext = dec
    member = f"{stem}{img_ext}"
    # Hot path during training: avoid re-opening tar files for every sample.
    tar = _get_cached_tar(tar_path)
    m = tar.getmember(member)
    data = tar.extractfile(m).read()
    return Image.open(io.BytesIO(data)).convert("RGB")


# -----------------------------------------------------------------------------
# Tarfile LRU cache (per-process; DataLoader workers are separate processes).
# -----------------------------------------------------------------------------
_TAR_CACHE_MAX_OPEN = int(os.environ.get("SAMCL_WDS_TAR_CACHE_MAX_OPEN", "8"))
_TAR_CACHE: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()


def _get_cached_tar(tar_path: str) -> tarfile.TarFile:
    tar_path = os.path.abspath(str(tar_path))
    tar = _TAR_CACHE.get(tar_path)
    if tar is not None:
        _TAR_CACHE.move_to_end(tar_path)
        return tar

    tar = tarfile.open(tar_path, "r:*")
    _TAR_CACHE[tar_path] = tar
    _TAR_CACHE.move_to_end(tar_path)
    while len(_TAR_CACHE) > max(1, _TAR_CACHE_MAX_OPEN):
        _, ev = _TAR_CACHE.popitem(last=False)
        try:
            ev.close()
        except Exception:
            pass
    return tar


def wds_shard_manifest_key(root: str, shard_glob: str) -> str:
    root = os.path.abspath(root)
    paths = sorted(glob.glob(os.path.join(root, shard_glob)))
    h = hashlib.sha256()
    for p in paths:
        st = os.stat(p)
        h.update(os.path.basename(p).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    h.update(shard_glob.encode())
    return h.hexdigest()


def discover_wds_training_shards(root: str, shard_glob: str) -> list[str]:
    root = os.path.abspath(root)
    paths = sorted(glob.glob(os.path.join(root, shard_glob)))
    if not paths:
        raise FileNotFoundError(
            f"No shards matching {shard_glob!r} under {root!r}. "
            "Check --wds_root and --wds_shard_glob."
        )
    return [os.path.abspath(p) for p in paths]


def _scan_one_tar_pair_specs(tar_path: str) -> list[tuple[str, str, str]]:
    """Return list of (stem, image_ext_with_dot, caption_text) for one WebDataset tar."""
    out: list[tuple[str, str, str]] = []
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]
        stem_to_exts: dict[str, set[str]] = {}
        for name in members:
            stem, ext = os.path.splitext(name)
            stem_to_exts.setdefault(stem, set()).add(ext)

        for stem in sorted(stem_to_exts.keys()):
            exts = stem_to_exts[stem]
            txt_ext = ".txt" if ".txt" in exts else None
            if txt_ext is None:
                continue
            img_ext = None
            for e in sorted(exts):
                if e in _IMG_EXTS or e.lower() in {x.lower() for x in _IMG_EXTS}:
                    img_ext = e
                    break
            if img_ext is None:
                continue
            cap_name = f"{stem}{txt_ext}"
            m = tar.getmember(cap_name)
            cap_bytes = tar.extractfile(m).read()
            caption = cap_bytes.decode("utf-8", errors="replace").strip()
            out.append((stem, img_ext, caption))
    return out


def _build_wds_rows(shards: list[str], max_pairs: int | None) -> list[tuple[str, str, str, str]]:
    """Rows: (tar_abs_path, stem, img_ext, caption)."""
    rows: list[tuple[str, str, str, str]] = []
    for tar_path in shards:
        for stem, img_ext, caption in _scan_one_tar_pair_specs(tar_path):
            rows.append((tar_path, stem, img_ext, caption))
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows
    return rows


def load_or_build_wds_rows(
    root: str,
    shard_glob: str,
    *,
    max_pairs: int | None,
    rebuild_index: bool,
) -> list[tuple[str, str, str, str]]:
    shards = discover_wds_training_shards(root, shard_glob)
    manifest = wds_shard_manifest_key(root, shard_glob)
    cache_dir = os.path.join(os.path.abspath(root), ".samcl_wds_index")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = str(max_pairs) if max_pairs is not None else "all"
    cache_path = os.path.join(cache_dir, f"{manifest}_{suffix}.pkl")

    if not rebuild_index and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return list(data["rows"])

    rows = _build_wds_rows(shards, max_pairs)
    with open(cache_path, "wb") as f:
        pickle.dump({"rows": rows, "manifest": manifest, "shard_glob": shard_glob}, f)
    return rows


class WdsPairsDataset(Dataset):
    """
    WebDataset tar shards (e.g. CC3M): one (image, caption) per key stem inside each tar.

    Same indexing contract as CocoPairsDataset for semantic batching.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        shard_glob: str = "cc3m-train-*.tar",
        max_pairs: int | None = None,
        rebuild_index: bool = False,
    ) -> None:
        self._root = Path(os.path.abspath(str(root_dir)))
        if not self._root.is_dir():
            raise ValueError(f"wds_root must be an existing directory; got {self._root}")

        rows = load_or_build_wds_rows(
            str(self._root),
            shard_glob,
            max_pairs=max_pairs,
            rebuild_index=rebuild_index,
        )

        path_to_image_id: dict[str, int] = {}
        next_img_id = 0
        self._pairs: list[WdsImageCaptionPair] = []
        self._image_to_caption_ids: dict[int, list[int]] = {}
        self._caption_to_image_id: dict[int, int] = {}

        for row_idx, (tar_abs, stem, img_ext, cap) in enumerate(rows):
            ref = encode_wds_image_ref(tar_abs, stem, img_ext)
            if ref not in path_to_image_id:
                path_to_image_id[ref] = next_img_id
                next_img_id += 1
            image_id = path_to_image_id[ref]
            caption_id = row_idx
            self._pairs.append(
                WdsImageCaptionPair(
                    image_id=image_id,
                    caption_id=caption_id,
                    caption=cap,
                    image_path=ref,
                )
            )
            self._caption_to_image_id[caption_id] = image_id
            self._image_to_caption_ids.setdefault(image_id, []).append(caption_id)

        for k in list(self._image_to_caption_ids.keys()):
            self._image_to_caption_ids[k].sort()

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        meta: dict[str, Any] | None = None
        if isinstance(idx, tuple) and len(idx) == 2:
            idx, meta = idx
        idx = int(idx)

        p = self._pairs[idx]
        image = load_pil_rgb_image_path(p.image_path)
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
    def pairs(self) -> list[WdsImageCaptionPair]:
        return self._pairs

    @property
    def image_ids(self) -> list[int]:
        return sorted(self._image_to_caption_ids.keys())

    def caption_ids_for_image(self, image_id: int) -> list[int]:
        return self._image_to_caption_ids.get(int(image_id), [])

    def image_id_for_caption(self, caption_id: int) -> int:
        return int(self._caption_to_image_id[int(caption_id)])
