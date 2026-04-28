"""
CSV pair dataset + SAMCL semantic BatchSampler for OpenCLIP training.
"""
from __future__ import annotations

import hashlib
import logging
import os
from functools import partial
from typing import Any, Union

import torch
from torch.utils.data import DataLoader

from open_clip_train.data import DataInfo, SharedEpoch
from open_clip_train.samcl_ext.coco_pairs import CocoPairsDataset
from open_clip_train.samcl_ext.pairs_dataset import (
    CsvPairsDataset,
    WdsPairsDataset,
    semantic_train_data_is_wds_dir,
)
from open_clip_train.samcl_ext.relations import SemanticRelationConfig, SemanticRelationOracle
from open_clip_train.samcl_ext.sampling.batch_samplers import (
    BinaryMix,
    RandomBatchSampler,
    SemanticBatchSampler,
    SemanticMix,
)
from open_clip_train.samcl_ext.teachers.cache import TeacherEmbeddingCache
from open_clip_train.samcl_ext.teachers.image_teacher import FrozenImageTeacher, ImageTeacherConfig
from open_clip_train.samcl_ext.teachers.text_teacher import FrozenTextTeacher, TextTeacherConfig

PairsDataset = Union[CsvPairsDataset, WdsPairsDataset, CocoPairsDataset]


def semantic_uses_coco_dataset(args: Any) -> bool:
    return bool(getattr(args, "coco_captions_json", None))


def coco_semantic_image_root(args: Any) -> str:
    root = getattr(args, "coco_images_dir", None) or args.train_data
    if not root:
        raise ValueError("COCO semantic mode requires --coco-images-dir or --train-data (COCO train images directory).")
    return os.path.abspath(str(root))


class SemanticTrainDataInfo(DataInfo):
    """Extends DataInfo so set_epoch updates SemanticBatchSampler."""

    def set_epoch(self, epoch: int) -> None:
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        bs = getattr(self.dataloader, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
        if self.sampler is not None:
            from torch.utils.data.distributed import DistributedSampler

            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(epoch)


def _teacher_cache_subdir(args: Any) -> str:
    if semantic_uses_coco_dataset(args):
        key = (
            f"coco|{coco_semantic_image_root(args)}|{os.path.abspath(str(args.coco_captions_json))}|"
            f"{getattr(args, 'semantic_max_pairs', None)}"
        )
    elif args.train_data and semantic_train_data_is_wds_dir(args.train_data):
        key = (
            f"wds|{os.path.abspath(args.train_data)}|{getattr(args, 'semantic_wds_shard_glob', '')}|"
            f"{getattr(args, 'semantic_max_pairs', None)}"
        )
    else:
        key = (
            f"{os.path.abspath(args.train_data)}|{args.csv_img_key}|{args.csv_caption_key}|"
            f"{getattr(args, 'semantic_max_pairs', None)}"
        )
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    base = getattr(args, "teacher_cache_dir", None) or os.path.join(args.cache_dir or ".cache", "samcl_teacher")
    return os.path.join(base, h)


def _collate_openclip(batch: list, preprocess: Any, tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]
    image_tensors = torch.stack([preprocess(im) for im in images])
    text_tensors = tokenizer(captions)
    return image_tensors, text_tensors


def _ensure_teacher_cache(teacher_cache: TeacherEmbeddingCache, args: Any) -> None:
    if not getattr(args, "distributed", False):
        teacher_cache.ensure_built()
        return
    import torch.distributed as dist

    from open_clip_train.distributed import is_master

    if is_master(args):
        teacher_cache.ensure_built()
    dist.barrier()
    if not is_master(args):
        if not teacher_cache.load_if_exists():
            logging.warning("Teacher cache missing on non-master; rebuilding (should not happen).")
            teacher_cache.ensure_built()


def make_semantic_pairs_dataset(args: Any) -> PairsDataset:
    """Load COCO, CSV/TSV, or WebDataset index (may create ``.openclip_semantic_index`` on first WDS run)."""
    if semantic_uses_coco_dataset(args):
        return CocoPairsDataset(
            coco_semantic_image_root(args),
            args.coco_captions_json,
            max_pairs=getattr(args, "semantic_max_pairs", None),
        )
    if not args.train_data:
        raise ValueError("semantic sampling requires --train-data (CSV/TSV file or WDS directory).")

    if semantic_train_data_is_wds_dir(args.train_data):
        return WdsPairsDataset(
            args.train_data,
            shard_glob=str(getattr(args, "semantic_wds_shard_glob", "cc3m-train-*.tar")),
            max_pairs=getattr(args, "semantic_max_pairs", None),
            rebuild_index=bool(getattr(args, "semantic_wds_rebuild_index", False)),
        )
    return CsvPairsDataset(
        args.train_data,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        max_pairs=getattr(args, "semantic_max_pairs", None),
    )


def build_semantic_teacher_caches_only(args: Any) -> str:
    """
    Build only: pair table + teacher embedding caches (no DataLoader / training).

    Use on a **CPU** Slurm job: set ``--device cpu`` and ``--semantic-sampler-compute-device cpu``.
    Returns the resolved teacher cache directory (``.../<hash>``) for logging.
    """
    import time as time_mod

    dbg = bool(getattr(args, "prep_debug_timings", False))
    t0_all = time_mod.perf_counter()

    t_ds = time_mod.perf_counter()
    dataset = make_semantic_pairs_dataset(args)
    if dbg:
        logging.info("[prep_timing] make_semantic_pairs_dataset: %.2fs", time_mod.perf_counter() - t_ds)

    device = torch.device(args.device)
    t_dev = device
    if str(getattr(args, "semantic_sampler_compute_device", "cuda")) == "cpu":
        t_dev = torch.device("cpu")

    t_te = time_mod.perf_counter()
    text_teacher = FrozenTextTeacher(
        TextTeacherConfig(
            model_name=args.teacher_text_model,
            batch_size=int(getattr(args, "teacher_text_batch_size", 128)),
        ),
        device=t_dev,
    )
    image_teacher = FrozenImageTeacher(
        ImageTeacherConfig(
            arch=args.teacher_image_arch,
            clip_model_name=args.teacher_image_model or "openai/clip-vit-base-patch32",
            batch_size=int(getattr(args, "teacher_image_batch_size", 64)),
        ),
        device=t_dev,
    )
    if dbg:
        logging.info("[prep_timing] init FrozenTextTeacher + FrozenImageTeacher: %.2fs", time_mod.perf_counter() - t_te)

    cache_dir = _teacher_cache_subdir(args)
    os.makedirs(cache_dir, exist_ok=True)
    teacher_cache = TeacherEmbeddingCache(
        cache_dir,
        dataset=dataset,
        text_teacher=text_teacher,
        image_teacher=image_teacher,
        prep_debug_timings=dbg,
        prep_wds_decode_workers=int(getattr(args, "prep_wds_decode_workers", 0)),
    )

    # Prep job is always single-process; skip distributed barrier logic.
    teacher_cache.ensure_built()
    if dbg:
        logging.info("[prep_timing] build_semantic_teacher_caches_only total: %.2fs", time_mod.perf_counter() - t0_all)
    logging.info("Semantic prep done. Teacher cache dir: %s", cache_dir)
    logging.info("Dataset size (pairs): %s", len(dataset))
    return cache_dir


def get_coco_random_baseline_train_data(args: Any, preprocess_train: Any, tokenizer: Any) -> SemanticTrainDataInfo:
    """
    COCO (image, caption) pairs with **uniform random** batches: no teacher models, no semantic mining.

    Use with ``--coco-captions-json`` and ``--train-data`` (or ``--coco-images-dir``) pointing at COCO train images.
    """
    if not semantic_uses_coco_dataset(args):
        raise ValueError("get_coco_random_baseline_train_data requires --coco-captions-json")
    img_root = coco_semantic_image_root(args)
    if not os.path.isdir(img_root):
        raise ValueError(f"COCO image root is not a directory: {img_root}")
    cap_json = str(getattr(args, "coco_captions_json", "") or "")
    if not os.path.isfile(os.path.abspath(cap_json)):
        raise ValueError(f"--coco-captions-json must be an existing file: {cap_json}")

    dataset = make_semantic_pairs_dataset(args)
    rel_seed = int(args.seed) + int(getattr(args, "rank", 0))
    batch_sampler = RandomBatchSampler(
        dataset,
        batch_size=int(args.batch_size),
        drop_last=True,
        seed=rel_seed,
    )
    collate_fn = partial(_collate_openclip, preprocess=preprocess_train, tokenizer=tokenizer)

    wt = int(getattr(args, "dataloader_worker_torch_threads", 1) or 1)

    def _seed_worker(worker_id: int) -> None:
        if wt > 0:
            try:
                torch.set_num_threads(wt)
            except Exception:
                pass
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        import random as _random

        np.random.seed(worker_seed)
        _random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    nw = int(args.workers)
    dl_kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker if nw > 0 else None,
        generator=gen,
        persistent_workers=nw > 0,
    )
    if nw > 0:
        pf = int(getattr(args, "prefetch_factor", 2) or 2)
        if pf < 2:
            raise ValueError("prefetch_factor must be >= 2 when workers > 0")
        dl_kwargs["prefetch_factor"] = pf
    loader = DataLoader(**dl_kwargs)

    loader.num_samples = len(dataset)
    loader.num_batches = len(loader)

    shared_epoch = SharedEpoch(epoch=0)
    logging.info(
        "COCO random baseline dataloader: pairs=%d batches/epoch=%d (no semantic sampling)",
        len(dataset),
        len(loader),
    )
    return SemanticTrainDataInfo(dataloader=loader, sampler=None, shared_epoch=shared_epoch)


def get_semantic_train_data(args: Any, preprocess_train: Any, tokenizer: Any) -> SemanticTrainDataInfo:
    """
    Build train DataLoader with SAMCL semantic batch sampling.

    - If ``--coco-captions-json`` is set, loads COCO captions + images (``--coco-images-dir`` or ``--train-data``).
    - Elif ``args.train_data`` is a directory of WebDataset shards, loads CC3M-style tars.
    - Otherwise expects a CSV/TSV with columns ``--csv-img-key`` / ``--csv-caption-key``.
    """
    dataset = make_semantic_pairs_dataset(args)

    device = torch.device(args.device)
    t_dev = device
    if str(getattr(args, "semantic_sampler_compute_device", "cuda")) == "cpu":
        t_dev = torch.device("cpu")

    text_teacher = FrozenTextTeacher(
        TextTeacherConfig(
            model_name=args.teacher_text_model,
            batch_size=int(getattr(args, "teacher_text_batch_size", 128)),
        ),
        device=t_dev,
    )
    image_teacher = FrozenImageTeacher(
        ImageTeacherConfig(
            arch=args.teacher_image_arch,
            clip_model_name=args.teacher_image_model or "openai/clip-vit-base-patch32",
            batch_size=int(getattr(args, "teacher_image_batch_size", 64)),
        ),
        device=t_dev,
    )

    cache_dir = _teacher_cache_subdir(args)
    os.makedirs(cache_dir, exist_ok=True)
    teacher_cache = TeacherEmbeddingCache(
        cache_dir,
        dataset=dataset,
        text_teacher=text_teacher,
        image_teacher=image_teacher,
        prep_debug_timings=bool(getattr(args, "prep_debug_timings", False)),
        prep_wds_decode_workers=int(getattr(args, "prep_wds_decode_workers", 0)),
    )
    _ensure_teacher_cache(teacher_cache, args)

    oracle = SemanticRelationOracle(
        dataset=dataset,
        teacher_cache=teacher_cache,
        cfg=SemanticRelationConfig(
            text_sim_threshold=float(args.text_sim_threshold),
            image_sim_threshold=float(args.image_sim_threshold),
            use_image_topk=bool(getattr(args, "use_image_topk", False)),
            image_topk=int(getattr(args, "image_topk", 50)),
            relation_mode=str(args.semantic_relation_mode),
        ),
    )

    rel_mode = str(args.semantic_relation_mode).lower().strip()
    seed = int(args.seed) + int(getattr(args, "rank", 0))

    if rel_mode == "full":
        mix = SemanticMix(args.mix_r1, args.mix_r2, args.mix_r3, args.mix_r4)
        batch_sampler = SemanticBatchSampler(
            dataset,
            oracle=oracle,
            mix=mix,
            relation_mode=rel_mode,
            mode=str(args.semantic_sampler_mode),
            num_anchors=int(args.semantic_sampler_num_anchors),
            min_anchor_matches=args.semantic_sampler_min_anchor_matches,
            global_num_candidates=int(args.semantic_sampler_global_num_candidates),
            num_blocks=int(args.semantic_sampler_num_blocks),
            compute_device=str(args.semantic_sampler_compute_device),
            cache_teacher_on_device=bool(args.semantic_sampler_cache_teacher_on_device),
            batch_size=args.batch_size,
            drop_last=True,
            seed=seed,
            max_tries=int(args.sampler_max_tries),
        )
    else:
        mix = BinaryMix(similar=float(args.mix_similar), different=float(args.mix_different))
        batch_sampler = SemanticBatchSampler(
            dataset,
            oracle=oracle,
            mix=mix,
            relation_mode=rel_mode,
            mode=str(args.semantic_sampler_mode),
            num_anchors=int(args.semantic_sampler_num_anchors),
            min_anchor_matches=args.semantic_sampler_min_anchor_matches,
            global_num_candidates=int(args.semantic_sampler_global_num_candidates),
            num_blocks=int(args.semantic_sampler_num_blocks),
            compute_device=str(args.semantic_sampler_compute_device),
            cache_teacher_on_device=bool(args.semantic_sampler_cache_teacher_on_device),
            batch_size=args.batch_size,
            drop_last=True,
            seed=seed,
            max_tries=int(args.sampler_max_tries),
        )

    collate_fn = partial(_collate_openclip, preprocess=preprocess_train, tokenizer=tokenizer)

    wt = int(getattr(args, "dataloader_worker_torch_threads", 1) or 1)

    def _seed_worker(worker_id: int) -> None:
        if wt > 0:
            try:
                torch.set_num_threads(wt)
            except Exception:
                pass
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        import random as _random

        np.random.seed(worker_seed)
        _random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    nw = int(args.workers)
    dl_kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker if nw > 0 else None,
        generator=gen,
        persistent_workers=nw > 0,
    )
    if nw > 0:
        pf = int(getattr(args, "prefetch_factor", 2) or 2)
        if pf < 2:
            raise ValueError("prefetch_factor must be >= 2 when workers > 0")
        dl_kwargs["prefetch_factor"] = pf
    loader = DataLoader(**dl_kwargs)

    loader.num_samples = len(dataset)
    loader.num_batches = len(loader)

    shared_epoch = SharedEpoch(epoch=0)
    return SemanticTrainDataInfo(dataloader=loader, sampler=None, shared_epoch=shared_epoch)
