#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from samcl.data.coco_pairs import CocoPairsDataset
from samcl.teachers.cache import TeacherEmbeddingCache
from samcl.teachers.image_teacher import FrozenImageTeacher, ImageTeacherConfig
from samcl.teachers.text_teacher import FrozenTextTeacher, TextTeacherConfig
from samcl.utils.device import get_device
from samcl.utils.seed import seed_all


def _sample_pair_indices(rng: np.random.Generator, n_items: int, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.integers(0, n_items, size=n_pairs, dtype=np.int64)
    b = rng.integers(0, n_items, size=n_pairs, dtype=np.int64)
    # avoid identical pairs by shifting b where needed
    same = a == b
    if same.any():
        b[same] = (b[same] + 1) % n_items
    return a, b


def _cos_sim_samples(emb: torch.Tensor, idx_a: np.ndarray, idx_b: np.ndarray, *, batch: int = 200000) -> np.ndarray:
    """
    emb: [N, D], assumed L2-normalized.
    returns sims: [num_pairs]
    """
    emb = emb.float().contiguous()
    out = np.empty((idx_a.shape[0],), dtype=np.float32)
    n = idx_a.shape[0]
    for s in range(0, n, batch):
        e = min(n, s + batch)
        a = torch.from_numpy(idx_a[s:e]).long()
        b = torch.from_numpy(idx_b[s:e]).long()
        sims = (emb[a] * emb[b]).sum(dim=-1)
        out[s:e] = sims.detach().cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser("plot_teacher_similarity")
    p.add_argument("--coco_images_dir", type=str, required=True)
    p.add_argument("--coco_captions_json", type=str, required=True)
    p.add_argument("--max_pairs", type=int, default=None, help="debug: limit dataset pairs")

    p.add_argument("--cache_dir", type=str, default="./cache", help="same cache_dir as training")
    p.add_argument("--out_dir", type=str, default="./plots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)

    # teacher config (must match what you used for embeddings)
    p.add_argument("--teacher_text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--teacher_image_arch", type=str, default="resnet50", choices=["resnet50", "clip"])
    p.add_argument("--teacher_image_model", type=str, default="openai/clip-vit-base-patch32")

    # sampling budget for distributions
    p.add_argument("--num_caption_pairs", type=int, default=300000)
    p.add_argument("--num_image_pairs", type=int, default=300000)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--recompute", action="store_true", help="ignore cached similarity samples and recompute")
    args = p.parse_args()

    seed_all(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset
    dataset = CocoPairsDataset(
        args.coco_images_dir,
        args.coco_captions_json,
        max_pairs=args.max_pairs,
    )

    # 2) Build/load teacher embedding cache
    text_teacher = FrozenTextTeacher(TextTeacherConfig(model_name=args.teacher_text_model), device=device)
    image_teacher = FrozenImageTeacher(
        ImageTeacherConfig(arch=args.teacher_image_arch, clip_model_name=args.teacher_image_model),
        device=device,
    )
    teacher_cache = TeacherEmbeddingCache(
        args.cache_dir,
        dataset=dataset,
        text_teacher=text_teacher,
        image_teacher=image_teacher,
    )
    teacher_cache.ensure_built()

    assert teacher_cache.caption_emb is not None and teacher_cache.image_emb is not None
    cap_emb = teacher_cache.caption_emb
    img_emb = teacher_cache.image_emb

    # 3) Similarity sample cache (depends on teacher + dataset size + seed + n_pairs)
    sample_dir = Path(args.cache_dir) / "similarity_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed": int(args.seed),
        "num_pairs_in_dataset": int(len(dataset)),
        "num_unique_images": int(len(dataset.image_ids)),
        "teacher_text_model": str(args.teacher_text_model),
        "teacher_image_arch": str(args.teacher_image_arch),
        "teacher_image_model": str(args.teacher_image_model),
        "num_caption_pairs": int(args.num_caption_pairs),
        "num_image_pairs": int(args.num_image_pairs),
    }
    meta_path = sample_dir / "meta.json"
    cap_path = sample_dir / "caption_caption_cosine.npy"
    img_path = sample_dir / "image_image_cosine.npy"

    def meta_matches() -> bool:
        if not meta_path.exists() or not cap_path.exists() or not img_path.exists():
            return False
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        keys = [
            "seed",
            "num_pairs_in_dataset",
            "num_unique_images",
            "teacher_text_model",
            "teacher_image_arch",
            "teacher_image_model",
            "num_caption_pairs",
            "num_image_pairs",
        ]
        return all(old.get(k) == meta.get(k) for k in keys)

    if (not args.recompute) and meta_matches():
        cap_sims = np.load(cap_path)
        img_sims = np.load(img_path)
    else:
        # caption-caption similarity distribution
        ia, ib = _sample_pair_indices(rng, cap_emb.shape[0], int(args.num_caption_pairs))
        cap_sims = _cos_sim_samples(cap_emb, ia, ib)

        # image-image similarity distribution
        ja, jb = _sample_pair_indices(rng, img_emb.shape[0], int(args.num_image_pairs))
        img_sims = _cos_sim_samples(img_emb, ja, jb)

        np.save(cap_path, cap_sims)
        np.save(img_path, img_sims)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 4) Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    axes[0].hist(cap_sims, bins=int(args.bins), density=True, alpha=0.85)
    axes[0].set_title("Caption–Caption cosine (teacher)")
    axes[0].set_xlabel("cosine similarity")
    axes[0].set_ylabel("density")

    axes[1].hist(img_sims, bins=int(args.bins), density=True, alpha=0.85, color="tab:orange")
    axes[1].set_title("Image–Image cosine (teacher)")
    axes[1].set_xlabel("cosine similarity")
    axes[1].set_ylabel("density")

    fig.suptitle("Teacher similarity distributions (sampled pairs)")
    fig.tight_layout()

    png_path = out_dir / "teacher_similarity_distributions.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"[ok] saved: {png_path}")
    print(f"[ok] sample cache: {sample_dir}")


if __name__ == "__main__":
    main()

