"""
SAMCL-style retrieval evaluation (R@1/5/10) for OpenCLIP models.

This mirrors `src/samcl/eval/retrieval.py`, but runs on an OpenCLIP-style model
(`encode_image`, `encode_text`) and uses the `open_clip_train.samcl_ext` COCO dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Subset

from open_clip_train.samcl_ext.coco_pairs import CocoPairsDataset


@dataclass(frozen=True)
class RetrievalMetrics:
    i2t_r1: float
    i2t_r5: float
    i2t_r10: float
    t2i_r1: float
    t2i_r5: float
    t2i_r10: float


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _batched_encode(
    *,
    model: torch.nn.Module,
    dataset: CocoPairsDataset,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    # Keep deterministic slice without shuffling (matches prior SAMCL eval semantics).
    eval_dataset = dataset
    if max_pairs > 0 and max_pairs < len(dataset):
        eval_dataset = Subset(dataset, range(int(max_pairs)))

    # Use identity collate to avoid default_collate issues with PIL Images.
    loader = DataLoader(
        eval_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )
    um = _unwrap_model(model)
    um.eval()

    image_embs: list[torch.Tensor] = []
    text_embs: list[torch.Tensor] = []
    image_ids: list[int] = []
    caption_ids: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            images = [x["image"] for x in batch]
            captions = [str(x["caption"]) for x in batch]

            pixel_values = torch.stack([preprocess(im) for im in images], dim=0).to(device, non_blocking=True)
            texts = tokenizer(captions).to(device, non_blocking=True)

            img = um.encode_image(pixel_values, normalize=True).detach().cpu()  # [B, D]
            txt = um.encode_text(texts, normalize=True).detach().cpu()  # [B, D]

            image_embs.append(img)
            text_embs.append(txt)
            image_ids.extend([int(x["image_id"]) for x in batch])
            caption_ids.extend([int(x["caption_id"]) for x in batch])

    I = torch.cat(image_embs, dim=0)  # [N, D]
    T = torch.cat(text_embs, dim=0)  # [N, D]
    return I, T, image_ids, caption_ids


def _recall_at_k_from_rankings(rankings: torch.Tensor, gt_sets: list[set[int]], k: int) -> float:
    # rankings: [Nq, Nc] int64 indices of candidates sorted by similarity desc
    k = int(k)
    hits = 0
    for q in range(rankings.shape[0]):
        topk = rankings[q, :k].tolist()
        if any(int(x) in gt_sets[q] for x in topk):
            hits += 1
    return float(hits) / float(rankings.shape[0]) if rankings.shape[0] > 0 else 0.0


@torch.no_grad()
def run_samcl_retrieval_eval(
    *,
    model: torch.nn.Module,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    args: Any,
    step: int,
    epoch: int,
    tb_writer: Optional[Any] = None,
) -> Optional[RetrievalMetrics]:
    """
    Returns RetrievalMetrics or None if dataset unsupported / disabled.

    Requirements:
      - COCO semantic mode: --coco-captions-json + --train-data (images dir) or --coco-images-dir
    """
    coco_json = getattr(args, "coco_captions_json", None)
    if not coco_json:
        logging.info("SAMCL retrieval eval skipped: not a COCO run (missing --coco-captions-json).")
        return None

    image_root = getattr(args, "coco_images_dir", None) or getattr(args, "train_data", None)
    if not image_root:
        logging.info("SAMCL retrieval eval skipped: missing COCO images root.")
        return None

    max_pairs = int(getattr(args, "samcl_retrieval_eval_max_pairs", 5000) or 5000)
    batch_size = int(getattr(args, "samcl_retrieval_eval_batch_size", 128) or 128)

    # Cache dataset object on args (master-only usage).
    ds = getattr(args, "_samcl_retrieval_eval_dataset", None)
    if ds is None:
        ds = CocoPairsDataset(str(image_root), str(coco_json), max_pairs=max_pairs if max_pairs > 0 else None)
        setattr(args, "_samcl_retrieval_eval_dataset", ds)

    I, T, image_ids, caption_ids = _batched_encode(
        model=model,
        dataset=ds,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_pairs=max_pairs,
    )

    # Similarities (cosine because normalized).
    sims = I @ T.T  # [N, N]
    # Rankings
    i2t_rank = torch.argsort(sims, dim=1, descending=True)  # for each image query, rank captions
    t2i_rank = torch.argsort(sims, dim=0, descending=True).T  # for each caption query, rank images

    # Build GT sets within this eval slice.
    # i2t GT: for each image query (row i), all caption indices whose caption_id is GT for that image_id.
    # t2i GT: for each caption query (row i), the single image index whose image_id matches GT.
    capid_to_col = {int(cid): j for j, cid in enumerate(caption_ids)}

    i2t_gt: list[set[int]] = []
    for iid in image_ids:
        cap_ids = ds.caption_ids_for_image(int(iid))
        cols = {int(capid_to_col[c]) for c in cap_ids if int(c) in capid_to_col}
        i2t_gt.append(cols)

    # t2i: same as src/samcl/eval/retrieval.py — GT is any gallery *row* whose COCO image_id matches.
    # (Many rows can share the same image_id; a dict image_id->row would be wrong.)
    t2i_gt: list[set[int]] = []
    for cid in caption_ids:
        gt_iid = int(ds.image_id_for_caption(int(cid)))
        rows = {int(j) for j in range(len(image_ids)) if int(image_ids[j]) == gt_iid}
        t2i_gt.append(rows)

    metrics = RetrievalMetrics(
        i2t_r1=_recall_at_k_from_rankings(i2t_rank, i2t_gt, 1),
        i2t_r5=_recall_at_k_from_rankings(i2t_rank, i2t_gt, 5),
        i2t_r10=_recall_at_k_from_rankings(i2t_rank, i2t_gt, 10),
        t2i_r1=_recall_at_k_from_rankings(t2i_rank, t2i_gt, 1),
        t2i_r5=_recall_at_k_from_rankings(t2i_rank, t2i_gt, 5),
        t2i_r10=_recall_at_k_from_rankings(t2i_rank, t2i_gt, 10),
    )

    logging.info(
        "Retrieval eval (SAMCL) epoch=%d step=%d max_pairs=%d i2t[R@1/5/10]=%.3f/%.3f/%.3f t2i[R@1/5/10]=%.3f/%.3f/%.3f",
        int(epoch),
        int(step),
        int(max_pairs),
        metrics.i2t_r1,
        metrics.i2t_r5,
        metrics.i2t_r10,
        metrics.t2i_r1,
        metrics.t2i_r5,
        metrics.t2i_r10,
    )

    if tb_writer is not None:
        tb_writer.add_scalar("train/retrieval_i2t_r1", metrics.i2t_r1, int(step))
        tb_writer.add_scalar("train/retrieval_i2t_r5", metrics.i2t_r5, int(step))
        tb_writer.add_scalar("train/retrieval_i2t_r10", metrics.i2t_r10, int(step))
        tb_writer.add_scalar("train/retrieval_t2i_r1", metrics.t2i_r1, int(step))
        tb_writer.add_scalar("train/retrieval_t2i_r5", metrics.t2i_r5, int(step))
        tb_writer.add_scalar("train/retrieval_t2i_r10", metrics.t2i_r10, int(step))

    return metrics

