from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from samcl.data.coco_pairs import CocoPairsDataset
from samcl.data.saycam_pairs import SayCamPairsDataset
from samcl.data.wds_pairs import WdsPairsDataset


@dataclass(frozen=True)
class RetrievalMetrics:
    i2t_r1: float
    i2t_r5: float
    i2t_r10: float
    t2i_r1: float
    t2i_r5: float
    t2i_r10: float


@torch.no_grad()
def evaluate_retrieval(
    *,
    model: Any,
    dataset: CocoPairsDataset | SayCamPairsDataset | WdsPairsDataset,
    collate_fn: Any,
    device: torch.device,
    batch_size: int = 128,
    max_pairs: int | None = 5000,
) -> RetrievalMetrics:
    """
    Minimal COCO-style retrieval on a slice of pairs.

    We compute:
      - Image->Text Recall@K: for each image query, if any of its GT captions appears in top-K
      - Text->Image Recall@K: for each caption query, if its GT image appears in top-K
    """
    # IMPORTANT: never mutate the training dataset in-place.
    eval_dataset = dataset
    if max_pairs is not None:
        mp = int(max_pairs)
        if mp > 0 and mp < len(dataset):
            eval_dataset = Subset(dataset, range(mp))

    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model.eval()

    image_embs: list[torch.Tensor] = []
    text_embs: list[torch.Tensor] = []
    image_ids: list[int] = []
    caption_ids: list[int] = []

    for batch in tqdm(loader, desc="eval_embed", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else None

        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        image_embs.append(out["image_emb"].detach().cpu())
        text_embs.append(out["text_emb"].detach().cpu())
        image_ids.extend([int(x) for x in batch["image_id"].tolist()])
        caption_ids.extend([int(x) for x in batch["caption_id"].tolist()])

    I = torch.cat(image_embs, dim=0)  # [N, D]
    T = torch.cat(text_embs, dim=0)  # [N, D]
    sims = I @ T.T  # cosine because normalized

    # Build GT mappings within this eval slice
    image_to_caption_ids: dict[int, set[int]] = {}
    caption_to_image_id: dict[int, int] = {}
    for iid, cid in zip(image_ids, caption_ids):
        image_to_caption_ids.setdefault(int(iid), set()).add(int(cid))
        caption_to_image_id[int(cid)] = int(iid)

    def recall_at_k_i2t(k: int) -> float:
        correct = 0
        for row, iid in enumerate(image_ids):
            topk = torch.topk(sims[row], k=min(k, sims.shape[1]), largest=True).indices.tolist()
            gt = image_to_caption_ids[int(iid)]
            if any(int(caption_ids[j]) in gt for j in topk):
                correct += 1
        return correct / max(1, len(image_ids))

    def recall_at_k_t2i(k: int) -> float:
        correct = 0
        sims_t2i = sims.T
        for row, cid in enumerate(caption_ids):
            topk = torch.topk(sims_t2i[row], k=min(k, sims_t2i.shape[1]), largest=True).indices.tolist()
            gt_iid = caption_to_image_id[int(cid)]
            if any(int(image_ids[j]) == gt_iid for j in topk):
                correct += 1
        return correct / max(1, len(caption_ids))

    return RetrievalMetrics(
        i2t_r1=recall_at_k_i2t(1),
        i2t_r5=recall_at_k_i2t(5),
        i2t_r10=recall_at_k_i2t(10),
        t2i_r1=recall_at_k_t2i(1),
        t2i_r5=recall_at_k_t2i(5),
        t2i_r10=recall_at_k_t2i(10),
    )

