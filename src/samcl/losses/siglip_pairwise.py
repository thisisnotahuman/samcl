from __future__ import annotations

import torch
import torch.nn.functional as F


def siglip_pairwise_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    SigLIP-style sigmoid loss on the full image-text similarity matrix (batch-local).

    Expects logits[i, j] = scale * cos(z_img_i, z_txt_j) + bias (cos from L2-normalized embeddings).
    Labels: +1 on the diagonal (matched pairs), -1 elsewhere.
    """
    n = logits.shape[0]
    if logits.shape[1] != n:
        raise ValueError(f"siglip_pairwise_loss expects square logits [n,n], got {tuple(logits.shape)}")
    labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0
    return -(F.logsigmoid(labels * logits)).mean()
