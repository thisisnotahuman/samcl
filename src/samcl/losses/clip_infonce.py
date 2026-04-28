from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_infonce_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Standard CLIP-style symmetric InfoNCE loss:
      - image -> text
      - text -> image
    The objective is fixed across experiments; sampling is the only variable.
    """
    b = logits_per_image.shape[0]
    targets = torch.arange(b, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) * 0.5

