from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from samcl.models.cvcl_dino_loader import CvclDinoConfig, load_cvcl_dino_backbone


@dataclass(frozen=True)
class CvclDualEncoderConfig:
    # CVCL vision backbone identifier (SayCam-pretrained by default)
    vision_model_name: str = "dino_sfp_vitb16"
    vision_pretrained: bool = True

    # CVCL text encoder: simple word embedding + mean pooling
    vocab_size: int = 32000
    embed_dim: int = 512

    # CLIP-style temperature parameterization
    init_temperature: float = 0.07
    max_logit_scale: float = 100.0


class CvclDualEncoder(nn.Module):
    """
    CVCL-style student dual encoder:
      - Vision: DINO ViT (e.g., dino_sfp_vitb16) with a trainable head to 512-d
      - Text: word embedding (512-d) with mean pooling over tokens
      - L2-normalize both embeddings
      - dot-product similarity with CLIP-style logit_scale

    This is compatible with the existing training loop interface:
      forward(pixel_values, input_ids, attention_mask) -> dict with image_emb/text_emb/logits.
    """

    def __init__(self, cfg: CvclDualEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.vision = load_cvcl_dino_backbone(
            CvclDinoConfig(model_name=cfg.vision_model_name, pretrained=cfg.vision_pretrained)
        )

        # CVCL ViT backbones expose embed dim via common names; default to 768 for ViT-B.
        vision_out_dim = int(getattr(self.vision, "embed_dim", 768))
        if not hasattr(self.vision, "head") or isinstance(getattr(self.vision, "head"), nn.Identity):
            # ensure a head exists for a unified output dim
            self.vision.head = nn.Linear(vision_out_dim, int(cfg.embed_dim))
        else:
            # replace whatever head with our projection for consistency
            self.vision.head = nn.Linear(vision_out_dim, int(cfg.embed_dim))

        self.text_embedding = nn.Embedding(int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=0)

        init_logit_scale = math.log(1.0 / float(cfg.init_temperature))
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # vision_transformer_dino_mugs exposes forward_features (alias to forward)
        if hasattr(self.vision, "forward_features"):
            feats = self.vision.forward_features(pixel_values)
        else:
            feats = self.vision(pixel_values)
        head = getattr(self.vision, "head", None)
        if head is not None:
            feats = head(feats)
        return F.normalize(feats, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # input_ids: [B,L], attention_mask: [B,L] (1=token, 0=pad)
        emb = self.text_embedding(input_ids)  # [B,L,D]
        if attention_mask is None:
            pooled = emb.mean(dim=1)
        else:
            m = attention_mask.to(dtype=emb.dtype).unsqueeze(-1)  # [B,L,1]
            summed = (emb * m).sum(dim=1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return F.normalize(pooled, dim=-1)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        img = self.encode_image(pixel_values)
        txt = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=float(self.cfg.max_logit_scale))
        logits = logit_scale * (img @ txt.t())
        return {
            "image_emb": img,
            "text_emb": txt,
            "logits_per_image": logits,
            "logits_per_text": logits.t(),
            "logit_scale": logit_scale,
        }

