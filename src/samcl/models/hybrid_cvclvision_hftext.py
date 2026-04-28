from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from samcl.models.cvcl_dino_loader import CvclDinoConfig, load_cvcl_dino_backbone


@dataclass(frozen=True)
class HybridCvclVisionHfTextConfig:
    # CVCL vision backbone
    cvcl_vision_model_name: str = "dino_sfp_vitb16"
    cvcl_vision_pretrained: bool = True

    # HF text backbone (e.g. BERT)
    text_model_name: str = "bert-base-uncased"
    text_pretrained: bool = True

    # Projection into shared space (same as original unimodal student)
    proj_dim: int = 256
    proj_hidden_dim: int = 512
    proj_layers: int = 2

    init_temperature: float = 0.07
    max_logit_scale: float = 100.0


def load_hf_text_tokenizer(*, text_model_name: str) -> Any:
    return AutoTokenizer.from_pretrained(text_model_name, use_fast=True)


def _make_proj(in_dim: int, cfg: HybridCvclVisionHfTextConfig) -> nn.Module:
    if int(cfg.proj_layers) <= 1:
        return nn.Linear(in_dim, int(cfg.proj_dim), bias=False)
    return nn.Sequential(
        nn.Linear(in_dim, int(cfg.proj_hidden_dim)),
        nn.GELU(),
        nn.Linear(int(cfg.proj_hidden_dim), int(cfg.proj_dim), bias=False),
    )


class HybridCvclVisionHfText(nn.Module):
    """
    Student dual encoder where:
      - vision = CVCL DINO/MUGS backbone (e.g. dino_sfp_vitb16)
      - text   = HF language model (e.g. bert-base-uncased), pretrained or from scratch

    Outputs match the existing training loop contract.
    """

    def __init__(self, cfg: HybridCvclVisionHfTextConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.vision = load_cvcl_dino_backbone(
            CvclDinoConfig(model_name=cfg.cvcl_vision_model_name, pretrained=cfg.cvcl_vision_pretrained)
        )

        if bool(cfg.text_pretrained):
            self.text = AutoModel.from_pretrained(cfg.text_model_name)
        else:
            tcfg = AutoConfig.from_pretrained(cfg.text_model_name)
            self.text = AutoModel.from_config(tcfg)

        # Infer dims
        vision_dim = int(getattr(self.vision, "embed_dim", 768))
        text_dim = int(getattr(self.text.config, "hidden_size"))

        self.vision_proj = _make_proj(vision_dim, cfg)
        self.text_proj = _make_proj(text_dim, cfg)

        init_logit_scale = math.log(1.0 / float(cfg.init_temperature))
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))

    def _pool_text(self, out: Any) -> torch.Tensor:
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        return out.last_hidden_state[:, 0, :]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vision, "forward_features"):
            feats = self.vision.forward_features(pixel_values)
        else:
            feats = self.vision(pixel_values)
        z = self.vision_proj(feats)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_text(out)
        z = self.text_proj(pooled)
        return F.normalize(z, dim=-1)

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

