from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


@dataclass(frozen=True)
class UniModalDualEncoderConfig:
    # Pure uni-modal pretrained backbones (no multimodal pretraining)
    vision_model_name: str = "google/vit-base-patch16-224-in21k"
    text_model_name: str = "bert-base-uncased"

    # If False, build the backbone from config with random init (train from scratch)
    vision_pretrained: bool = True
    text_pretrained: bool = True

    # Connection / projection layers
    proj_dim: int = 256
    proj_hidden_dim: int = 512  # used when proj_layers=2
    proj_layers: int = 2  # 1: linear, 2: MLP

    # CLIP-style temperature parameterization
    init_temperature: float = 0.07
    max_logit_scale: float = 100.0


def load_unimodal_processors(
    *,
    vision_model_name: str,
    text_model_name: str,
) -> tuple[Any, Any]:
    """
    Returns (image_processor, tokenizer) for the specified uni-modal models.
    """
    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
    return image_processor, tokenizer


def _make_proj(in_dim: int, cfg: UniModalDualEncoderConfig) -> nn.Module:
    if int(cfg.proj_layers) <= 1:
        return nn.Linear(in_dim, int(cfg.proj_dim), bias=False)
    return nn.Sequential(
        nn.Linear(in_dim, int(cfg.proj_hidden_dim)),
        nn.GELU(),
        nn.Linear(int(cfg.proj_hidden_dim), int(cfg.proj_dim), bias=False),
    )


class UniModalDualEncoder(nn.Module):
    """
    Uni-modal dual encoder:
      - Vision backbone: pretrained on vision-only data (e.g., ImageNet / DINO)
      - Text backbone: pretrained on text-only data (e.g., BERT)
      - Connection layers: projection heads (linear/MLP) into shared embedding space

    No fusion, no cross-attention, no joint encoder.
    """

    def __init__(self, cfg: UniModalDualEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if bool(cfg.vision_pretrained):
            self.vision = AutoModel.from_pretrained(cfg.vision_model_name)
        else:
            vcfg = AutoConfig.from_pretrained(cfg.vision_model_name)
            self.vision = AutoModel.from_config(vcfg)

        if bool(cfg.text_pretrained):
            self.text = AutoModel.from_pretrained(cfg.text_model_name)
        else:
            tcfg = AutoConfig.from_pretrained(cfg.text_model_name)
            self.text = AutoModel.from_config(tcfg)

        vision_dim = int(getattr(self.vision.config, "hidden_size"))
        text_dim = int(getattr(self.text.config, "hidden_size"))

        self.vision_proj = _make_proj(vision_dim, cfg)
        self.text_proj = _make_proj(text_dim, cfg)

        # logit_scale is stored in log-space like CLIP.
        init_logit_scale = math.log(1.0 / float(cfg.init_temperature))
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))
        # Scalar bias (SigLIP); constant offset cancels in batch-softmax InfoNCE, so default 0 keeps old behavior.
        self.logit_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _pool(self, out: Any) -> torch.Tensor:
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        # Fallback to CLS token
        return out.last_hidden_state[:, 0, :]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision(pixel_values=pixel_values)
        pooled = self._pool(out)
        z = self.vision_proj(pooled)
        return torch.nn.functional.normalize(z, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out)
        z = self.text_proj(pooled)
        return torch.nn.functional.normalize(z, dim=-1)

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
        logits = logit_scale * (img @ txt.t()) + self.logit_bias
        return {
            "image_emb": img,
            "text_emb": txt,
            "logits_per_image": logits,
            "logits_per_text": logits.t(),
            "logit_scale": logit_scale,
        }

