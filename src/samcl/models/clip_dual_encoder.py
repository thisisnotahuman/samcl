from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


@dataclass(frozen=True)
class ClipDualEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch32"


def load_clip_processors(model_name: str) -> tuple[Any, Any]:
    """
    Returns (image_processor, tokenizer).
    """
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return image_processor, tokenizer


class ClipDualEncoder(nn.Module):
    """
    CLIP-style dual encoder (no fusion, no cross-attention, no joint encoder).
    Uses pretrained CLIP encoders; unfrozen during training.
    """

    def __init__(self, cfg: ClipDualEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip = CLIPModel.from_pretrained(cfg.model_name)

        # By default, transformers modules require_grad=True; keep unfrozen.
        self.logit_scale = self.clip.logit_scale  # nn.Parameter

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Be compatible across transformers versions:
        # Some versions return a ModelOutput object from get_image_features/get_text_features.
        try:
            z = self.clip.get_image_features(pixel_values=pixel_values)  # ideally Tensor [B, D]
            if hasattr(z, "pooler_output"):
                # ModelOutputWithPooling
                z = z.pooler_output
            if not isinstance(z, torch.Tensor):
                raise TypeError("get_image_features did not return a Tensor")
        except Exception:
            vision_out = self.clip.vision_model(pixel_values=pixel_values)
            pooled = getattr(vision_out, "pooler_output", None)
            if pooled is None:
                pooled = vision_out.last_hidden_state[:, 0, :]
            z = self.clip.visual_projection(pooled)

        z = torch.nn.functional.normalize(z, dim=-1)
        return z

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        try:
            z = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)  # ideally Tensor [B, D]
            if hasattr(z, "pooler_output"):
                z = z.pooler_output
            if not isinstance(z, torch.Tensor):
                raise TypeError("get_text_features did not return a Tensor")
        except Exception:
            text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(text_out, "pooler_output", None)
            if pooled is None:
                pooled = text_out.last_hidden_state[:, 0, :]
            z = self.clip.text_projection(pooled)

        z = torch.nn.functional.normalize(z, dim=-1)
        return z

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        img = self.encode_image(pixel_values)
        txt = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (img @ txt.t())
        return {
            "image_emb": img,
            "text_emb": txt,
            "logits_per_image": logits,
            "logits_per_text": logits.t(),
            "logit_scale": logit_scale,
        }

