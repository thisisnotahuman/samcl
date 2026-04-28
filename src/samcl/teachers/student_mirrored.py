from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from samcl.models.unimodal_dual_encoder import UniModalDualEncoder


@dataclass(frozen=True)
class StudentMirroredTextTeacherConfig:
    batch_size: int = 128


@dataclass(frozen=True)
class StudentMirroredImageTeacherConfig:
    batch_size: int = 64


class StudentMirroredTextTeacher:
    """
    Semantic-sampling text embeddings via the same UniModalDualEncoder.encode_text
    as the student (L2-normalized projected space).
    """

    def __init__(
        self,
        model: UniModalDualEncoder,
        tokenizer: Any,
        device: torch.device,
        *,
        max_text_len: int,
        batch_size: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_text_len = int(max_text_len)
        self.cfg = StudentMirroredTextTeacherConfig(batch_size=int(batch_size))

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        self.model.eval()
        outs: list[torch.Tensor] = []
        bs = int(self.cfg.batch_size)
        for s in range(0, len(texts), bs):
            chunk = texts[s : s + bs]
            tok = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outs.append(self.model.encode_text(input_ids, attention_mask))
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), device=self.device)


class StudentMirroredImageTeacher:
    """
    Semantic-sampling image embeddings via the same UniModalDualEncoder.encode_image
    as the student (L2-normalized projected space).
    """

    def __init__(
        self,
        model: UniModalDualEncoder,
        image_processor: Any,
        device: torch.device,
        *,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.cfg = StudentMirroredImageTeacherConfig(batch_size=int(batch_size))

    @torch.no_grad()
    def encode_images(self, pil_images: list) -> torch.Tensor:
        self.model.eval()
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        return self.model.encode_image(pixel_values)
