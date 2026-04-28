from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

# Avoid tokenizer thread pool + later subprocess/fork (e.g. nvidia-smi) spamming warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


@dataclass(frozen=True)
class TextTeacherConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 128


class FrozenTextTeacher:
    """Frozen sentence encoder for sampling-time similarity only."""

    def __init__(self, cfg: TextTeacherConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency for text teacher. Install `sentence-transformers`."
            ) from e

        self.model = SentenceTransformer(cfg.model_name, device=str(device))
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return emb.to(self.device)

    @torch.no_grad()
    def encode_iter(self, texts: Iterable[str], *, chunk_size: int = 4096) -> torch.Tensor:
        buf: list[str] = []
        outs: list[torch.Tensor] = []
        for t in texts:
            buf.append(t)
            if len(buf) >= chunk_size:
                outs.append(self.encode(buf).cpu())
                buf = []
        if buf:
            outs.append(self.encode(buf).cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0))
