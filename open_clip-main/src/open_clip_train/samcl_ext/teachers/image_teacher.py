from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


@dataclass(frozen=True)
class ImageTeacherConfig:
    arch: str = "resnet50"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 64
    image_size: int = 224


class FrozenImageTeacher(nn.Module):
    """Frozen vision encoder for sampling-time similarity only."""

    def __init__(self, cfg: ImageTeacherConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.mode = str(cfg.arch).lower()
        self.backbone: nn.Module
        self.preprocess = None
        self.clip_processor = None

        if self.mode == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            backbone = resnet50(weights=weights)
            backbone.fc = nn.Identity()
            self.backbone = backbone.to(device)
            self.backbone.eval()

            mean = weights.transforms().mean
            std = weights.transforms().std
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(cfg.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            for p in self.backbone.parameters():
                p.requires_grad_(False)

        elif self.mode == "clip":
            try:
                from transformers import CLIPImageProcessor, CLIPModel
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Missing dependency for CLIP image teacher. Install `transformers`.") from e

            self.clip_processor = CLIPImageProcessor.from_pretrained(cfg.clip_model_name)
            clip = CLIPModel.from_pretrained(cfg.clip_model_name)
            clip.eval()
            for p in clip.parameters():
                p.requires_grad_(False)
            self.backbone = clip.to(device)
        else:
            raise ValueError(f"Unsupported teacher arch: {cfg.arch} (use 'resnet50' or 'clip')")

    @torch.no_grad()
    def encode_images(self, pil_images: list) -> torch.Tensor:
        if self.mode == "resnet50":
            assert self.preprocess is not None
            x = torch.stack([self.preprocess(im) for im in pil_images], dim=0).to(self.device)
            feat = self.backbone(x)
            return torch.nn.functional.normalize(feat, dim=-1)

        assert self.mode == "clip"
        assert self.clip_processor is not None
        clip = self.backbone
        inputs = self.clip_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        vision_out = clip.vision_model(pixel_values=pixel_values)
        pooled = getattr(vision_out, "pooler_output", None)
        if pooled is None:
            pooled = vision_out.last_hidden_state[:, 0, :]
        feat = clip.visual_projection(pooled)
        return torch.nn.functional.normalize(feat, dim=-1)
