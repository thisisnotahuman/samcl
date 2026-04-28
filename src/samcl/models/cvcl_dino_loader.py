from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from huggingface_hub import hf_hub_download
from torchvision import models as torchvision_models

from samcl.models import vision_transformer_dino_mugs as vits


@dataclass(frozen=True)
class CvclDinoConfig:
    """
    CVCL-style DINO/MUGS model identifiers follow:
      "{alg}_{data}_{model_spec}"
    Examples:
      - dino_sfp_vitb16   (SayCam-FPS? pretraining; ViT-Base/16)
      - dino_sfp_resnext50
    """

    model_name: str = "dino_sfp_vitb16"
    pretrained: bool = True
    checkpoint_key: str = "teacher"


def _parse_model_name(model_name: str) -> Tuple[str, str, str]:
    parts = str(model_name).split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected model_name like 'dino_sfp_vitb16', got {model_name!r}")
    alg, data, model_spec = parts
    if alg not in ("dino", "mugs", "mae"):
        raise ValueError(f"Unrecognized algorithm {alg!r}")
    if data not in ("say", "s", "sfp", "a", "y", "imagenet1k", "imagenet100", "imagenet10", "imagenet3", "imagenet1"):
        raise ValueError(f"Unrecognized data {data!r}")
    if model_spec not in ("resnext50", "vitb14", "vitl16", "vitb16", "vits16"):
        raise ValueError(f"Unrecognized architecture {model_spec!r}")
    return alg, data, model_spec


def _build_dino_mugs(arch: str, patch_size: int | None) -> torch.nn.Module:
    if arch in vits.__dict__.keys():
        return vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    if arch in torchvision_models.__dict__.keys():
        m = torchvision_models.__dict__[arch]()
        m.fc = torch.nn.Identity()
        return m
    raise ValueError(f"Unknown architecture: {arch}")


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str, checkpoint_key: str | None) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_key is not None and isinstance(state_dict, dict) and checkpoint_key in state_dict:
        state_dict = state_dict[checkpoint_key]

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint did not contain a state_dict dict")

    # remove common prefixes
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)


def load_cvcl_dino_backbone(cfg: CvclDinoConfig) -> torch.nn.Module:
    alg, data, model_spec = _parse_model_name(cfg.model_name)

    if model_spec == "resnext50":
        arch, patch_size = "resnext50_32x4d", None
    elif model_spec == "vitb14":
        arch, patch_size = "vit_base", 14
    elif model_spec == "vitl16":
        arch, patch_size = "vit_large", 16
    elif model_spec == "vitb16":
        arch, patch_size = "vit_base", 16
    elif model_spec == "vits16":
        arch, patch_size = "vit_small", 16
    else:
        raise ValueError(f"Unhandled model_spec: {model_spec}")

    # download checkpoint (matches CVCL utils_mcl.py behavior)
    if data == "imagenet1k":
        checkpoint = "./model/dino_resnet50_pretrain_full_checkpoint.pth"
    else:
        checkpoint = hf_hub_download(repo_id="eminorhan/" + cfg.model_name, filename=cfg.model_name + ".pth")

    if alg in ("dino", "mugs"):
        model = _build_dino_mugs(arch, patch_size)
        if cfg.pretrained:
            _load_state_dict(model, checkpoint, cfg.checkpoint_key)
        return model

    raise NotImplementedError("MAE support not implemented (aligns with CVCL TODO)")

