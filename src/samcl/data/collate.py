from __future__ import annotations

from typing import Any, Callable

import torch

from torchvision import transforms

from samcl.data.cvcl_vocab import CvclVocab, cvcl_tokenize_batch


class ClipCollator:
    """
    Converts a list of dataset dicts into model-ready tensors using provided processors.

    We keep IDs (image_id, caption_id) for downstream semantic lookup and logging.
    """

    def __init__(
        self,
        *,
        image_processor: Any,
        text_tokenizer: Any,
        max_length: int = 77,
    ) -> None:
        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]

        image_inputs = self.image_processor(images=images, return_tensors="pt")
        text_inputs = self.text_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Optional sampler-provided metadata (used for sampler diagnostics/logging).
        # We keep these as tensors so training code can aggregate them cheaply.
        metas = [x.get("sampling_meta", None) for x in batch]
        if any(m is not None for m in metas):
            # Default values for samples without meta.
            is_fallback = [int((m or {}).get("is_fallback", 0)) for m in metas]
            tries = [int((m or {}).get("tries", 0)) for m in metas]
            target_relation = [int((m or {}).get("target_relation", -1)) for m in metas]
            found_relation = [int((m or {}).get("found_relation", -1)) for m in metas]
        else:
            is_fallback = None
            tries = None
            target_relation = None
            found_relation = None

        return {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask", None),
            "image_id": torch.tensor([int(x["image_id"]) for x in batch], dtype=torch.long),
            "caption_id": torch.tensor([int(x["caption_id"]) for x in batch], dtype=torch.long),
            "sample_is_fallback": (torch.tensor(is_fallback, dtype=torch.long) if is_fallback is not None else None),
            "sample_tries": (torch.tensor(tries, dtype=torch.long) if tries is not None else None),
            "sample_target_relation": (
                torch.tensor(target_relation, dtype=torch.long) if target_relation is not None else None
            ),
            "sample_found_relation": (
                torch.tensor(found_relation, dtype=torch.long) if found_relation is not None else None
            ),
        }


class CvclCollator:
    """
    CVCL-style collator:
      - images -> torchvision transforms (CLIP normalization)
      - text -> whitespace tokenization with a dataset-built vocab

    Output keys are aligned with ClipCollator so training/eval loops stay unchanged.
    """

    def __init__(self, *, vocab: CvclVocab, image_size: int = 224, use_strong_aug: bool = False) -> None:
        self.vocab = vocab
        self.image_size = int(image_size)
        self.use_strong_aug = bool(use_strong_aug)

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        if self.use_strong_aug:
            self.image_tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=self.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.image_tf = transforms.Compose(
                [
                    transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]

        pixel_values = torch.stack([self.image_tf(im) for im in images], dim=0)

        token_lists, lengths = cvcl_tokenize_batch(captions, self.vocab)
        max_len = max(lengths) if lengths else 1
        input_ids = torch.full((len(token_lists), max_len), int(self.vocab.pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(token_lists), max_len), dtype=torch.long)
        for i, ids in enumerate(token_lists):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        metas = [x.get("sampling_meta", None) for x in batch]
        if any(m is not None for m in metas):
            is_fallback = [int((m or {}).get("is_fallback", 0)) for m in metas]
            tries = [int((m or {}).get("tries", 0)) for m in metas]
            target_relation = [int((m or {}).get("target_relation", -1)) for m in metas]
            found_relation = [int((m or {}).get("found_relation", -1)) for m in metas]
        else:
            is_fallback = None
            tries = None
            target_relation = None
            found_relation = None

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_id": torch.tensor([int(x["image_id"]) for x in batch], dtype=torch.long),
            "caption_id": torch.tensor([int(x["caption_id"]) for x in batch], dtype=torch.long),
            "sample_is_fallback": (torch.tensor(is_fallback, dtype=torch.long) if is_fallback is not None else None),
            "sample_tries": (torch.tensor(tries, dtype=torch.long) if tries is not None else None),
            "sample_target_relation": (
                torch.tensor(target_relation, dtype=torch.long) if target_relation is not None else None
            ),
            "sample_found_relation": (
                torch.tensor(found_relation, dtype=torch.long) if found_relation is not None else None
            ),
        }


class CvclImageHfTextCollator:
    """
    Hybrid collator:
      - Image pipeline: CVCL-style (torchvision transforms with CLIP mean/std)
      - Text pipeline: HF tokenizer (e.g. bert-base-uncased)
    """

    def __init__(
        self,
        *,
        text_tokenizer: Any,
        max_length: int = 77,
        image_size: int = 224,
        use_strong_aug: bool = False,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.max_length = int(max_length)
        self.image_size = int(image_size)
        self.use_strong_aug = bool(use_strong_aug)

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        if self.use_strong_aug:
            self.image_tf = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=self.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.image_tf = transforms.Compose(
                [
                    transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]

        pixel_values = torch.stack([self.image_tf(im) for im in images], dim=0)
        text_inputs = self.text_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        metas = [x.get("sampling_meta", None) for x in batch]
        if any(m is not None for m in metas):
            is_fallback = [int((m or {}).get("is_fallback", 0)) for m in metas]
            tries = [int((m or {}).get("tries", 0)) for m in metas]
            target_relation = [int((m or {}).get("target_relation", -1)) for m in metas]
            found_relation = [int((m or {}).get("found_relation", -1)) for m in metas]
        else:
            is_fallback = None
            tries = None
            target_relation = None
            found_relation = None

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask", None),
            "image_id": torch.tensor([int(x["image_id"]) for x in batch], dtype=torch.long),
            "caption_id": torch.tensor([int(x["caption_id"]) for x in batch], dtype=torch.long),
            "sample_is_fallback": (torch.tensor(is_fallback, dtype=torch.long) if is_fallback is not None else None),
            "sample_tries": (torch.tensor(tries, dtype=torch.long) if tries is not None else None),
            "sample_target_relation": (
                torch.tensor(target_relation, dtype=torch.long) if target_relation is not None else None
            ),
            "sample_found_relation": (
                torch.tensor(found_relation, dtype=torch.long) if found_relation is not None else None
            ),
        }

