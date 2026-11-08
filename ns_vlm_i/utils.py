from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def validate_prompt(prompt: str) -> str:
    value = str(prompt).strip()
    if not value:
        raise ValueError("prompt must be a non-empty string.")
    return value


def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown dtype: {name!r}.")
    return mapping[key]


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    weights = expanded.to(dtype=values.dtype)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(float(eps))
    return numerator / denominator


def load_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("A NumPy image must have shape [height, width, 3 or 4].")
        if array.dtype != np.uint8:
            numeric = np.asarray(array, dtype=np.float32)
            if numeric.size and float(numeric.max()) <= 1.0:
                numeric = numeric * 255.0
            array = np.clip(numeric, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if hasattr(value, "read"):
        payload = value.read()
        if hasattr(value, "seek"):
            value.seek(0)
        return Image.open(io.BytesIO(payload)).convert("RGB")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def coerce_images(images: Any) -> List[Image.Image]:
    atomic = (str, Path, bytes, bytearray, memoryview, Image.Image, np.ndarray)
    if isinstance(images, atomic) or hasattr(images, "read"):
        values = [images]
    elif isinstance(images, Sequence):
        values = list(images)
    else:
        raise TypeError("images must be one image or a sequence of images.")
    if not values:
        raise ValueError("At least one image is required.")
    return [load_rgb_image(value) for value in values]


def normalize_direction(values: torch.Tensor, eps: float) -> torch.Tensor:
    return F.normalize(values.float(), p=2.0, dim=-1, eps=float(eps)).to(dtype=values.dtype)
