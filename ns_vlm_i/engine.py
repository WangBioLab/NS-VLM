from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from .config import InferenceConfig
from .model import ModelInferenceOutput, NextStateVLM
from .utils import coerce_images, validate_prompt


ARCHITECTURE_NAME = NextStateVLM.architecture_name


def resolve_device(requested: str | torch.device | None) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested is not None and str(requested).strip() and str(requested).lower() != "auto":
        device = torch.device(str(requested))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and (
            not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return {"state_dict": load_file(str(path), device="cpu")}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("The checkpoint must contain a mapping.")
    return payload


def _checkpoint_architecture(payload: Mapping[str, Any]) -> str | None:
    direct = payload.get("architecture")
    if isinstance(direct, str):
        return direct.strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("architecture")
        if isinstance(nested, str):
            return nested.strip()
    return None


def _extract_state_dict(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    state_values = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state_values, Mapping):
        raise TypeError("The checkpoint does not contain a model state dictionary.")
    state: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_values.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        while key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    if not state:
        raise ValueError("The checkpoint contains no tensor parameters.")
    return state


def _is_pretrained_backbone_value(key: str) -> bool:
    return key.startswith("backbone.model.") and "lora_" not in key


def load_inference_checkpoint(
    model: NextStateVLM,
    checkpoint_path: str | Path,
) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = _read_checkpoint(path)
    declared_architecture = _checkpoint_architecture(payload)
    if declared_architecture and declared_architecture != ARCHITECTURE_NAME:
        raise ValueError(
            f"This runtime requires architecture {ARCHITECTURE_NAME!r}; "
            f"the checkpoint declares {declared_architecture!r}."
        )

    source = _extract_state_dict(payload)
    target = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    shape_errors: List[str] = []
    for key, value in source.items():
        if key not in target:
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_errors.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(target[key].shape)}"
            )
            continue
        compatible[key] = value

    if shape_errors:
        joined = "\n".join(shape_errors[:20])
        raise ValueError(f"Checkpoint tensor shapes do not match NS-VLM-I:\n{joined}")
    missing_required = sorted(
        key
        for key in target
        if key not in compatible and not _is_pretrained_backbone_value(key)
    )
    if missing_required:
        joined = "\n".join(missing_required[:20])
        raise ValueError(
            "Checkpoint is missing NS-VLM inference parameters. "
            f"A {ARCHITECTURE_NAME!r} checkpoint is required:\n{joined}"
        )

    result = model.load_state_dict(compatible, strict=False)
    unresolved_required = sorted(
        key for key in result.missing_keys if not _is_pretrained_backbone_value(key)
    )
    if unresolved_required or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint loading produced unresolved keys: "
            f"missing={unresolved_required}, unexpected={result.unexpected_keys}"
        )


class NSVLMI:
    """NS-VLM prompt-and-image inference interface."""

    def __init__(
        self,
        config: InferenceConfig | None = None,
    ):
        self.config = config if config is not None else InferenceConfig()
        checkpoint_path = str(self.config.checkpoint_path).strip()
        if not checkpoint_path:
            raise RuntimeError("checkpoint_path is required in InferenceConfig.")
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
        self.device = resolve_device(self.config.device)
        self.model = NextStateVLM(self.config)
        load_inference_checkpoint(self.model, resolved_checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_environment(cls) -> "NSVLMI":
        return cls(InferenceConfig.from_environment())

    def infer(self, prompt: str, images: Any) -> ModelInferenceOutput:
        question = validate_prompt(prompt)
        image_list = coerce_images(images)
        with torch.inference_mode():
            return self.model.infer(
                question,
                image_list,
                self.device,
                max_new_tokens=int(self.config.max_new_tokens),
            )

    def __call__(self, prompt: str, images: Any) -> ModelInferenceOutput:
        return self.infer(prompt, images)
