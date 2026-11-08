from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .config import InferenceConfig
from .utils import get_torch_dtype, masked_mean


@dataclass(frozen=True)
class PrefillState:
    pooled_state: torch.Tensor
    token_bank: torch.Tensor
    token_mask: torch.Tensor


@contextmanager
def temporary_padding_side(tokenizer: Any, padding_side: str):
    original = tokenizer.padding_side
    tokenizer.padding_side = str(padding_side)
    try:
        yield
    finally:
        tokenizer.padding_side = original


def infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for path in ("hidden_size", "text_config.hidden_size", "llm_config.hidden_size"):
        current = config
        valid = True
        for name in path.split("."):
            if current is None or not hasattr(current, name):
                valid = False
                break
            current = getattr(current, name)
        if valid and isinstance(current, int):
            return int(current)
    raise AttributeError("Unable to infer the language hidden size from the backbone configuration.")


class QwenBackbone(nn.Module):
    """Frozen Qwen3-VL encoder and terminal-state verbalizer used during inference."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        dtype = get_torch_dtype(config.dtype)
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            min_pixels=int(config.min_pixels),
            max_pixels=int(config.max_pixels),
            local_files_only=bool(config.local_files_only),
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = int(config.min_pixels)
            if hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = int(config.max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=bool(config.local_files_only),
        )
        self.hidden_size = infer_hidden_size(self.model)
        self._configure_lora()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _configure_lora(self) -> None:
        if not self.config.use_lora:
            return
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = self._find_language_lora_targets()
        peft_config = LoraConfig(
            r=int(self.config.lora_r),
            lora_alpha=int(self.config.lora_alpha),
            lora_dropout=float(self.config.lora_dropout),
            bias=str(self.config.lora_bias),
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)

    def _find_language_lora_targets(self) -> List[str]:
        requested = {
            name.strip()
            for name in str(self.config.lora_target_modules).split(",")
            if name.strip()
        }
        if not requested:
            raise ValueError("lora_target_modules must name the language-model projections.")
        found: List[str] = []
        for name, _module in self.model.named_modules():
            lowered = name.lower()
            if "visual" in lowered or "vision" in lowered:
                continue
            if name.split(".")[-1] in requested:
                found.append(name)
        if not found:
            raise ValueError("No configured language-model LoRA projection was found in the backbone.")
        return sorted(set(found))

    def _token_ids_for_tokens(self, tokens: Sequence[str]) -> List[int]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            return []
        vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        result: List[int] = []
        for token in tokens:
            if vocabulary and token not in vocabulary:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in result:
                result.append(int(token_id))
        return result

    def _resize_to_pixels(self, image: Image.Image, target_pixels: int) -> Image.Image:
        width, height = image.size
        current_pixels = max(1, int(width) * int(height))
        if target_pixels <= 0 or current_pixels <= target_pixels:
            return image
        scale = math.sqrt(float(target_pixels) / float(current_pixels))
        new_width = max(32, int(round(width * scale)))
        new_height = max(32, int(round(height * scale)))
        resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        return image.resize((new_width, new_height), resample=resampling)

    def normalize_images(self, images: Sequence[Image.Image]) -> List[Image.Image]:
        normalized = [image.convert("RGB") for image in images]
        if not normalized:
            raise ValueError("At least one image is required.")
        if not self.config.adaptive_resize or self.config.max_total_pixels <= 0:
            return normalized
        pixel_counts = [max(1, image.width * image.height) for image in normalized]
        total_pixels = sum(pixel_counts)
        if total_pixels <= int(self.config.max_total_pixels):
            return normalized
        scale = math.sqrt(float(self.config.max_total_pixels) / float(total_pixels))
        effective_minimum = max(
            32 * 32,
            min(
                int(self.config.per_image_min_pixels),
                int(self.config.max_total_pixels) // len(normalized),
            ),
        )
        resized: List[Image.Image] = []
        for image, pixel_count in zip(normalized, pixel_counts):
            target = max(effective_minimum, int(pixel_count * scale * scale))
            resized.append(self._resize_to_pixels(image, target))
        return resized

    def _multimodal_messages(
        self,
        images: Sequence[Image.Image],
        prompt: str,
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        normalized = self.normalize_images(images)
        content: List[Dict[str, Any]] = [
            {"type": "image", "image": image} for image in normalized
        ]
        content.append({"type": "text", "text": str(prompt)})
        return [{"role": "user", "content": content}], normalized

    def prepare_multimodal_inputs(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        add_generation_prompt: bool,
        padding_side: str,
    ) -> Dict[str, Any]:
        messages, normalized = self._multimodal_messages(images, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        with temporary_padding_side(tokenizer, padding_side):
            inputs = self.processor(
                text=[text],
                images=normalized,
                padding=True,
                return_tensors="pt",
            )
        return dict(inputs)

    def prepare_text_generation_inputs(
        self,
        prompt: str,
        padding_side: str,
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": str(prompt)}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        with temporary_padding_side(tokenizer, padding_side):
            inputs = tokenizer(
                [text],
                padding=True,
                return_tensors="pt",
            )
        return dict(inputs)

    def _move_inputs(self, inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    def _prepend_terminal_state(
        self,
        inputs: Dict[str, Any],
        terminal_state: torch.Tensor,
        include_input_ids: bool,
    ) -> Dict[str, Any]:
        if "input_ids" not in inputs:
            raise ValueError("input_ids are required for terminal-state injection.")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        token_embeddings = self.model.get_input_embeddings()(input_ids)
        prefix = terminal_state.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        if prefix.ndim != 3 or prefix.shape[0] != input_ids.shape[0]:
            raise ValueError("terminal_state must have shape [batch, state_tokens, hidden].")
        prefix_mask = torch.ones(
            prefix.shape[:2],
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
            device=prefix.device,
        )
        token_mask = (
            attention_mask.to(prefix.device)
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=prefix_mask.dtype)
        )
        result: Dict[str, Any] = {
            "inputs_embeds": torch.cat((prefix, token_embeddings), dim=1),
            "attention_mask": torch.cat((prefix_mask, token_mask), dim=1),
        }
        if include_input_ids:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            padding_id = tokenizer.pad_token_id
            if padding_id is None:
                padding_id = tokenizer.eos_token_id
            if padding_id is None:
                raise ValueError("The tokenizer must define a padding or end token.")
            prefix_ids = torch.full(
                prefix.shape[:2],
                int(padding_id),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            result["input_ids"] = torch.cat((prefix_ids, input_ids), dim=1)
        for key, value in inputs.items():
            if key not in {"input_ids", "attention_mask", "inputs_embeds"}:
                result[key] = value
        return result

    def _encode_model(self) -> nn.Module:
        root = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        for path in (("model",), ("language_model", "model"), ("language_model",)):
            current = root
            valid = True
            for name in path:
                if not hasattr(current, name):
                    valid = False
                    break
                current = getattr(current, name)
            if valid and isinstance(current, nn.Module):
                return current
        return root

    def _extract_last_hidden(self, outputs: Any) -> torch.Tensor:
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) > 0:
            return hidden_states[-1]
        raise AttributeError("Backbone output contains no usable hidden state.")

    def prefill(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        device: torch.device,
    ) -> PrefillState:
        prepared = self.prepare_multimodal_inputs(
            images,
            prompt,
            add_generation_prompt=False,
            padding_side="right",
        )
        model_inputs = self._move_inputs(prepared, device)
        with torch.inference_mode():
            outputs = self._encode_model()(
                **model_inputs,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            hidden = self._extract_last_hidden(outputs)
            token_mask = model_inputs["attention_mask"].bool()
            pooled = masked_mean(hidden, token_mask)
        return PrefillState(
            pooled_state=pooled.detach(),
            token_bank=hidden.detach(),
            token_mask=token_mask,
        )

    def _generation_stop_ids(self) -> List[int]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        result: List[int] = []
        for name in ("eos_token_id", "sep_token_id", "eod_id"):
            value = getattr(tokenizer, name, None)
            if isinstance(value, int) and value >= 0 and value not in result:
                result.append(int(value))
        for token_id in self._token_ids_for_tokens(
            ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>")
        ):
            if token_id not in result:
                result.append(token_id)
        if not result:
            raise ValueError("Unable to infer a generation stop token.")
        return result

    def generate(
        self,
        prompt: str,
        terminal_state: torch.Tensor,
        device: torch.device,
        max_new_tokens: int,
    ) -> str:
        base_inputs = self.prepare_text_generation_inputs(prompt, padding_side="left")
        inputs = self._move_inputs(base_inputs, device)
        inputs = self._prepend_terminal_state(inputs, terminal_state.to(device), include_input_ids=True)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        stop_ids = self._generation_stop_ids()
        padding_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=bool(self.config.generate_use_cache),
            eos_token_id=stop_ids if len(stop_ids) > 1 else stop_ids[0],
            pad_token_id=int(padding_id),
            repetition_penalty=float(self.config.generate_repetition_penalty),
            no_repeat_ngram_size=int(self.config.generate_no_repeat_ngram_size),
            renormalize_logits=True,
        )
        continuation = generated[0, prompt_length:] if generated.shape[1] > prompt_length else generated[0]
        stop_positions: List[int] = []
        for token_id in stop_ids:
            matches = continuation.eq(int(token_id)).nonzero(as_tuple=False)
            if matches.numel() > 0:
                stop_positions.append(int(matches[0].item()))
        if stop_positions:
            continuation = continuation[: min(stop_positions)]
        text = tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text.strip()
