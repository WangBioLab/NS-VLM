from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """NS-VLM inference configuration."""

    checkpoint_path: str = ""
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "auto"
    dtype: str = "bf16"
    attn_implementation: str = "sdpa"
    local_files_only: bool = False

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    min_pixels: int = 224 * 224
    max_pixels: int = 448 * 448
    max_total_pixels: int = 4 * 384 * 384
    per_image_min_pixels: int = 224 * 224
    adaptive_resize: bool = True
    state_dim: int = 2048
    conclusion_dim: int = 512
    branch_count: int = 4
    policy_layers: int = 4
    policy_heads: int = 8
    updater_heads: int = 8
    transformer_ff_multiplier: int = 4
    transformer_dropout: float = 0.0

    max_state_depth: int = 10
    min_state_depth: int = 2
    kappa_max: float = 32.0
    branch_temperature: float = 1.0
    evidence_gate_temperature: float = 1.0
    numerical_epsilon: float = 1e-6
    stop_threshold: float = 0.5

    max_new_tokens: int = 256
    generate_use_cache: bool = True
    generate_no_repeat_ngram_size: int = 3
    generate_repetition_penalty: float = 1.05

    @classmethod
    def from_environment(cls) -> "InferenceConfig":
        defaults = cls()
        local_only = os.environ.get("NS_VLM_I_LOCAL_FILES_ONLY", "0").strip().lower()
        return cls(
            checkpoint_path=os.environ.get("NS_VLM_I_CHECKPOINT", "").strip(),
            model_name=os.environ.get("NS_VLM_I_MODEL", defaults.model_name).strip(),
            device=os.environ.get("NS_VLM_I_DEVICE", defaults.device).strip(),
            dtype=os.environ.get("NS_VLM_I_DTYPE", defaults.dtype).strip(),
            local_files_only=local_only in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        if self.state_dim < 1:
            raise ValueError("state_dim must be positive.")
        if self.conclusion_dim < 2:
            raise ValueError("conclusion_dim must be at least two.")
        if self.branch_count < 2:
            raise ValueError("branch_count must be at least two for branch deletion.")
        if self.policy_layers != 4:
            raise ValueError("NS-VLM uses a four-layer reasoning policy.")
        for name in ("policy_heads", "updater_heads"):
            heads = int(getattr(self, name))
            if heads < 1 or self.state_dim % heads != 0:
                raise ValueError(f"state_dim must be divisible by {name}.")
        if self.transformer_ff_multiplier < 1:
            raise ValueError("transformer_ff_multiplier must be positive.")
        if not 0.0 <= self.transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1).")
        if self.max_state_depth < 2:
            raise ValueError("max_state_depth must be at least two.")
        if not 2 <= self.min_state_depth <= self.max_state_depth:
            raise ValueError("min_state_depth must be between two and max_state_depth.")
        if self.kappa_max <= 0.0:
            raise ValueError("kappa_max must be positive.")
        if self.branch_temperature <= 0.0:
            raise ValueError("branch_temperature must be positive.")
        if self.evidence_gate_temperature <= 0.0:
            raise ValueError("evidence_gate_temperature must be positive.")
        if self.numerical_epsilon <= 0.0:
            raise ValueError("numerical_epsilon must be positive.")
        if not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError("stop_threshold must be in [0, 1].")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
