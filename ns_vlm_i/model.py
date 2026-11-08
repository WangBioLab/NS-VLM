from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .backbone import QwenBackbone
from .config import InferenceConfig
from .utils import normalize_direction, validate_prompt


@dataclass(frozen=True)
class StepTrace:
    state_index: int
    branch_weights: Tuple[float, ...]
    evidence_uncertainty: float
    evidence_necessity: float
    concentration: float
    evidence_gate_mean: float
    stop_probability: float
    stopped: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelInferenceOutput:
    answer: str
    selected_step: int
    stopped_by_controller: bool
    traces: Tuple[StepTrace, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "selected_step": self.selected_step,
            "stopped_by_controller": self.stopped_by_controller,
            "traces": [trace.to_dict() for trace in self.traces],
        }


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class StateFusion(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.fusion = MLP(state_dim * 2, state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, evidence: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fusion(torch.cat((evidence, memory), dim=-1)))


class ReasoningPolicy(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden = int(config.state_dim)
        self.action_seed = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.token_types = nn.Parameter(torch.randn(3, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config.policy_heads),
            dim_feedforward=hidden * int(config.transformer_ff_multiplier),
            dropout=float(config.transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(config.policy_layers),
            norm=nn.LayerNorm(hidden),
        )
        self.action_head = MLP(hidden, hidden, hidden)

    def forward(
        self,
        state: torch.Tensor,
        evidence: torch.Tensor,
        memory: torch.Tensor,
        step_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action_token = self.action_seed.expand(batch_size, -1, -1).squeeze(1) + state
        tokens = torch.stack((action_token, evidence, memory), dim=1)
        tokens = tokens + self.token_types.unsqueeze(0) + step_embedding.unsqueeze(1)
        encoded = self.transformer(tokens)
        return F.layer_norm(self.action_head(encoded[:, 0]), (encoded.shape[-1],))


class EvidenceRetriever(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden = int(config.state_dim)
        self.branch_count = int(config.branch_count)
        self.temperature = float(config.branch_temperature)
        self.branch_embeddings = nn.Parameter(
            torch.randn(self.branch_count, hidden) * 0.02
        )
        self.query = MLP(hidden * 2, hidden, hidden)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=int(config.updater_heads),
            dropout=float(config.transformer_dropout),
            batch_first=True,
        )
        self.branch_score = MLP(hidden * 4, hidden, 1)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        token_bank: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_query = self.query(torch.cat((state, action), dim=-1)).unsqueeze(1)
        queries = self.norm(base_query + self.branch_embeddings.unsqueeze(0))
        attended, _ = self.cross_attention(
            query=queries,
            key=token_bank,
            value=token_bank,
            key_padding_mask=~token_mask.bool(),
            need_weights=False,
        )
        branches = self.norm(attended)

        state_expanded = state.unsqueeze(1).expand(-1, self.branch_count, -1)
        action_expanded = action.unsqueeze(1).expand(-1, self.branch_count, -1)
        branch_features = torch.cat(
            (
                state_expanded,
                action_expanded,
                branches,
                state_expanded * branches,
            ),
            dim=-1,
        )
        learned_logits = self.branch_score(branch_features).squeeze(-1)
        branch_logits = learned_logits.float()
        branch_weights = torch.softmax(branch_logits / self.temperature, dim=-1)
        aggregate = torch.einsum(
            "bk,bkh->bh",
            branch_weights.to(dtype=branches.dtype),
            branches,
        )
        return branches, branch_weights, aggregate


class EvidenceStateUpdater(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden = int(config.state_dim)
        self.temperature = float(config.evidence_gate_temperature)
        self.gate = MLP(hidden * 3, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        evidence: torch.Tensor,
        action: torch.Tensor,
        branches: torch.Tensor,
        aggregate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(
            self.gate(torch.cat((evidence, action, aggregate), dim=-1)) / self.temperature
        )
        committed = self.norm((1.0 - gate) * evidence + gate * aggregate)
        branch_states = self.norm(
            (1.0 - gate.unsqueeze(1)) * evidence.unsqueeze(1)
            + gate.unsqueeze(1) * branches
        )
        return committed, branch_states, gate


class MemoryUpdater(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        feature_dim = state_dim * 3
        self.proposal = MLP(feature_dim, state_dim, state_dim)
        self.gate = MLP(feature_dim, state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        memory: torch.Tensor,
        evidence: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((memory, evidence, action), dim=-1)
        proposal = torch.tanh(self.proposal(features))
        gate = torch.sigmoid(self.gate(features))
        return self.norm((1.0 - gate) * memory + gate * proposal)


class ConclusionHead(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden = int(config.state_dim)
        self.eps = float(config.numerical_epsilon)
        self.kappa_max = float(config.kappa_max)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.direction = nn.Linear(hidden, int(config.conclusion_dim))
        self.concentration = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        direction = normalize_direction(self.direction(features), self.eps)
        concentration = self.kappa_max * torch.sigmoid(
            self.concentration(features).squeeze(-1)
        )
        return direction, concentration


class ReasoningGate(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden = int(config.state_dim)
        self.kappa_max = float(config.kappa_max)
        self.network = MLP(hidden + 3, hidden, 1)

    def forward(
        self,
        state: torch.Tensor,
        uncertainty: torch.Tensor,
        concentration: torch.Tensor,
        necessity: torch.Tensor,
    ) -> torch.Tensor:
        normalized_kappa = (concentration / self.kappa_max).clamp(0.0, 1.0)
        features = torch.cat(
            (
                state,
                uncertainty.unsqueeze(-1),
                normalized_kappa.unsqueeze(-1),
                necessity.unsqueeze(-1),
            ),
            dim=-1,
        )
        return torch.sigmoid(self.network(features).squeeze(-1))


class NextStateVLM(nn.Module):
    """NS-VLM inference model."""

    architecture_name = "ns_vlm"

    def __init__(self, config: InferenceConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = QwenBackbone(config)
        if int(self.backbone.hidden_size) != int(config.state_dim):
            raise ValueError(
                "The configured state_dim must equal the Qwen language hidden size: "
                f"state_dim={config.state_dim}, backbone={self.backbone.hidden_size}."
            )

        hidden = int(config.state_dim)
        self.evidence_initialization = MLP(hidden, hidden, hidden)
        self.memory_initialization = MLP(hidden, hidden, hidden)
        self.state_fusion = StateFusion(hidden)
        self.step_embedding = nn.Embedding(int(config.max_state_depth) + 1, hidden)
        self.reasoning_policy = ReasoningPolicy(config)
        self.evidence_retriever = EvidenceRetriever(config)
        self.state_updater = EvidenceStateUpdater(config)
        self.memory_updater = MemoryUpdater(hidden)
        self.conclusion_head = ConclusionHead(config)
        self.reasoning_gate = ReasoningGate(config)
        self.terminal_token_types = nn.Parameter(torch.randn(3, hidden) * 0.02)
        self.terminal_norm = nn.LayerNorm(hidden)

    def _fuse_state(self, evidence: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.state_fusion(evidence, memory)

    def _initial_state(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        evidence = F.layer_norm(self.evidence_initialization(pooled), (pooled.shape[-1],))
        memory = F.layer_norm(self.memory_initialization(pooled), (pooled.shape[-1],))
        return evidence, memory

    def _provisional_metrics(
        self,
        previous_evidence: torch.Tensor,
        previous_memory: torch.Tensor,
        committed_evidence: torch.Tensor,
        branch_evidence: torch.Tensor,
        retrieved_branches: torch.Tensor,
        branch_weights: torch.Tensor,
        evidence_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = float(self.config.numerical_epsilon)
        branch_memory = previous_memory.unsqueeze(1).expand_as(branch_evidence)
        branch_states = self._fuse_state(branch_evidence, branch_memory)
        branch_directions, _branch_kappa = self.conclusion_head(branch_states)
        resultant = torch.einsum(
            "bk,bkd->bd",
            branch_weights.to(dtype=branch_directions.dtype),
            branch_directions,
        )
        uncertainty = (1.0 - torch.linalg.vector_norm(resultant.float(), dim=-1)).clamp(0.0, 1.0)

        committed_state = self._fuse_state(committed_evidence, previous_memory)
        committed_direction, concentration = self.conclusion_head(committed_state)
        necessity_terms: List[torch.Tensor] = []
        for removed_index in range(int(self.config.branch_count)):
            retained = branch_weights.clone()
            retained[:, removed_index] = 0.0
            retained = retained / retained.sum(dim=-1, keepdim=True).clamp_min(eps)
            counterfactual_aggregate = torch.einsum(
                "bk,bkh->bh",
                retained.to(dtype=retrieved_branches.dtype),
                retrieved_branches,
            )
            counterfactual_evidence = F.layer_norm(
                (1.0 - evidence_gate) * previous_evidence
                + evidence_gate * counterfactual_aggregate,
                (previous_evidence.shape[-1],),
            )
            counterfactual_state = self._fuse_state(
                counterfactual_evidence,
                previous_memory,
            )
            counterfactual_direction, counterfactual_kappa = self.conclusion_head(
                counterfactual_state
            )
            directional_support = counterfactual_kappa * (
                counterfactual_direction * committed_direction.detach()
            ).sum(dim=-1)
            relative_drop = (
                (concentration - directional_support)
                / (concentration + eps)
            ).clamp(0.0, 1.0)
            necessity_terms.append(branch_weights[:, removed_index] * relative_drop)
        necessity = torch.stack(necessity_terms, dim=1).sum(dim=1).clamp(0.0, 1.0)
        return uncertainty.to(dtype=committed_evidence.dtype), necessity, concentration

    def _terminal_tokens(
        self,
        evidence: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        state = self._fuse_state(evidence, memory)
        tokens = torch.stack((evidence, memory, state), dim=1)
        return self.terminal_norm(tokens + self.terminal_token_types.unsqueeze(0))

    def infer(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        device: torch.device,
        max_new_tokens: int,
    ) -> ModelInferenceOutput:
        question = validate_prompt(prompt)
        prefill = self.backbone.prefill(
            images,
            question,
            device,
        )
        reasoning_dtype = self.evidence_initialization.net[0].weight.dtype
        pooled = prefill.pooled_state.to(device=device, dtype=reasoning_dtype)
        token_bank = prefill.token_bank.to(device=device, dtype=reasoning_dtype)
        token_mask = prefill.token_mask.to(device=device)
        evidence, memory = self._initial_state(pooled)

        traces: List[StepTrace] = []
        stopped_by_controller = False
        selected_state_index = 1
        for transition_index in range(1, int(self.config.max_state_depth)):
            state = self._fuse_state(evidence, memory)
            step_ids = torch.full(
                (state.shape[0],),
                transition_index,
                dtype=torch.long,
                device=device,
            )
            action = self.reasoning_policy(
                state,
                evidence,
                memory,
                self.step_embedding(step_ids),
            )
            retrieved_branches, branch_weights, aggregate = self.evidence_retriever(
                state,
                action,
                token_bank,
                token_mask,
            )
            next_evidence, branch_evidence, evidence_gate = self.state_updater(
                evidence,
                action,
                retrieved_branches,
                aggregate,
            )
            uncertainty, necessity, concentration = self._provisional_metrics(
                evidence,
                memory,
                next_evidence,
                branch_evidence,
                retrieved_branches,
                branch_weights,
                evidence_gate,
            )
            next_memory = self.memory_updater(memory, next_evidence, action)
            next_state = self._fuse_state(next_evidence, next_memory)
            stop_probability = self.reasoning_gate(
                next_state,
                uncertainty,
                concentration,
                necessity,
            )

            state_index = transition_index + 1
            can_stop = state_index >= int(self.config.min_state_depth)
            is_last_state = state_index >= int(self.config.max_state_depth)
            controller_stop = bool(
                can_stop
                and stop_probability.item() >= float(self.config.stop_threshold)
            )
            stopped = controller_stop or is_last_state
            weights = tuple(float(value) for value in branch_weights[0].detach().cpu().tolist())
            traces.append(
                StepTrace(
                    state_index=state_index,
                    branch_weights=weights,
                    evidence_uncertainty=float(uncertainty.item()),
                    evidence_necessity=float(necessity.item()),
                    concentration=float(concentration.item()),
                    evidence_gate_mean=float(evidence_gate.mean().item()),
                    stop_probability=float(stop_probability.item()),
                    stopped=stopped,
                )
            )
            evidence, memory = next_evidence, next_memory
            selected_state_index = state_index
            if stopped:
                stopped_by_controller = controller_stop
                break

        terminal_state = self._terminal_tokens(evidence, memory)
        answer = self.backbone.generate(
            question,
            terminal_state,
            device,
            max_new_tokens=int(max_new_tokens),
        ).strip()
        return ModelInferenceOutput(
            answer=answer if answer else "Uncertain.",
            selected_step=selected_state_index,
            stopped_by_controller=stopped_by_controller,
            traces=tuple(traces),
        )
