from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TokenChange:
    step: int
    context_position: int
    original_token_id: int
    replacement_token_id: int
    original_token: str
    replacement_token: str
    approximate_score: float
    objective_before: float
    objective_after: float


@dataclass
class AttackState:
    input_ids: torch.LongTensor
    changed_positions: frozenset[int]
    changes: tuple[TokenChange, ...]
    objective: float
    query_similarity: float
    target_similarity: float | None = None


@dataclass
class AttackResult:
    original_text: str
    attacked_text: str
    original_input_ids: list[int]
    attacked_input_ids: list[int]
    objective_before: float
    objective_after: float
    query_similarity_before: float
    query_similarity_after: float
    target_similarity_before: float | None
    target_similarity_after: float | None
    changes: list[TokenChange] = field(default_factory=list)
    stop_reason: str = "budget_exhausted"
    forward_passes: int = 0
    backward_passes: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["changes"] = [asdict(change) for change in self.changes]
        return result
