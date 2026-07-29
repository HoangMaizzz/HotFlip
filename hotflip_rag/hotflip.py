from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F

from .types import AttackResult, AttackState, TokenChange


AttackMode = Literal["untargeted", "targeted"]
SearchStrategy = Literal["greedy", "beam"]


@dataclass
class HotFlipConfig:
    attack_mode: AttackMode = "untargeted"
    search_strategy: SearchStrategy = "greedy"
    max_token_changes: int = 3
    beam_width: int = 3
    hotflip_top_k: int = 20
    candidates_per_state: int = 20
    candidate_policy: str = "tokenizer_safe"
    candidate_vocab_size: int | None = 30000
    exact_rerank: bool = True
    min_objective_improvement: float = 0.0
    preserve_token_class: bool = True
    preserve_leading_space: bool = True
    disallow_punctuation_replacement: bool = True
    disallow_numeric_replacement: bool = False
    allow_revisit_position: bool = False
    target_weight: float = 1.0
    untargeted_answer_weight: float = 1.0
    score_chunk_size: int = 2048
    max_context_tokens: int = 512
    trace: bool = False

    def validate(self) -> None:
        if self.attack_mode not in {"untargeted", "targeted"}:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        if self.search_strategy not in {"greedy", "beam"}:
            raise ValueError(f"Unsupported search strategy: {self.search_strategy}")
        if self.max_token_changes < 0 or self.hotflip_top_k < 1:
            raise ValueError("Attack budget must be non-negative and hotflip_top_k positive")
        if self.attack_mode == "targeted" and self.target_weight <= 0:
            raise ValueError("target_weight must be positive")
        if self.untargeted_answer_weight <= 0:
            raise ValueError("untargeted_answer_weight must be positive")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class CandidateFilter:
    def __init__(self, tokenizer, config: HotFlipConfig):
        self.tokenizer = tokenizer
        self.config = config
        self._special_ids = set(getattr(tokenizer, "all_special_ids", []))
        self._cache: dict[int, tuple[str, str, bool]] = {}
        get_vocab = getattr(tokenizer, "get_vocab", None)
        vocabulary = get_vocab() if callable(get_vocab) else {}
        self._uses_wordpiece = any(
            str(token).startswith("##") for token in vocabulary
        )

    def _properties(self, token_id: int) -> tuple[str, str, bool]:
        if token_id not in self._cache:
            raw = self.tokenizer.convert_ids_to_tokens(int(token_id))
            decoded = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            if self._uses_wordpiece:
                # WordPiece marks continuation pieces with ##. Treat word-start
                # and continuation pieces as different spacing classes.
                leading_space = not str(raw).startswith("##")
            else:
                leading_space = bool(decoded[:1].isspace()) or str(raw).startswith(
                    ("Ġ", "▁")
                )
            self._cache[token_id] = (str(decoded), token_class(str(decoded)), leading_space)
        return self._cache[token_id]

    def valid_candidate_ids(
        self, current_token_id: int, base_candidate_ids: torch.LongTensor
    ) -> torch.LongTensor:
        current_text, current_class, current_leading = self._properties(current_token_id)
        valid: list[int] = []
        for candidate in base_candidate_ids.detach().cpu().tolist():
            candidate = int(candidate)
            if candidate == current_token_id or candidate in self._special_ids:
                continue
            text, cls, leading = self._properties(candidate)
            if not text or not text.strip() or any(unicodedata.category(ch) == "Cc" for ch in text):
                continue
            if self.config.preserve_token_class and cls != current_class:
                continue
            if self.config.preserve_leading_space and leading != current_leading:
                continue
            if self.config.disallow_punctuation_replacement and current_class == "punct":
                continue
            if self.config.disallow_numeric_replacement and current_class == "numeric":
                continue
            valid.append(candidate)
        return torch.tensor(valid, dtype=torch.long, device=base_candidate_ids.device)


def token_class(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if all(ch.isalpha() for ch in stripped):
        return "alpha"
    if all(ch.isdigit() for ch in stripped):
        return "numeric"
    if all(unicodedata.category(ch).startswith("P") for ch in stripped):
        return "punct"
    return "mixed"


class ContrieverHotFlipAttacker:
    """HotFlip over Gold Context tokens using the Contriever retrieval objective.

    Sign convention: every search path *maximizes* ``objective``.
    When a gold-answer embedding is supplied, the retrieval-preserving
    untargeted objective is ``cos(query, context) -
    untargeted_answer_weight*cos(gold_answer, context)``. Targeted objective is
    ``cos(query, context) + target_weight*cos(target, context)``. It keeps the
    poisoned Gold Context retrievable while moving its representation toward the
    attacker-selected target answer.
    """

    def __init__(self, model, tokenizer, config: HotFlipConfig, device: str | torch.device):
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_layer = self.model.get_input_embeddings()
        self.candidate_filter = CandidateFilter(tokenizer, config)
        self.forward_passes = 0
        self.backward_passes = 0

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        output = self.model(**batch)
        self.forward_passes += 1
        return F.normalize(mean_pool(output.last_hidden_state, batch["attention_mask"]), dim=-1)

    def tokenize_context(self, text: str) -> tuple[torch.LongTensor, torch.LongTensor]:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens,
        )
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

    def context_offsets(
        self, text: str, expected_length: int
    ) -> list[tuple[int, int]] | None:
        """Return fast-tokenizer character offsets when the tokenizer supports them."""
        try:
            batch = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_tokens,
                return_offsets_mapping=True,
            )
        except (TypeError, ValueError, NotImplementedError):
            return None
        mapping = batch.get("offset_mapping")
        if mapping is None:
            return None
        offsets = [
            (int(start), int(end)) for start, end in mapping[0].tolist()
        ]
        return offsets if len(offsets) == expected_length else None

    def render_attacked_text(
        self,
        original_text: str,
        attacked_ids: torch.LongTensor,
        changes: tuple[TokenChange, ...],
        offsets: list[tuple[int, int]] | None,
    ) -> str:
        """Preserve original formatting and replace only actual HotFlip spans."""
        if not changes:
            return original_text
        if offsets is None:
            return self.tokenizer.decode(
                attacked_ids[0], skip_special_tokens=True
            )
        replacements: list[tuple[int, int, str]] = []
        for change in changes:
            position = change.context_position
            if position >= len(offsets):
                continue
            start, end = offsets[position]
            if end <= start:
                continue
            replacement_id = int(attacked_ids[0, position])
            raw = str(self.tokenizer.convert_ids_to_tokens(replacement_id))
            replacement = raw
            for prefix in ("##", "Ġ", "▁"):
                if replacement.startswith(prefix):
                    replacement = replacement[len(prefix):]
                    break
            if not replacement:
                replacement = self.tokenizer.decode(
                    [replacement_id], skip_special_tokens=True
                ).strip()
            replacements.append((start, end, replacement))
        rendered = original_text
        for start, end, replacement in sorted(replacements, reverse=True):
            rendered = rendered[:start] + replacement + rendered[end:]
        return rendered

    def build_modifiable_mask(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.BoolTensor:
        mask = attention_mask.bool().clone()
        for special_id in getattr(self.tokenizer, "all_special_ids", []):
            mask &= input_ids.ne(int(special_id))
        for position, token_id in enumerate(input_ids[0].tolist()):
            text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            cls = token_class(text)
            if not text.strip() or (self.config.disallow_punctuation_replacement and cls == "punct"):
                mask[0, position] = False
            if self.config.disallow_numeric_replacement and cls == "numeric":
                mask[0, position] = False
        return mask

    def _objective_from_embedding(
        self, context_embedding: torch.Tensor, query_embedding: torch.Tensor,
        target_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        query_similarity = F.cosine_similarity(context_embedding, query_embedding).mean()
        target_similarity = None
        if self.config.attack_mode == "untargeted":
            if target_embedding is None:
                # Backward-compatible representation-only mode.
                objective = -query_similarity
            else:
                target_similarity = F.cosine_similarity(
                    context_embedding, target_embedding
                ).mean()
                objective = (
                    query_similarity
                    - self.config.untargeted_answer_weight * target_similarity
                )
        else:
            if target_embedding is None:
                raise ValueError("target_answer is required for targeted attack")
            target_similarity = F.cosine_similarity(context_embedding, target_embedding).mean()
            objective = query_similarity + self.config.target_weight * target_similarity
        return objective, query_similarity, target_similarity

    def objective_and_gradient(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor,
        query_embedding: torch.Tensor, target_embedding: torch.Tensor | None,
    ) -> tuple[float, float, float | None, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        inputs_embeds = self.embedding_layer(input_ids).detach().requires_grad_(True)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        self.forward_passes += 1
        context_embedding = F.normalize(mean_pool(output.last_hidden_state, attention_mask), dim=-1)
        objective, query_sim, target_sim = self._objective_from_embedding(
            context_embedding, query_embedding, target_embedding
        )
        objective.backward()
        self.backward_passes += 1
        if inputs_embeds.grad is None:
            raise RuntimeError("Could not obtain gradients for context input embeddings")
        return (
            float(objective.detach()),
            float(query_sim.detach()),
            None if target_sim is None else float(target_sim.detach()),
            inputs_embeds.grad.detach()[0],
        )

    @torch.no_grad()
    def exact_objective(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor,
        query_embedding: torch.Tensor, target_embedding: torch.Tensor | None,
    ) -> tuple[float, float, float | None]:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.forward_passes += 1
        context_embedding = F.normalize(mean_pool(output.last_hidden_state, attention_mask), dim=-1)
        objective, query_sim, target_sim = self._objective_from_embedding(
            context_embedding, query_embedding, target_embedding
        )
        return (
            float(objective),
            float(query_sim),
            None if target_sim is None else float(target_sim),
        )

    def candidate_vocabulary(self) -> torch.LongTensor:
        vocab_size = int(self.embedding_layer.weight.shape[0])
        ids = torch.arange(vocab_size, device=self.device)
        if self.config.candidate_vocab_size and self.config.candidate_vocab_size < vocab_size:
            # Token IDs are only a deterministic compact-vocabulary fallback, not a
            # claim that IDs encode corpus frequency.
            ids = ids[: self.config.candidate_vocab_size]
        return ids

    def score_replacements(
        self, input_ids: torch.LongTensor, gradients: torch.Tensor,
        modifiable_mask: torch.BoolTensor, changed_positions: frozenset[int],
    ) -> list[tuple[float, int, int]]:
        weight = self.embedding_layer.weight.detach().float()
        base_candidates = self.candidate_vocabulary()
        scored: list[tuple[float, int, int]] = []
        positions = torch.where(modifiable_mask[0])[0].tolist()
        for position in positions:
            if position in changed_positions and not self.config.allow_revisit_position:
                continue
            current_id = int(input_ids[0, position])
            valid_ids = self.candidate_filter.valid_candidate_ids(current_id, base_candidates)
            if valid_ids.numel() == 0:
                continue
            grad = gradients[position].float()
            current_score = torch.dot(weight[current_id], grad)
            for start in range(0, valid_ids.numel(), self.config.score_chunk_size):
                chunk = valid_ids[start : start + self.config.score_chunk_size]
                values = weight[chunk] @ grad - current_score
                keep = min(self.config.hotflip_top_k, values.numel())
                top_values, top_indices = torch.topk(values, k=keep)
                for value, index in zip(top_values.tolist(), top_indices.tolist()):
                    scored.append((float(value), int(position), int(chunk[index])))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[: self.config.candidates_per_state]

    def _expand_state(
        self, state: AttackState, attention_mask: torch.LongTensor,
        modifiable_mask: torch.BoolTensor, query_embedding: torch.Tensor,
        target_embedding: torch.Tensor | None, step: int,
    ) -> list[AttackState]:
        before, _, _, gradients = self.objective_and_gradient(
            state.input_ids, attention_mask, query_embedding, target_embedding
        )
        candidates = self.score_replacements(
            state.input_ids, gradients, modifiable_mask, state.changed_positions
        )
        children: list[AttackState] = []
        for approx, position, replacement_id in candidates:
            child_ids = state.input_ids.clone()
            original_id = int(child_ids[0, position])
            child_ids[0, position] = replacement_id
            if self.config.exact_rerank:
                objective, query_sim, target_sim = self.exact_objective(
                    child_ids, attention_mask, query_embedding, target_embedding
                )
            else:
                objective = before + approx
                query_sim, target_sim = state.query_similarity, state.target_similarity
            change = TokenChange(
                step=step,
                context_position=position,
                original_token_id=original_id,
                replacement_token_id=replacement_id,
                original_token=self.tokenizer.decode([original_id]),
                replacement_token=self.tokenizer.decode([replacement_id]),
                approximate_score=approx,
                objective_before=before,
                objective_after=objective,
            )
            children.append(
                AttackState(
                    input_ids=child_ids,
                    changed_positions=state.changed_positions | {position},
                    changes=state.changes + (change,),
                    objective=objective,
                    query_similarity=query_sim,
                    target_similarity=target_sim,
                )
            )
        children.sort(key=lambda state_: state_.objective, reverse=True)
        return children

    def attack(
        self,
        question: str,
        gold_context: str,
        target_answer: str | None = None,
        avoid_answer: str | None = None,
    ) -> AttackResult:
        self.forward_passes = self.backward_passes = 0
        query_embedding = self.encode_text(question).detach()
        target_embedding = None
        if self.config.attack_mode == "targeted":
            if not target_answer:
                raise ValueError("target_answer is required in targeted mode")
            target_embedding = self.encode_text(target_answer).detach()
        elif avoid_answer:
            target_embedding = self.encode_text(avoid_answer).detach()
        input_ids, attention_mask = self.tokenize_context(gold_context)
        offsets = self.context_offsets(gold_context, input_ids.shape[1])
        original_ids = input_ids.clone()
        modifiable_mask = self.build_modifiable_mask(input_ids, attention_mask)
        initial_obj, initial_query_sim, initial_target_sim = self.exact_objective(
            input_ids, attention_mask, query_embedding, target_embedding
        )
        initial = AttackState(
            input_ids=input_ids,
            changed_positions=frozenset(),
            changes=(),
            objective=initial_obj,
            query_similarity=initial_query_sim,
            target_similarity=initial_target_sim,
        )
        beam = [initial]
        stop_reason = "budget_exhausted"
        if self.config.trace:
            similarity_label = (
                "gold_answer_cos"
                if self.config.attack_mode == "untargeted"
                else "target_answer_cos"
            )
            print(
                "[HOTFLIP TRACE] "
                f"mode={self.config.attack_mode} "
                f"strategy={self.config.search_strategy} "
                f"beam_width={self.config.beam_width} "
                f"budget={self.config.max_token_changes}",
                flush=True,
            )
            print(
                "[HOTFLIP TRACE] "
                f"step=0 objective={initial.objective:.6f} "
                f"query_cos={initial.query_similarity:.6f} "
                f"{similarity_label}="
                f"{initial.target_similarity if initial.target_similarity is not None else 'N/A'} "
                f"tokens={input_ids.shape[1]} "
                f"modifiable={int(modifiable_mask.sum())}",
                flush=True,
            )
        for step in range(1, self.config.max_token_changes + 1):
            if self.config.trace:
                print(
                    "[HOTFLIP TRACE] "
                    f"step={step} expanding_parents={len(beam)} "
                    f"candidates_per_parent<={self.config.candidates_per_state}",
                    flush=True,
                )
            expanded: list[AttackState] = []
            for state in beam:
                expanded.extend(
                    self._expand_state(
                        state, attention_mask, modifiable_mask, query_embedding,
                        target_embedding, step,
                    )
                )
            if not expanded:
                stop_reason = "no_valid_candidate"
                if self.config.trace:
                    print(
                        f"[HOTFLIP TRACE] step={step} stop={stop_reason}",
                        flush=True,
                    )
                break
            deduplicated: dict[tuple[int, ...], AttackState] = {}
            for state in expanded:
                key = tuple(state.input_ids[0].tolist())
                if key not in deduplicated or state.objective > deduplicated[key].objective:
                    deduplicated[key] = state
            ranked = sorted(deduplicated.values(), key=lambda state: state.objective, reverse=True)
            best_previous = max(state.objective for state in beam)
            if ranked[0].objective - best_previous < self.config.min_objective_improvement:
                stop_reason = "insufficient_improvement"
                if self.config.trace:
                    print(
                        "[HOTFLIP TRACE] "
                        f"step={step} stop={stop_reason} "
                        f"best_delta={ranked[0].objective - best_previous:.6f}",
                        flush=True,
                    )
                break
            width = 1 if self.config.search_strategy == "greedy" else self.config.beam_width
            beam = ranked[:width]
            if self.config.trace:
                print(
                    "[HOTFLIP TRACE] "
                    f"step={step} expanded={len(expanded)} "
                    f"unique={len(ranked)} kept={len(beam)}",
                    flush=True,
                )
                for rank, state in enumerate(beam, 1):
                    change = state.changes[-1]
                    similarity_label = (
                        "gold_answer_cos"
                        if self.config.attack_mode == "untargeted"
                        else "target_answer_cos"
                    )
                    target_value = (
                        f"{state.target_similarity:.6f}"
                        if state.target_similarity is not None
                        else "N/A"
                    )
                    print(
                        "[HOTFLIP TRACE] "
                        f"step={step} beam_rank={rank} "
                        f"position={change.context_position} "
                        f"{change.original_token!r}->{change.replacement_token!r} "
                        f"approx={change.approximate_score:.6f} "
                        f"objective={state.objective:.6f} "
                        f"query_cos={state.query_similarity:.6f} "
                        f"{similarity_label}={target_value}",
                        flush=True,
                    )
        best = max(beam, key=lambda state: state.objective)
        if len(best.changed_positions) > self.config.max_token_changes:
            raise AssertionError("HotFlip exceeded the unique-position attack budget")
        changed = original_ids.ne(best.input_ids)
        if torch.any(changed & ~modifiable_mask):
            raise AssertionError("HotFlip modified a token outside the Gold Context mask")
        attacked_text = self.render_attacked_text(
            gold_context, best.input_ids, best.changes, offsets
        )
        if self.config.trace:
            print(
                "[HOTFLIP TRACE] "
                f"finished stop={stop_reason} changes={len(best.changes)} "
                f"objective={initial.objective:.6f}->{best.objective:.6f} "
                f"query_cos={initial.query_similarity:.6f}->"
                f"{best.query_similarity:.6f} "
                f"forward_passes={self.forward_passes} "
                f"backward_passes={self.backward_passes}",
                flush=True,
            )
        return AttackResult(
            original_text=gold_context,
            attacked_text=attacked_text,
            original_input_ids=original_ids[0].tolist(),
            attacked_input_ids=best.input_ids[0].tolist(),
            objective_before=initial.objective,
            objective_after=best.objective,
            query_similarity_before=initial.query_similarity,
            query_similarity_after=best.query_similarity,
            target_similarity_before=initial.target_similarity,
            target_similarity_after=best.target_similarity,
            changes=list(best.changes),
            stop_reason=stop_reason,
            forward_passes=self.forward_passes,
            backward_passes=self.backward_passes,
        )
