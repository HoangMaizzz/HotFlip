from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch
from torch import nn

from hotflip_rag.hotflip import (
    ContrieverHotFlipAttacker,
    HotFlipConfig,
    mean_pool,
)
from hotflip_rag.types import TokenChange


class TinyTokenizer:
    def __init__(self):
        self.tokens = ["[PAD]", "[CLS]", "[SEP]", "alpha", "bravo", "cider", "delta", "echo"]
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.all_special_ids = [0, 1, 2]

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id):
        return self.tokens[int(token_id)]

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        words = [
            self.tokens[int(token_id)]
            for token_id in ids
            if not skip_special_tokens or int(token_id) not in self.all_special_ids
        ]
        return " ".join(words)

    def __call__(
        self, text, return_tensors="pt", truncation=True, max_length=512, padding=False
    ):
        texts = [text] if isinstance(text, str) else text
        rows = []
        for value in texts:
            ids = [1] + [self.vocab[word] for word in value.split()] + [2]
            rows.append(ids[:max_length])
        width = max(len(row) for row in rows)
        padded = [row + [0] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return TinyBatch(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(mask, dtype=torch.long),
        )


class TinyBatch(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

    def to(self, device):
        return TinyBatch(**{key: value.to(device) for key, value in self.items()})


class TinyRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(8, 3)
        with torch.no_grad():
            self.embeddings.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.8, 0.2, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
            )

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden)


class HotFlipTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.model = TinyRetriever()

    def attacker(self, **overrides):
        values = dict(
            max_token_changes=1,
            candidate_vocab_size=None,
            preserve_leading_space=False,
            preserve_token_class=True,
            disallow_punctuation_replacement=True,
            disallow_numeric_replacement=True,
            candidates_per_state=20,
            hotflip_top_k=20,
            exact_rerank=True,
        )
        values.update(overrides)
        return ContrieverHotFlipAttacker(
            self.model, self.tokenizer, HotFlipConfig(**values), "cpu"
        )

    def test_numeric_replacement_is_enabled_by_default(self):
        self.assertFalse(HotFlipConfig().disallow_numeric_replacement)

    def test_render_preserves_unmodified_capitalization(self):
        attacker = self.attacker()
        change = TokenChange(
            step=1,
            context_position=1,
            original_token_id=3,
            replacement_token_id=6,
            original_token="alpha",
            replacement_token="delta",
            approximate_score=1.0,
            objective_before=0.0,
            objective_after=1.0,
        )
        attacked_ids = torch.tensor([[1, 6, 4, 2]], dtype=torch.long)
        rendered = attacker.render_attacked_text(
            "Alpha Bravo",
            attacked_ids,
            (change,),
            [(0, 0), (0, 5), (6, 11), (0, 0)],
        )
        self.assertEqual(rendered, "delta Bravo")

    def test_render_with_no_changes_returns_original_exactly(self):
        attacker = self.attacker()
        attacked_ids = torch.tensor([[1, 3, 4, 2]], dtype=torch.long)
        rendered = attacker.render_attacked_text(
            "Alpha Bravo", attacked_ids, (), None
        )
        self.assertEqual(rendered, "Alpha Bravo")

    def test_hotflip_formula(self):
        weight = self.model.embeddings.weight.detach()
        grad = torch.tensor([0.25, -0.5, 1.0])
        current, candidate = 3, 6
        direct = grad @ (weight[candidate] - weight[current])
        decomposed = grad @ weight[candidate] - grad @ weight[current]
        self.assertTrue(torch.allclose(direct, decomposed))

    def test_context_mask_excludes_special_tokens(self):
        attacker = self.attacker()
        ids, attention = attacker.tokenize_context("alpha bravo")
        mask = attacker.build_modifiable_mask(ids, attention)
        self.assertEqual(mask.tolist(), [[False, True, True, False]])

    def test_untargeted_increases_objective_and_decreases_similarity(self):
        attacker = self.attacker(attack_mode="untargeted")
        result = attacker.attack("alpha", "alpha bravo")
        self.assertGreater(result.objective_after, result.objective_before)
        self.assertLess(result.query_similarity_after, result.query_similarity_before)

    def test_retrieval_preserving_untargeted_uses_gold_answer_embedding(self):
        attacker = self.attacker(
            attack_mode="untargeted", untargeted_answer_weight=2.0
        )
        result = attacker.attack(
            "alpha", "alpha bravo", avoid_answer="delta"
        )
        self.assertIsNotNone(result.target_similarity_before)
        self.assertIsNotNone(result.target_similarity_after)
        self.assertGreaterEqual(result.objective_after, result.objective_before)
        self.assertLessEqual(len({change.context_position for change in result.changes}), 1)

    def test_targeted_moves_context_toward_target(self):
        attacker = self.attacker(attack_mode="targeted", target_weight=5.0)
        result = attacker.attack("alpha", "alpha bravo", target_answer="delta")
        self.assertGreater(result.objective_after, result.objective_before)
        self.assertGreater(result.target_similarity_after, result.target_similarity_before)

    def test_budget_and_no_special_token_change(self):
        attacker = self.attacker(attack_mode="untargeted", max_token_changes=2)
        result = attacker.attack("alpha", "alpha bravo delta")
        changed = [
            index
            for index, (before, after) in enumerate(
                zip(result.original_input_ids, result.attacked_input_ids)
            )
            if before != after
        ]
        self.assertLessEqual(len(changed), 2)
        self.assertNotIn(0, changed)
        self.assertNotIn(len(result.original_input_ids) - 1, changed)

    def test_trace_prints_each_search_step(self):
        attacker = self.attacker(
            attack_mode="untargeted",
            trace=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            attacker.attack(
                "alpha", "alpha bravo", avoid_answer="delta"
            )
        trace = output.getvalue()
        self.assertIn("[HOTFLIP TRACE] step=0", trace)
        self.assertIn("[HOTFLIP TRACE] step=1", trace)
        self.assertIn("[HOTFLIP TRACE] finished", trace)

    def test_mean_pool_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 1.0]])))


if __name__ == "__main__":
    unittest.main()
