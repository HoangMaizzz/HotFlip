from __future__ import annotations

import unittest

from hotflip_rag.baseline import hotpot_passages
from hotflip_rag.generate_targets import validate_target_token_ids
from hotflip_rag.pipeline import QAGenerator


class BaselineTests(unittest.TestCase):
    def test_wrong_target_tokens_must_be_inside_candidate_vocabulary(self):
        class RetrieverTokenizer:
            unk_token_id = 100

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [200, 29999] if text == "valid" else [30001]}

            def convert_ids_to_tokens(self, token_ids):
                return [str(token_id) for token_id in token_ids]

        tokenizer = RetrieverTokenizer()
        self.assertTrue(
            validate_target_token_ids("valid", tokenizer, 30000)["valid"]
        )
        self.assertFalse(
            validate_target_token_ids("invalid", tokenizer, 30000)["valid"]
        )

    def test_prompt_requires_answer_only(self):
        prompt = QAGenerator.build_prompt("Where?", "Place: Here.")
        self.assertIn("Return exactly one shortest final-answer span", prompt)
        self.assertIn("Do not explain", prompt)
        self.assertTrue(prompt.endswith("<answer>"))

    def test_generated_explanation_after_closing_tag_is_discarded(self):
        answer = QAGenerator.extract_final_answer(
            "1969–1974</answer> This is supported by the context."
        )
        self.assertEqual(answer, "1969–1974")

    def test_opening_and_closing_answer_tags_are_removed(self):
        answer = QAGenerator.extract_final_answer(
            "<answer>Richard Nixon</answer>\nExtra explanation"
        )
        self.assertEqual(answer, "Richard Nixon")

    def test_judge_prompt_requires_yes_or_no(self):
        prompt = QAGenerator.build_judge_prompt("Who?", "John Smith", "Smith")
        self.assertIn("Return exactly one token: YES or NO", prompt)
        self.assertIn("merely containing the reference answer is NOT sufficient", prompt)
        self.assertIn("another clause adds a conflicting", prompt)
        self.assertIn("1969-1974, 1974-1978", prompt)
        self.assertIn("Reference answer: John Smith", prompt)
        self.assertIn("Predicted answer: Smith", prompt)

    def test_exact_match_is_accepted_without_model_judge(self):
        generator = QAGenerator.__new__(QAGenerator)
        judgment = generator.judge_answer(
            "Which city?", "The Paris.", "paris"
        )
        self.assertTrue(judgment["correct"])
        self.assertEqual(judgment["method"], "normalized_exact_match")
        self.assertEqual(judgment["raw"], "EXACT_MATCH")

    def test_hotpot_passages_keep_documents_separate_and_hide_no_candidates(self):
        item = {
            "context": {
                "title": ["Gold A", "Distractor", "Gold B"],
                "sentences": [["Fact A."], ["Noise."], ["Fact B."]],
            },
            "supporting_facts": {
                "title": ["Gold A", "Gold B"],
                "sent_id": [0, 0],
            },
        }
        passages = hotpot_passages(item)
        self.assertEqual(len(passages), 3)
        self.assertEqual([p["source"] for p in passages], ["gold", "distractor", "gold"])
        self.assertEqual(passages[0]["text"], "Gold A: Fact A.")


if __name__ == "__main__":
    unittest.main()
