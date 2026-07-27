from __future__ import annotations

import unittest

from hotflip_rag.baseline import hotpot_passages
from hotflip_rag.pipeline import QAGenerator


class BaselineTests(unittest.TestCase):
    def test_prompt_requires_answer_only(self):
        prompt = QAGenerator.build_prompt("Where?", "Place: Here.")
        self.assertIn("Return only the final answer", prompt)
        self.assertIn("no explanation", prompt)
        self.assertTrue(prompt.endswith("Final answer:"))

    def test_judge_prompt_requires_yes_or_no(self):
        prompt = QAGenerator.build_judge_prompt("Who?", "John Smith", "Smith")
        self.assertIn("Return exactly one token: YES or NO", prompt)
        self.assertIn("Accept extra explanation", prompt)
        self.assertIn("contains the correct reference answer", prompt)
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
