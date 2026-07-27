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
