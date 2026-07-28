from __future__ import annotations

import unittest

from hotflip_rag.compare_attacks import (
    aggregate_results,
    parse_optional_bool,
    reconstruct_baseline_documents,
)


class CompareAttackTests(unittest.TestCase):
    def test_reconstructs_old_baseline_csv_context_without_json_column(self):
        passages = [
            {"document_id": "x:0", "title": "A", "text": "A: first", "source": "gold"},
            {"document_id": "x:1", "title": "B", "text": "B: second", "source": "distractor"},
            {"document_id": "x:2", "title": "C", "text": "C: third", "source": "gold"},
        ]
        row = {"retrieved_context": "C: third\n\nA: first"}
        reconstructed = reconstruct_baseline_documents(row, passages)
        self.assertEqual(
            [document["document_id"] for document in reconstructed],
            ["x:2", "x:0"],
        )

    def test_asr_uses_llm_judgments(self):
        results = [
            {
                "baseline": {"correct": True},
                "attacks": {
                    "untargeted": {
                        "gold_judge": {"correct": False},
                        "modified_document_retrieved": True,
                    },
                    "targeted": {
                        "gold_judge": {"correct": False},
                        "baseline_target_judge": {"correct": False},
                        "target_judge": {"correct": True},
                        "modified_document_retrieved": True,
                    },
                },
            },
            {
                "baseline": {"correct": False},
                "attacks": {
                    "untargeted": {
                        "gold_judge": {"correct": False},
                        "modified_document_retrieved": False,
                    },
                    "targeted": {
                        "gold_judge": {"correct": True},
                        "baseline_target_judge": {"correct": False},
                        "target_judge": {"correct": False},
                        "modified_document_retrieved": False,
                    },
                },
            },
        ]
        aggregate = aggregate_results(results, ["untargeted", "targeted"])
        self.assertEqual(aggregate["baseline_accuracy"], 0.5)
        self.assertEqual(aggregate["untargeted"]["asr"], 1.0)
        self.assertEqual(aggregate["untargeted"]["asr_eligible_examples"], 1)
        self.assertEqual(aggregate["targeted"]["asr"], 0.5)
        self.assertEqual(aggregate["targeted"]["asr_eligible_examples"], 2)

    def test_optional_bool_parser(self):
        self.assertIs(parse_optional_bool("True"), True)
        self.assertIs(parse_optional_bool("False"), False)
        self.assertIsNone(parse_optional_bool(""))


if __name__ == "__main__":
    unittest.main()
