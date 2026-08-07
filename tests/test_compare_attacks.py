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
                "baseline": {"correct": True, "any_gold_retrieved": True},
                "attacks": {
                    "untargeted": {
                        "gold_judge": {"correct": False},
                        "attacked_vs_baseline_judge": {"correct": False},
                        "attack_success": True,
                        "relaxed_attack_success": True,
                        "modified_document_retrieved": True,
                        "any_gold_retrieved": True,
                    },
                    "targeted": {
                        "gold_judge": {"correct": False},
                        "baseline_target_judge": {"correct": False},
                        "target_judge": {"correct": True},
                        "strict_attack_success": True,
                        "relaxed_attack_success": True,
                        "modified_document_retrieved": True,
                        "any_gold_retrieved": True,
                    },
                },
            },
            {
                "baseline": {"correct": False, "any_gold_retrieved": False},
                "attacks": {
                    "untargeted": {
                        "gold_judge": {"correct": False},
                        "attacked_vs_baseline_judge": {"correct": False},
                        "attack_success": False,
                        "relaxed_attack_success": True,
                        "modified_document_retrieved": True,
                        "any_gold_retrieved": True,
                    },
                    "targeted": {
                        "gold_judge": {"correct": True},
                        "baseline_target_judge": {"correct": False},
                        "target_judge": {"correct": False},
                        "strict_attack_success": False,
                        "relaxed_attack_success": False,
                        "modified_document_retrieved": False,
                        "any_gold_retrieved": False,
                    },
                },
            },
        ]
        aggregate = aggregate_results(results, ["untargeted", "targeted"])
        self.assertEqual(aggregate["baseline_accuracy"], 0.5)
        self.assertEqual(aggregate["untargeted"]["asr"], 1.0)
        self.assertEqual(aggregate["untargeted"]["asr_overall"], 0.5)
        self.assertEqual(aggregate["untargeted"]["relaxed_asr_eligible"], 1.0)
        self.assertEqual(
            aggregate["untargeted"]["relaxed_asr_on_baseline_incorrect"], 1.0
        )
        self.assertEqual(
            aggregate["untargeted"][
                "baseline_no_gold_then_any_gold_retrieval_rate"
            ],
            1.0,
        )
        self.assertEqual(
            aggregate["untargeted"][
                "baseline_no_gold_then_modified_gold_retrieval_rate"
            ],
            1.0,
        )
        self.assertEqual(
            aggregate["untargeted"]["asr_on_baseline_correct"], 1.0
        )
        self.assertEqual(aggregate["untargeted"]["asr_eligible_examples"], 1)
        self.assertEqual(aggregate["targeted"]["asr"], 0.5)
        self.assertEqual(aggregate["targeted"]["asr_overall"], 0.5)
        self.assertEqual(
            aggregate["targeted"]["asr_on_baseline_correct"], 1.0
        )
        self.assertEqual(aggregate["targeted"]["asr_eligible_examples"], 2)
        self.assertEqual(
            aggregate["targeted"]["asr_on_baseline_correct_examples"], 1
        )
        self.assertEqual(
            aggregate["targeted"]["relaxed_targeted_asr_eligible"], 0.5
        )
        self.assertEqual(
            aggregate["targeted"]["strict_targeted_asr_eligible"], 0.5
        )

    def test_targeted_strict_success_requires_modified_document_retrieval(self):
        results = [{
            "baseline": {"correct": True, "any_gold_retrieved": True},
            "attacks": {
                "targeted": {
                    "gold_judge": {"correct": False},
                    "baseline_target_judge": {"correct": False},
                    "target_judge": {"correct": True},
                    "strict_attack_success": False,
                    "relaxed_attack_success": True,
                    "modified_document_retrieved": False,
                    "any_gold_retrieved": True,
                }
            },
        }]
        aggregate = aggregate_results(results, ["targeted"])
        self.assertEqual(
            aggregate["targeted"]["relaxed_targeted_asr_eligible"], 1.0
        )
        self.assertEqual(
            aggregate["targeted"]["strict_targeted_asr_eligible"], 0.0
        )

    def test_optional_bool_parser(self):
        self.assertIs(parse_optional_bool("True"), True)
        self.assertIs(parse_optional_bool("False"), False)
        self.assertIsNone(parse_optional_bool(""))

    def test_untargeted_failure_when_modified_document_is_not_retrieved(self):
        results = [{
            "baseline": {"correct": True, "any_gold_retrieved": True},
            "attacks": {
                "untargeted": {
                    "gold_judge": {"correct": False},
                    "attacked_vs_baseline_judge": {"correct": False},
                    "attack_success": False,
                    "relaxed_attack_success": False,
                    "modified_document_retrieved": False,
                    "any_gold_retrieved": False,
                }
            },
        }]
        aggregate = aggregate_results(results, ["untargeted"])
        self.assertEqual(aggregate["untargeted"]["asr"], 0.0)
        self.assertEqual(aggregate["untargeted"]["relaxed_asr_eligible"], 0.0)

    def test_relaxed_success_rejects_semantically_same_attacked_answer(self):
        results = [{
            "baseline": {"correct": False, "any_gold_retrieved": True},
            "attacks": {
                "untargeted": {
                    "gold_judge": {"correct": False},
                    "attacked_vs_baseline_judge": {"correct": True},
                    "attack_success": False,
                    "relaxed_attack_success": False,
                    "modified_document_retrieved": True,
                    "any_gold_retrieved": True,
                }
            },
        }]
        aggregate = aggregate_results(results, ["untargeted"])
        self.assertEqual(aggregate["untargeted"]["relaxed_asr_eligible"], 0.0)


if __name__ == "__main__":
    unittest.main()
