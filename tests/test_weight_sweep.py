from __future__ import annotations

import unittest

from hotflip_rag.weight_sweep import config_label, select_stratified_rows


class WeightSweepTests(unittest.TestCase):
    def test_selects_balanced_fixed_sample_and_excludes_yes_no(self):
        rows = []
        targets = {}
        for index in range(20):
            example_id = f"id-{index}"
            correct = index < 10
            rows.append({
                "id": example_id,
                "gold_answer": "Paris" if index != 19 else "yes",
                "llm_answer": f"baseline-{index}",
                "llm_judge_correct": str(correct),
                "retrieved_any_gold": str(index % 4 != 0),
            })
            targets[example_id] = f"target-{index}"

        selected = select_stratified_rows(rows, targets, 10, seed=7)
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            sum(row["llm_judge_correct"] == "True" for row in selected), 5
        )
        self.assertEqual(
            sum(row["llm_judge_correct"] == "False" for row in selected), 5
        )
        self.assertNotIn("id-19", {row["id"] for row in selected})

    def test_selection_is_reproducible(self):
        rows = [{
            "id": f"id-{index}",
            "gold_answer": "Paris",
            "llm_answer": f"answer-{index}",
            "llm_judge_correct": str(index < 10),
            "retrieved_any_gold": "True",
        } for index in range(20)]
        targets = {row["id"]: f"target-{row['id']}" for row in rows}
        first = select_stratified_rows(rows, targets, 10, seed=42)
        second = select_stratified_rows(rows, targets, 10, seed=42)
        self.assertEqual(
            [row["id"] for row in first], [row["id"] for row in second]
        )

    def test_config_label_is_path_safe(self):
        self.assertEqual(config_label("untargeted", 0.5), "untargeted_beta_0p5")
        self.assertEqual(config_label("targeted", 3.0), "targeted_lambda_3")


if __name__ == "__main__":
    unittest.main()
