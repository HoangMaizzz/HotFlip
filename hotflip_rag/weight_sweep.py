from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .compare_attacks import (
    build_parser as build_comparison_parser,
    load_baseline_rows,
    parse_optional_bool,
    prepare_comparison_runtime,
    run_comparison,
)
from .metrics import normalize_answer
from .pipeline import load_targets, target_for_example


YES_NO = {"yes", "no"}


def _truthy(value: Any) -> bool:
    return parse_optional_bool(value) is True


def select_stratified_rows(
    rows: list[dict[str, Any]],
    target_by_id: dict[str, str],
    num_examples: int,
    seed: int,
    no_gold_fraction: float = 0.2,
) -> list[dict[str, Any]]:
    """Select a fixed, balanced tuning set used by every weight."""

    if num_examples < 2:
        raise ValueError("num_examples must be at least 2")
    if not 0.0 <= no_gold_fraction <= 1.0:
        raise ValueError("no_gold_fraction must be between 0 and 1")

    eligible: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row.get("id", ""))
        target = str(target_by_id.get(example_id, "")).strip()
        baseline_correct = parse_optional_bool(row.get("llm_judge_correct"))
        if baseline_correct is None or not target:
            continue
        if normalize_answer(str(row.get("gold_answer", ""))) in YES_NO:
            continue
        if normalize_answer(str(row.get("llm_answer", ""))) == normalize_answer(target):
            continue
        eligible.append(row)

    rng = random.Random(seed)
    correct_quota = num_examples // 2
    incorrect_quota = num_examples - correct_quota

    def choose(correct: bool, quota: int) -> list[dict[str, Any]]:
        group = [
            row for row in eligible
            if parse_optional_bool(row.get("llm_judge_correct")) is correct
        ]
        with_gold = [row for row in group if _truthy(row.get("retrieved_any_gold"))]
        without_gold = [row for row in group if not _truthy(row.get("retrieved_any_gold"))]
        rng.shuffle(with_gold)
        rng.shuffle(without_gold)

        desired_without_gold = round(quota * no_gold_fraction)
        chosen = without_gold[:desired_without_gold]
        chosen.extend(with_gold[: quota - len(chosen)])
        if len(chosen) < quota:
            used = {str(row["id"]) for row in chosen}
            leftovers = [row for row in group if str(row["id"]) not in used]
            rng.shuffle(leftovers)
            chosen.extend(leftovers[: quota - len(chosen)])
        if len(chosen) < quota:
            label = "correct" if correct else "incorrect"
            raise ValueError(
                f"Only {len(chosen)} eligible baseline-{label} rows; need {quota}."
            )
        return chosen

    selected = choose(True, correct_quota) + choose(False, incorrect_quota)
    rng.shuffle(selected)
    return selected


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_or_none(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return mean(valid) if valid else None


def hotflip_means(result_path: Path, mode: str) -> dict[str, float | None]:
    hotflip: list[dict[str, Any]] = []
    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                hotflip.append(record["attacks"][mode]["hotflip"])

    return {
        "mean_token_changes": _mean_or_none([len(item["changes"]) for item in hotflip]),
        "mean_objective_before": _mean_or_none(
            [item.get("objective_before") for item in hotflip]
        ),
        "mean_objective_after": _mean_or_none(
            [item.get("objective_after") for item in hotflip]
        ),
        "mean_query_cosine_before": _mean_or_none(
            [item.get("query_similarity_before") for item in hotflip]
        ),
        "mean_query_cosine_after": _mean_or_none(
            [item.get("query_similarity_after") for item in hotflip]
        ),
        "mean_answer_cosine_before": _mean_or_none(
            [item.get("target_similarity_before") for item in hotflip]
        ),
        "mean_answer_cosine_after": _mean_or_none(
            [item.get("target_similarity_after") for item in hotflip]
        ),
    }


def config_label(mode: str, weight: float) -> str:
    token = format(weight, "g").replace("-", "m").replace(".", "p")
    parameter = "beta" if mode == "untargeted" else "lambda"
    return f"{mode}_{parameter}_{token}"


def comparison_args(
    sweep_args: argparse.Namespace,
    subset_path: Path,
    mode: str,
    weight: float,
    output_dir: Path,
) -> argparse.Namespace:
    args = build_comparison_parser().parse_args([
        "--baseline-results", str(subset_path),
        "--target-answer-file", str(sweep_args.target_answer_file),
    ])
    args.split = sweep_args.split
    args.num_examples = None
    args.modes = [mode]
    args.seed = sweep_args.seed
    args.retriever_model = sweep_args.retriever_model
    args.generator_model = sweep_args.generator_model
    args.device = sweep_args.device
    args.load_in_4bit = sweep_args.load_in_4bit
    args.prefer_bfloat16 = sweep_args.prefer_bfloat16
    args.top_k = sweep_args.top_k
    args.max_context_tokens = sweep_args.max_context_tokens
    args.max_generator_input_tokens = sweep_args.max_generator_input_tokens
    args.max_new_tokens = sweep_args.max_new_tokens
    args.target_weight = weight if mode == "targeted" else 3.0
    args.untargeted_answer_weight = weight if mode == "untargeted" else 1.0
    args.search_strategy = "beam"
    args.max_token_changes = sweep_args.max_token_changes
    args.beam_width = sweep_args.beam_width
    args.hotflip_top_k = sweep_args.hotflip_top_k
    args.candidates_per_state = sweep_args.candidates_per_state
    args.candidate_policy = sweep_args.candidate_policy
    args.candidate_vocab_size = sweep_args.candidate_vocab_size
    args.score_chunk_size = sweep_args.score_chunk_size
    args.min_objective_improvement = sweep_args.min_objective_improvement
    args.exact_rerank = True
    args.preserve_token_class = True
    args.preserve_leading_space = True
    args.disallow_punctuation_replacement = True
    args.disallow_numeric_replacement = False
    args.output_dir = str(output_dir)
    args.fail_fast = sweep_args.fail_fast
    args.trace_hotflip = False
    return args


def summary_row(
    mode: str,
    weight: float,
    aggregate: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    metrics = aggregate[mode]
    if mode == "untargeted":
        strict_asr = metrics["strict_asr_on_baseline_correct"]
        strict_successes = metrics["asr_on_baseline_correct_successes"]
        strict_examples = metrics["asr_on_baseline_correct_examples"]
        relaxed_asr = metrics["relaxed_asr_eligible"]
        relaxed_successes = metrics["relaxed_asr_eligible_successes"]
        relaxed_examples = metrics["relaxed_asr_eligible_examples"]
        parameter = "beta"
    else:
        strict_asr = metrics["strict_targeted_asr_eligible"]
        strict_successes = metrics["strict_targeted_successes"]
        strict_examples = metrics["targeted_eligible_examples"]
        relaxed_asr = metrics["relaxed_targeted_asr_eligible"]
        relaxed_successes = metrics["relaxed_targeted_successes"]
        relaxed_examples = metrics["targeted_eligible_examples"]
        parameter = "lambda"

    row = {
        "mode": mode,
        "parameter": parameter,
        "weight": weight,
        "examples": aggregate["examples"],
        "baseline_accuracy": aggregate["baseline_accuracy"],
        "accuracy_after_attack": metrics["accuracy_after_attack"],
        "accuracy_drop": metrics["accuracy_drop"],
        "strict_asr": strict_asr,
        "strict_successes": strict_successes,
        "strict_eligible_examples": strict_examples,
        "relaxed_asr": relaxed_asr,
        "relaxed_successes": relaxed_successes,
        "relaxed_eligible_examples": relaxed_examples,
        "modified_document_retrieval_rate": metrics[
            "modified_document_retrieval_rate"
        ],
        "attacked_any_gold_retrieval_rate": metrics[
            "attacked_any_gold_retrieval_rate"
        ],
    }
    row.update(hotflip_means(result_path, mode))
    return row


def run_weight_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_baseline_rows(args.baseline_results, None)
    targets = load_targets(args.target_answer_file, None)

    # The shared runtime avoids loading Contriever, Qwen, and HotpotQA six times.
    bootstrap_args = comparison_args(
        args,
        Path(args.baseline_results),
        "untargeted",
        args.untargeted_weights[0],
        output_dir / "_bootstrap",
    )
    runtime = prepare_comparison_runtime(bootstrap_args)
    target_by_id: dict[str, str] = {}
    for row in baseline_rows:
        example_id = str(row["id"])
        dataset_index = runtime.id_to_index.get(example_id)
        if dataset_index is None:
            continue
        item = runtime.dataset[dataset_index]
        target = target_for_example(targets, item, dataset_index)
        if target:
            target_by_id[example_id] = str(target)

    selected = select_stratified_rows(
        baseline_rows,
        target_by_id,
        args.num_examples,
        args.seed,
        args.no_gold_fraction,
    )
    subset_path = output_dir / "selected_tuning_baseline.csv"
    write_csv_rows(subset_path, selected)

    manifest = {
        "selected_examples": len(selected),
        "baseline_correct": sum(
            parse_optional_bool(row.get("llm_judge_correct")) is True
            for row in selected
        ),
        "baseline_incorrect": sum(
            parse_optional_bool(row.get("llm_judge_correct")) is False
            for row in selected
        ),
        "baseline_any_gold": sum(
            _truthy(row.get("retrieved_any_gold")) for row in selected
        ),
        "baseline_no_gold": sum(
            not _truthy(row.get("retrieved_any_gold")) for row in selected
        ),
        "ids": [str(row["id"]) for row in selected],
        "untargeted_weights": args.untargeted_weights,
        "targeted_weights": args.targeted_weights,
    }
    with (output_dir / "selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\nWEIGHT SWEEP SAMPLE")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    summaries: list[dict[str, Any]] = []
    configurations = [
        ("untargeted", weight) for weight in args.untargeted_weights
    ] + [("targeted", weight) for weight in args.targeted_weights]

    for number, (mode, weight) in enumerate(configurations, 1):
        label = config_label(mode, weight)
        config_dir = output_dir / label
        aggregate_path = config_dir / "comparison_aggregate.json"
        result_path = config_dir / "comparison_results.jsonl"
        print("\n" + "#" * 110)
        print(
            f"SWEEP CONFIG {number}/{len(configurations)}: "
            f"mode={mode} weight={weight:g}",
            flush=True,
        )
        if args.resume and aggregate_path.exists() and result_path.exists():
            print(f"[resume] Reusing completed output: {config_dir}", flush=True)
            with aggregate_path.open(encoding="utf-8") as handle:
                aggregate = json.load(handle)
        else:
            run_args = comparison_args(
                args, subset_path, mode, weight, config_dir
            )
            aggregate = run_comparison(run_args, runtime=runtime)
        summaries.append(summary_row(mode, weight, aggregate, result_path))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_csv = output_dir / "weight_sweep_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "weight_sweep_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    recommendations: dict[str, Any] = {}
    for mode in ("untargeted", "targeted"):
        candidates = [row for row in summaries if row["mode"] == mode]
        best = max(
            candidates,
            key=lambda row: (
                row["strict_asr"],
                row["modified_document_retrieval_rate"],
                row["relaxed_asr"],
            ),
        )
        recommendations[mode] = {
            "parameter": best["parameter"],
            "recommended_weight": best["weight"],
            "selection_rule": (
                "highest strict ASR; ties broken by modified-document "
                "retrieval rate, then relaxed ASR"
            ),
            "pilot_metrics": best,
        }
    with (output_dir / "recommended_weights.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(recommendations, handle, ensure_ascii=False, indent=2)

    print("\nWEIGHT SWEEP SUMMARY")
    for row in summaries:
        print(
            f"{row['mode']:10} {row['parameter']}={row['weight']:g} "
            f"| strict_ASR={row['strict_asr'] * 100:.2f}% "
            f"| relaxed_ASR={row['relaxed_asr'] * 100:.2f}% "
            f"| attacked_accuracy={row['accuracy_after_attack'] * 100:.2f}% "
            f"| modified_doc_retrieved="
            f"{row['modified_document_retrieval_rate'] * 100:.2f}%"
        )
    print("\nRECOMMENDED PILOT WEIGHTS")
    print(json.dumps(recommendations, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {output_dir.resolve()}")
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune untargeted beta and targeted lambda on one fixed sample"
    )
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--target-answer-file", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gold-fraction", type=float, default=0.2)
    parser.add_argument(
        "--untargeted-weights", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--targeted-weights", type=float, nargs="+", default=[1.5, 3.0, 6.0]
    )
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prefer-bfloat16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-generator-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-token-changes", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--hotflip-top-k", type=int, default=10)
    parser.add_argument("--candidates-per-state", type=int, default=10)
    parser.add_argument(
        "--candidate-policy",
        choices=["full_vocab", "tokenizer_safe"],
        default="tokenizer_safe",
    )
    parser.add_argument("--candidate-vocab-size", type=int, default=30000)
    parser.add_argument("--score-chunk-size", type=int, default=2048)
    parser.add_argument("--min-objective-improvement", type=float, default=0.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", default="outputs/weight_sweep")
    return parser


def main() -> None:
    run_weight_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()
