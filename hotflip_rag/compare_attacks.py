from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from .baseline import hotpot_passages, load_models
from .hotflip import ContrieverHotFlipAttacker, HotFlipConfig
from .pipeline import (
    ContrieverRetriever,
    QAGenerator,
    context_diff,
    load_targets,
    select_gold_document,
    set_seed,
    target_for_example,
)


def parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def load_baseline_rows(path: str, limit: int | None) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No baseline rows found in {path}")
    required = {"id", "question", "gold_answer", "llm_answer", "llm_judge_correct"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Baseline CSV is missing columns: {sorted(missing)}")
    return rows


def reconstruct_baseline_documents(
    row: dict[str, Any], passages: list[dict[str, str]]
) -> list[dict[str, Any]]:
    serialized = row.get("retrieved_passages_json", "").strip()
    if serialized:
        try:
            value = json.loads(serialized)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass

    context = row.get("retrieved_context", "")
    found: list[tuple[int, dict[str, Any]]] = []
    for passage in passages:
        position = context.find(passage["text"])
        if position >= 0:
            found.append((position, passage))
    if found:
        return [passage for _, passage in sorted(found, key=lambda pair: pair[0])]
    return [{
        "document_id": "unknown",
        "title": "(retrieved_context from baseline CSV)",
        "text": context,
        "source": "unknown",
    }]


def make_attacker(
    mode: str,
    args: argparse.Namespace,
    retriever_model,
    retriever_tokenizer,
    device: torch.device,
) -> ContrieverHotFlipAttacker:
    config = HotFlipConfig(
        attack_mode=mode,
        search_strategy=args.search_strategy,
        max_token_changes=args.max_token_changes,
        beam_width=args.beam_width,
        hotflip_top_k=args.hotflip_top_k,
        candidates_per_state=args.candidates_per_state,
        candidate_policy=args.candidate_policy,
        candidate_vocab_size=args.candidate_vocab_size,
        exact_rerank=args.exact_rerank,
        min_objective_improvement=args.min_objective_improvement,
        preserve_token_class=args.preserve_token_class,
        preserve_leading_space=args.preserve_leading_space,
        disallow_punctuation_replacement=args.disallow_punctuation_replacement,
        disallow_numeric_replacement=args.disallow_numeric_replacement,
        target_weight=args.target_weight,
        score_chunk_size=args.score_chunk_size,
        max_context_tokens=args.max_context_tokens,
    )
    return ContrieverHotFlipAttacker(
        retriever_model, retriever_tokenizer, config, device
    )


def print_documents(label: str, passages: list[dict[str, Any]], attacked_id: str) -> None:
    print(f"\n{label}:")
    for rank, passage in enumerate(passages, 1):
        is_modified = passage.get("document_id") == attacked_id
        marker = "YES" if is_modified else "NO"
        score = passage.get("score")
        score_text = f" | cosine={score:.6f}" if isinstance(score, (int, float)) else ""
        print(
            f"  [{rank}] {passage.get('title')} | source={passage.get('source')} "
            f"| contains_modified_document={marker}{score_text}"
        )
        print(f"      {passage.get('text', '')}")


def safe_accuracy(values: list[bool | None]) -> tuple[float, int]:
    valid = [value for value in values if value is not None]
    return (
        sum(bool(value) for value in valid) / len(valid) if valid else 0.0,
        len(valid),
    )


def aggregate_results(results: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    baseline_values = [result["baseline"]["correct"] for result in results]
    baseline_accuracy, baseline_valid = safe_accuracy(baseline_values)
    aggregate: dict[str, Any] = {
        "examples": len(results),
        "baseline_accuracy": baseline_accuracy,
        "baseline_valid_judgments": baseline_valid,
    }
    for mode in modes:
        attacked_values = [
            result["attacks"][mode]["gold_judge"]["correct"] for result in results
        ]
        attacked_accuracy, attacked_valid = safe_accuracy(attacked_values)
        if mode == "untargeted":
            eligible = [
                result for result in results
                if result["baseline"]["correct"] is True
                and result["attacks"][mode]["gold_judge"]["correct"] is not None
            ]
            successes = sum(
                result["attacks"][mode]["gold_judge"]["correct"] is False
                for result in eligible
            )
        else:
            eligible = [
                result for result in results
                if result["attacks"][mode]["baseline_target_judge"]["correct"] is False
                and result["attacks"][mode]["target_judge"]["correct"] is not None
            ]
            successes = sum(
                result["attacks"][mode]["target_judge"]["correct"] is True
                for result in eligible
            )
        aggregate[mode] = {
            "accuracy_after_attack": attacked_accuracy,
            "accuracy_valid_judgments": attacked_valid,
            "accuracy_drop": baseline_accuracy - attacked_accuracy,
            "asr": successes / len(eligible) if eligible else 0.0,
            "asr_successes": successes,
            "asr_eligible_examples": len(eligible),
            "modified_document_retrieval_rate": (
                sum(result["attacks"][mode]["modified_document_retrieved"] for result in results)
                / len(results)
                if results else 0.0
            ),
        }
    return aggregate


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    set_seed(args.seed)
    rows = load_baseline_rows(args.baseline_results, args.num_examples)
    if "targeted" in args.modes and not (
        args.target_answer or args.target_answer_file
    ):
        raise ValueError(
            "Targeted comparison requires --target-answer or --target-answer-file"
        )

    device = torch.device(args.device)
    retriever_model, retriever_tokenizer, generator_model, generator_tokenizer = (
        load_models(args)
    )
    retriever = ContrieverRetriever(
        retriever_model, retriever_tokenizer, device, args.max_context_tokens
    )
    generator = QAGenerator(
        generator_model, generator_tokenizer, args.max_generator_input_tokens
    )
    attackers = {
        mode: make_attacker(
            mode, args, retriever_model, retriever_tokenizer, device
        )
        for mode in args.modes
    }

    print(
        f"[dataset] Loading hotpotqa/hotpot_qa, config=distractor, split={args.split}",
        flush=True,
    )
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    id_to_index = {str(item_id): index for index, item_id in enumerate(dataset["id"])}
    targets = load_targets(args.target_answer_file, args.target_answer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for number, row in enumerate(rows, 1):
        example_id = str(row["id"])
        try:
            if example_id not in id_to_index:
                raise KeyError(f"ID {example_id} is absent from {args.split}")
            dataset_index = id_to_index[example_id]
            item = dataset[dataset_index]
            documents = hotpot_passages(item)
            selected_gold = select_gold_document(
                item["question"], documents, retriever
            )
            baseline_documents = reconstruct_baseline_documents(row, documents)
            baseline_correct = parse_optional_bool(row["llm_judge_correct"])
            target_answer = target_for_example(
                targets, item, dataset_index
            )

            result: dict[str, Any] = {
                "id": example_id,
                "dataset_index": dataset_index,
                "question": item["question"],
                "gold_answer": item["answer"],
                "target_answer": target_answer,
                "selected_document": {
                    key: selected_gold[key]
                    for key in ("document_id", "title", "text", "source", "score")
                },
                "baseline": {
                    "answer": row["llm_answer"],
                    "correct": baseline_correct,
                    "judge_method": row.get("llm_judge_method"),
                    "judge_raw": row.get("llm_judge_raw"),
                    "retrieved_documents": baseline_documents,
                    "modified_document_retrieved": any(
                        passage.get("document_id") == selected_gold["document_id"]
                        for passage in baseline_documents
                    ),
                },
                "attacks": {},
            }

            print("\n" + "=" * 110)
            print(f"EXAMPLE {number}/{len(rows)} | ID={example_id}")
            print(f"QUESTION        : {item['question']}")
            print(f"GOLD ANSWER     : {item['answer']}")
            print(
                f"BASELINE ANSWER : {row['llm_answer']} "
                f"| correct={baseline_correct}"
            )
            print(f"TARGET ANSWER   : {target_answer}")
            print(
                f"DOCUMENT CHOSEN : {selected_gold['title']} "
                f"| id={selected_gold['document_id']} "
                f"| cosine={selected_gold['score']:.6f}"
            )
            print(f"ORIGINAL DOCUMENT:\n{selected_gold['text']}")
            print_documents(
                "BASELINE RETRIEVED DOCUMENTS",
                baseline_documents,
                selected_gold["document_id"],
            )

            for mode in args.modes:
                if mode == "targeted" and not target_answer:
                    raise ValueError(f"Missing target answer for ID {example_id}")
                attack_result = attackers[mode].attack(
                    item["question"],
                    selected_gold["text"],
                    target_answer=target_answer if mode == "targeted" else None,
                )
                attacked_document = {
                    **selected_gold,
                    "text": attack_result.attacked_text,
                    "source": "attacked_gold",
                }
                attacked_pool = [
                    attacked_document
                    if document["document_id"] == selected_gold["document_id"]
                    else document
                    for document in documents
                ]
                retrieved = retriever.retrieve(
                    item["question"], attacked_pool, args.top_k
                )
                context = "\n\n".join(passage["text"] for passage in retrieved)
                answer = generator.generate(
                    item["question"], context, args.max_new_tokens
                )
                gold_judge = generator.judge_answer(
                    item["question"], item["answer"], answer
                )
                target_judge = None
                baseline_target_judge = None
                if mode == "targeted":
                    target_judge = generator.judge_answer(
                        item["question"], target_answer, answer
                    )
                    baseline_target_judge = generator.judge_answer(
                        item["question"], target_answer, row["llm_answer"]
                    )
                    success = (
                        baseline_target_judge["correct"] is False
                        and target_judge["correct"] is True
                    )
                else:
                    success = (
                        baseline_correct is True
                        and gold_judge["correct"] is False
                    )
                modified_retrieved = any(
                    passage["document_id"] == selected_gold["document_id"]
                    for passage in retrieved
                )
                mode_result = {
                    "answer": answer,
                    "gold_judge": gold_judge,
                    "target_judge": target_judge,
                    "baseline_target_judge": baseline_target_judge,
                    "attack_success": success,
                    "modified_document": attack_result.attacked_text,
                    "modified_document_retrieved": modified_retrieved,
                    "retrieved_documents": retrieved,
                    "context_diff": context_diff(
                        selected_gold["text"], attack_result.attacked_text
                    ),
                    "hotflip": attack_result.to_dict(),
                }
                result["attacks"][mode] = mode_result
                print(f"\n--- {mode.upper()} ---")
                print(f"MODIFIED DOCUMENT:\n{attack_result.attacked_text}")
                print(f"DIFF             : {mode_result['context_diff']}")
                print(f"ATTACKED ANSWER  : {answer}")
                print(
                    f"GOLD JUDGE       : correct={gold_judge['correct']} "
                    f"| method={gold_judge['method']} | raw={gold_judge['raw']!r}"
                )
                if target_judge:
                    print(
                        f"TARGET JUDGE     : correct={target_judge['correct']} "
                        f"| method={target_judge['method']} "
                        f"| raw={target_judge['raw']!r}"
                    )
                print(f"ATTACK SUCCESS   : {success}")
                print_documents(
                    f"{mode.upper()} RETRIEVED DOCUMENTS",
                    retrieved,
                    selected_gold["document_id"],
                )
            results.append(result)
        except Exception as error:
            failures.append({"id": example_id, "reason": repr(error)})
            print(f"\n[FAILED] {example_id}: {error}", flush=True)
            if args.fail_fast:
                raise

    aggregate = aggregate_results(results, args.modes)
    with (output_dir / "comparison_results.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "comparison_aggregate.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    with (output_dir / "comparison_failures.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)

    columns = [
        "id", "question", "gold_answer", "target_answer", "baseline_answer",
        "baseline_correct", "attack_mode", "attacked_answer",
        "attacked_correct", "target_correct", "attack_success",
        "attacked_document_title", "attacked_document_id",
        "modified_document_retrieved", "original_document", "modified_document",
        "retrieved_documents_json",
    ]
    with (output_dir / "comparison_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for mode in args.modes:
                attack = result["attacks"][mode]
                writer.writerow({
                    "id": result["id"],
                    "question": result["question"],
                    "gold_answer": result["gold_answer"],
                    "target_answer": result["target_answer"],
                    "baseline_answer": result["baseline"]["answer"],
                    "baseline_correct": result["baseline"]["correct"],
                    "attack_mode": mode,
                    "attacked_answer": attack["answer"],
                    "attacked_correct": attack["gold_judge"]["correct"],
                    "target_correct": (
                        attack["target_judge"]["correct"]
                        if attack["target_judge"] else None
                    ),
                    "attack_success": attack["attack_success"],
                    "attacked_document_title": result["selected_document"]["title"],
                    "attacked_document_id": result["selected_document"]["document_id"],
                    "modified_document_retrieved": attack["modified_document_retrieved"],
                    "original_document": result["selected_document"]["text"],
                    "modified_document": attack["modified_document"],
                    "retrieved_documents_json": json.dumps(
                        attack["retrieved_documents"], ensure_ascii=False
                    ),
                })

    print("\n" + "=" * 110)
    print("FINAL COMPARISON")
    print(
        f"BASELINE ACCURACY : {aggregate['baseline_accuracy'] * 100:.2f}% "
        f"({aggregate['baseline_valid_judgments']} valid judgments)"
    )
    for mode in args.modes:
        metrics = aggregate[mode]
        print(
            f"{mode.upper():10} ACCURACY AFTER={metrics['accuracy_after_attack'] * 100:.2f}% "
            f"| DROP={metrics['accuracy_drop'] * 100:.2f} points "
            f"| ASR={metrics['asr'] * 100:.2f}% "
            f"({metrics['asr_successes']}/{metrics['asr_eligible_examples']}) "
            f"| MODIFIED DOC RETRIEVED="
            f"{metrics['modified_document_retrieval_rate'] * 100:.2f}%"
        )
    print(f"Results saved to: {output_dir.resolve()}")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline_results.csv with one-gold-document HotFlip attacks"
        )
    )
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument(
        "--modes", nargs="+", choices=["untargeted", "targeted"],
        default=["untargeted", "targeted"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-bfloat16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-generator-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--target-answer")
    parser.add_argument("--target-answer-file")
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--search-strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--max-token-changes", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--hotflip-top-k", type=int, default=20)
    parser.add_argument("--candidates-per-state", type=int, default=20)
    parser.add_argument(
        "--candidate-policy",
        choices=["full_vocab", "tokenizer_safe"],
        default="tokenizer_safe",
    )
    parser.add_argument("--candidate-vocab-size", type=int, default=5000)
    parser.add_argument("--score-chunk-size", type=int, default=2048)
    parser.add_argument("--min-objective-improvement", type=float, default=0.0)
    parser.add_argument("--exact-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-token-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-leading-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disallow-punctuation-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disallow-numeric-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", default="outputs/attack_comparison")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    run_comparison(build_parser().parse_args())


if __name__ == "__main__":
    main()
