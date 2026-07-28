from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from .baseline import repair_broken_bitsandbytes, repair_broken_optional_torchvision
from .pipeline import QAGenerator, set_seed


def validate_target_token_ids(
    target_answer: str, retriever_tokenizer, candidate_vocab_size: int
) -> dict[str, Any]:
    encoded = retriever_tokenizer(
        target_answer, add_special_tokens=False
    )["input_ids"]
    token_ids = [int(token_id) for token_id in encoded]
    unk_id = getattr(retriever_tokenizer, "unk_token_id", None)
    invalid_ids = [
        token_id
        for token_id in token_ids
        if token_id >= candidate_vocab_size
        or (unk_id is not None and token_id == int(unk_id))
    ]
    return {
        "valid": bool(token_ids) and not invalid_ids,
        "token_ids": token_ids,
        "tokens": retriever_tokenizer.convert_ids_to_tokens(token_ids),
        "invalid_token_ids": invalid_ids,
        "candidate_vocab_size": candidate_vocab_size,
    }


def load_rows(path: str, limit: int | None) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "question", "gold_answer", "llm_answer"}
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Baseline CSV is missing columns: {sorted(missing)}")
    return rows[:limit] if limit is not None else rows


def load_target_generator(args: argparse.Namespace):
    repair_broken_optional_torchvision()
    if args.load_in_4bit:
        repair_broken_bitsandbytes()
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    device = torch.device(args.device)
    retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model)
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    if args.load_in_4bit:
        if device.type != "cuda":
            raise ValueError("--load-in-4bit requires CUDA")
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported() and args.prefer_bfloat16
            else torch.float16
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        generator_model = AutoModelForCausalLM.from_pretrained(
            args.generator_model,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        generator_model = AutoModelForCausalLM.from_pretrained(
            args.generator_model,
            dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            generator_model.to(device)
    generator_model.eval()
    return retriever_tokenizer, generator_model, generator_tokenizer


def run_target_generation(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Colab GPU runtime")
    rows = load_rows(args.baseline_results, args.num_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever_tokenizer, model, tokenizer = load_target_generator(args)
    generator = QAGenerator(model, tokenizer, args.max_generator_input_tokens)

    targets: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, row in enumerate(rows, 1):
        rejected: list[dict[str, Any]] = []
        accepted: dict[str, Any] | None = None
        for attempt in range(1, args.max_attempts + 1):
            rejected_answers = [
                item["candidate"] for item in rejected if item["candidate"]
            ]
            candidate = generator.generate_wrong_target(
                row["question"],
                row["gold_answer"],
                row["llm_answer"],
                attempt,
                rejected_answers,
                args.max_new_tokens,
            )
            token_validation = validate_target_token_ids(
                candidate, retriever_tokenizer, args.candidate_vocab_size
            )
            gold_judge = generator.judge_answer(
                row["question"], row["gold_answer"], candidate
            )
            baseline_target_judge = generator.judge_answer(
                row["question"], candidate, row["llm_answer"]
            )
            reasons: list[str] = []
            if not candidate:
                reasons.append("empty")
            if not token_validation["valid"]:
                reasons.append("outside_candidate_vocabulary")
            if gold_judge["correct"] is not False:
                reasons.append("not_confirmed_wrong_against_gold")
            if baseline_target_judge["correct"] is not False:
                reasons.append("equivalent_to_baseline_answer")
            attempt_result = {
                "attempt": attempt,
                "candidate": candidate,
                "retriever_tokens": token_validation,
                "gold_judge": gold_judge,
                "baseline_target_judge": baseline_target_judge,
                "rejection_reasons": reasons,
            }
            if not reasons:
                accepted = attempt_result
                break
            rejected.append(attempt_result)

        print("\n" + "=" * 100)
        print(f"TARGET {number}/{len(rows)} | ID={row['id']}")
        print(f"QUESTION        : {row['question']}")
        print(f"GOLD ANSWER     : {row['gold_answer']}")
        print(f"BASELINE ANSWER : {row['llm_answer']}")
        if accepted:
            target = accepted["candidate"]
            targets[str(row["id"])] = target
            detail = {
                "id": str(row["id"]),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "baseline_answer": row["llm_answer"],
                "wrong_target_answer": target,
                "retriever_token_ids": accepted["retriever_tokens"]["token_ids"],
                "retriever_tokens": accepted["retriever_tokens"]["tokens"],
                "attempt": accepted["attempt"],
            }
            details.append(detail)
            print(f"WRONG TARGET    : {target}")
            print(f"TOKEN IDs       : {detail['retriever_token_ids']}")
            print(f"TOKENS          : {detail['retriever_tokens']}")
            print("VALIDATION       : wrong_vs_gold=YES | different_from_baseline=YES")
        else:
            failure = {
                "id": str(row["id"]),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "baseline_answer": row["llm_answer"],
                "attempts": rejected,
            }
            failures.append(failure)
            print("WRONG TARGET    : FAILED")
            print(f"REJECTED        : {[item['candidate'] for item in rejected]}")

    aggregate = {
        "requested_examples": len(rows),
        "valid_targets": len(targets),
        "failed_targets": len(failures),
        "candidate_vocab_size": args.candidate_vocab_size,
        "generator_model": args.generator_model,
        "retriever_tokenizer": args.retriever_model,
    }
    with (output_dir / "wrong_targets.json").open("w", encoding="utf-8") as handle:
        json.dump(targets, handle, ensure_ascii=False, indent=2)
    with (output_dir / "wrong_target_details.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    with (output_dir / "wrong_target_failures.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    with (output_dir / "wrong_target_aggregate.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    with (output_dir / "wrong_target_details.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        columns = [
            "id", "question", "gold_answer", "baseline_answer",
            "wrong_target_answer", "retriever_token_ids", "retriever_tokens",
            "attempt",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for detail in details:
            writer.writerow({
                **detail,
                "retriever_token_ids": json.dumps(detail["retriever_token_ids"]),
                "retriever_tokens": json.dumps(
                    detail["retriever_tokens"], ensure_ascii=False
                ),
            })
    print("\n" + "=" * 100)
    print("WRONG TARGET GENERATION SUMMARY")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Saved to: {output_dir.resolve()}")
    if args.require_all and failures:
        raise RuntimeError(
            f"{len(failures)} targets failed validation; inspect "
            f"{output_dir / 'wrong_target_failures.json'}"
        )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate false target answers for HotFlip"
    )
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--num-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-bfloat16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-vocab-size", type=int, default=30000)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-generator-input-tokens", type=int, default=3072)
    parser.add_argument("--require-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/generated_targets")
    return parser


def main() -> None:
    run_target_generation(build_parser().parse_args())


if __name__ == "__main__":
    main()
