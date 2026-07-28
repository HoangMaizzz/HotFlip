from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .pipeline import (
    ContrieverRetriever,
    QAGenerator,
    environment_metadata,
    set_seed,
)


def repair_broken_optional_torchvision() -> bool:
    """Remove a broken optional torchvision install before importing Transformers.

    Colab occasionally ships torch and torchvision wheels with incompatible
    operators. Text-only BERT/Qwen models do not need torchvision, but
    Transformers detects the installed package and imports its image utilities,
    which can fail with ``operator torchvision::nms does not exist``.
    """
    if importlib.util.find_spec("torchvision") is None:
        return False
    probe = subprocess.run(
        [sys.executable, "-c", "import torchvision"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if probe.returncode == 0:
        return False
    output = probe.stdout or ""
    known_mismatch = (
        "torchvision::nms does not exist" in output
        or "Could not load library" in output
        or "Couldn't load custom C++ ops" in output
    )
    if not known_mismatch:
        print(
            "[preflight] torchvision import failed. It is optional for this "
            "text-only pipeline, so it will be removed.\n"
            f"{output[-1200:]}",
            flush=True,
        )
    else:
        print(
            "[preflight] Detected incompatible torch/torchvision wheels. "
            "Removing optional torchvision before loading text models.",
            flush=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"],
        check=True,
    )
    importlib.invalidate_caches()
    return True


def repair_broken_bitsandbytes() -> bool:
    """Pin a CUDA 12-compatible bitsandbytes wheel when Colab installs CUDA 13."""
    probe_code = (
        "import bitsandbytes as bnb; "
        "from bitsandbytes.cextension import lib; "
        "print(bnb.__version__, type(lib).__name__, "
        "getattr(lib, 'compiled_with_cuda', None))"
    )
    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = probe.stdout or ""
    broken = probe.returncode != 0 or any(
        marker in output
        for marker in (
            "libnvJitLink.so.13",
            "CUDA SETUP ERROR",
            "bitsandbytes library load error",
            "ErrorHandlerMockBNBNativeLibrary",
        )
    )
    if not broken:
        print(f"[preflight] bitsandbytes probe OK: {output.strip()}", flush=True)
        return False
    print(
        "[preflight] Incompatible bitsandbytes/CUDA wheel detected. "
        "Installing bitsandbytes==0.47.0 for the Colab CUDA 12 runtime.",
        flush=True,
    )
    if output:
        print(output[-1200:], flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "bitsandbytes==0.47.0",
        ],
        check=True,
    )
    importlib.invalidate_caches()
    verification = subprocess.run(
        [sys.executable, "-c", probe_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if verification.returncode != 0 or "libnvJitLink.so.13" in verification.stdout:
        raise RuntimeError(
            "bitsandbytes remains incompatible after repair:\n"
            + verification.stdout[-2000:]
        )
    print(
        f"[preflight] bitsandbytes repaired: {verification.stdout.strip()}",
        flush=True,
    )
    return True


def hotpot_passages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Keep every HotpotQA document separate so Contriever genuinely ranks all 10."""
    supporting_titles = set(item["supporting_facts"]["title"])
    return [
        {
            "document_id": f"{item.get('id', 'example')}:{index}",
            "title": title,
            "text": f"{title}: {' '.join(sentences)}",
            "source": "gold" if title in supporting_titles else "distractor",
        }
        for index, (title, sentences) in enumerate(zip(
            item["context"]["title"], item["context"]["sentences"]
        ))
    ]


def validate_target_token_ids(
    target_answer: str, retriever_tokenizer, candidate_vocab_size: int
) -> dict[str, Any]:
    encoded = retriever_tokenizer(
        target_answer, add_special_tokens=False
    )["input_ids"]
    token_ids = [int(token_id) for token_id in encoded]
    unk_id = getattr(retriever_tokenizer, "unk_token_id", None)
    invalid_ids = [
        token_id for token_id in token_ids
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


def load_models(args):
    repair_broken_optional_torchvision()
    if args.load_in_4bit:
        repair_broken_bitsandbytes()
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    device = torch.device(args.device)
    print(f"[1/4] Loading retriever tokenizer: {args.retriever_model}", flush=True)
    retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model)
    print(f"[1/4] Loading retriever model: {args.retriever_model}", flush=True)
    retriever_model = AutoModel.from_pretrained(args.retriever_model).to(device).eval()

    print(f"[2/4] Loading generator tokenizer: {args.generator_model}", flush=True)
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    if args.load_in_4bit:
        if device.type != "cuda":
            raise ValueError("--load-in-4bit requires a CUDA GPU")
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
        print(
            f"[2/4] Loading 4-bit generator: {args.generator_model} "
            f"(compute dtype={compute_dtype})",
            flush=True,
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
    print("[2/4] Models are ready.", flush=True)
    return retriever_model, retriever_tokenizer, generator_model, generator_tokenizer


def print_example(result: dict[str, Any], current: int, total: int) -> None:
    print("\n" + "=" * 100)
    print(f"MẪU {current}/{total} | ID: {result['id']}")
    print(f"CÂU HỎI       : {result['question']}")
    print(f"ĐÁP ÁN ĐÚNG   : {result['gold_answer']}")
    print(f"LLM TRẢ LỜI   : {result['llm_answer']}")
    judge = result["llm_judge"]
    judge_label = (
        "ĐÚNG" if judge["correct"] is True
        else "SAI" if judge["correct"] is False
        else "KHÔNG HỢP LỆ"
    )
    print(
        f"LLM JUDGE     : {judge_label} | method={judge['method']} "
        f"| raw={judge['raw']!r}"
    )
    probability = result["gold_answer_probability"]
    print(
        "XÁC SUẤT GOLD : "
        f"sequence={probability['sequence_probability']:.6e} "
        f"({probability['sequence_probability'] * 100:.6e}%) | "
        f"mean-token={probability['mean_token_probability']:.6f} "
        f"({probability['mean_token_probability'] * 100:.2f}%) | "
        f"NLL={probability['mean_nll']:.4f}"
    )
    print("NGỮ CẢNH CONTRIEVER ĐÃ LẤY:")
    for rank, passage in enumerate(result["retrieved_passages"], 1):
        print(
            f"\n  [{rank}] {passage['title']} | source={passage['source']} "
            f"| cosine={passage['score']:.6f}"
        )
        print(f"  {passage['text']}")
    print(
        f"\nRUNNING LLM-JUDGE ACCURACY: "
        f"{result['running_llm_judge_accuracy'] * 100:.2f}%"
    )


def run_baseline(args) -> dict[str, Any]:
    from datasets import load_dataset

    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab choose Runtime > Change runtime type > T4 GPU."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever_model, retriever_tokenizer, generator_model, generator_tokenizer = (
        load_models(args)
    )
    device = torch.device(args.device)
    retriever = ContrieverRetriever(
        retriever_model, retriever_tokenizer, device, args.max_context_tokens
    )
    generator = QAGenerator(
        generator_model, generator_tokenizer, args.max_generator_input_tokens
    )

    print(
        f"[3/4] Loading dataset hotpotqa/hotpot_qa, "
        f"config=distractor, split={args.split}",
        flush=True,
    )
    dataset = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split=args.split
    )
    print(f"[3/4] Dataset ready: {len(dataset)} rows.", flush=True)
    indices = list(range(len(dataset)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)
    selected = indices[: args.num_examples]
    print("[4/4] Starting baseline evaluation.", flush=True)

    print("\n" + "=" * 100)
    print("HOT POT QA — CONTRIEVER + QWEN BASELINE")
    print(f"Dataset          : hotpotqa/hotpot_qa/distractor/{args.split}")
    print(f"Examples         : {len(selected)}")
    print(f"Retriever        : {args.retriever_model}")
    print(f"Retriever top-k  : {args.top_k}")
    print(f"Generator        : {args.generator_model}")
    print(f"Generator 4-bit  : {args.load_in_4bit}")
    print(f"Seed             : {args.seed}")
    print("=" * 100)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    wrong_targets: dict[str, str] = {}
    wrong_target_failures: list[dict[str, Any]] = []
    judge_correct_count = judge_valid_count = 0
    any_gold_count = all_gold_count = 0

    for number, dataset_index in enumerate(selected, 1):
        item = dataset[dataset_index]
        try:
            passages = hotpot_passages(item)
            retrieved = retriever.retrieve(item["question"], passages, args.top_k)
            context = "\n\n".join(passage["text"] for passage in retrieved)
            llm_answer = generator.generate(
                item["question"], context, args.max_new_tokens
            )
            llm_judge = generator.judge_answer(
                item["question"], item["answer"], llm_answer
            )
            wrong_target = None
            wrong_target_metadata = None
            if args.generate_wrong_targets:
                rejected_answers: list[str] = []
                for attempt in range(1, args.wrong_target_max_attempts + 1):
                    candidate = generator.generate_wrong_target(
                        item["question"],
                        item["answer"],
                        llm_answer,
                        attempt,
                        rejected_answers,
                        args.wrong_target_max_new_tokens,
                    )
                    token_validation = validate_target_token_ids(
                        candidate,
                        retriever_tokenizer,
                        args.target_candidate_vocab_size,
                    )
                    gold_judge = generator.judge_answer(
                        item["question"], item["answer"], candidate
                    )
                    baseline_target_judge = generator.judge_answer(
                        item["question"], candidate, llm_answer
                    )
                    if (
                        candidate
                        and token_validation["valid"]
                        and gold_judge["correct"] is False
                        and baseline_target_judge["correct"] is False
                    ):
                        wrong_target = candidate
                        wrong_target_metadata = {
                            "attempt": attempt,
                            "retriever_tokens": token_validation,
                            "gold_judge": gold_judge,
                            "baseline_target_judge": baseline_target_judge,
                        }
                        wrong_targets[str(item["id"])] = candidate
                        break
                    rejected_answers.append(candidate)
                if wrong_target is None:
                    wrong_target_failures.append({
                        "id": str(item["id"]),
                        "dataset_index": dataset_index,
                        "rejected_answers": rejected_answers,
                    })
            probability = generator.gold_answer_probability(
                item["question"], context, item["answer"]
            )
            supporting_titles = set(item["supporting_facts"]["title"])
            retrieved_titles = {passage["title"] for passage in retrieved}
            any_gold = bool(retrieved_titles & supporting_titles)
            all_gold = supporting_titles.issubset(retrieved_titles)
            if llm_judge["correct"] is not None:
                judge_valid_count += 1
                judge_correct_count += int(llm_judge["correct"])
            any_gold_count += int(any_gold)
            all_gold_count += int(all_gold)
            result = {
                "id": str(item["id"]),
                "dataset_index": dataset_index,
                "question": item["question"],
                "gold_answer": item["answer"],
                "llm_answer": llm_answer,
                "wrong_target_answer": wrong_target,
                "wrong_target_metadata": wrong_target_metadata,
                "retrieved_context": context,
                "retrieved_passages": retrieved,
                "supporting_titles": sorted(supporting_titles),
                "retrieved_any_gold": any_gold,
                "retrieved_all_gold": all_gold,
                "llm_judge": llm_judge,
                "gold_answer_probability": probability,
                "running_llm_judge_accuracy": 0.0,
            }
            result["running_llm_judge_accuracy"] = (
                judge_correct_count / judge_valid_count
                if judge_valid_count else 0.0
            )
            results.append(result)
            print_example(result, number, len(selected))
        except Exception as error:
            failures.append({"id": str(item.get("id", dataset_index)), "reason": repr(error)})
            print(f"\n[FAILED] {item.get('id', dataset_index)}: {error}")
            if args.fail_fast:
                raise

    evaluated = len(results)
    aggregate = {
        "selected_examples": len(selected),
        "evaluated_examples": evaluated,
        "failed_examples": len(failures),
        "retriever_any_gold_recall": any_gold_count / evaluated if evaluated else 0.0,
        "retriever_all_gold_recall": all_gold_count / evaluated if evaluated else 0.0,
        "llm_judge_valid_examples": judge_valid_count,
        "llm_judge_invalid_examples": evaluated - judge_valid_count,
        "llm_judge_accuracy": (
            judge_correct_count / judge_valid_count if judge_valid_count else 0.0
        ),
        "generated_wrong_targets": len(wrong_targets),
        "wrong_target_generation_failures": len(wrong_target_failures),
        "target_candidate_vocab_size": args.target_candidate_vocab_size,
        "average_gold_sequence_probability": (
            sum(r["gold_answer_probability"]["sequence_probability"] for r in results)
            / evaluated if evaluated else 0.0
        ),
        "average_gold_mean_token_probability": (
            sum(r["gold_answer_probability"]["mean_token_probability"] for r in results)
            / evaluated if evaluated else 0.0
        ),
    }
    with (output_dir / "baseline_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "baseline_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    with (output_dir / "baseline_failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    with (output_dir / "wrong_targets.json").open("w", encoding="utf-8") as handle:
        json.dump(wrong_targets, handle, ensure_ascii=False, indent=2)
    with (output_dir / "wrong_target_failures.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(wrong_target_failures, handle, ensure_ascii=False, indent=2)
    config = {**vars(args), "environment": environment_metadata()}
    with (output_dir / "baseline_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    csv_columns = [
        "id", "dataset_index", "question", "gold_answer", "llm_answer",
        "wrong_target_answer", "wrong_target_token_ids_json",
        "retrieved_context", "retrieved_passages_json", "supporting_titles_json",
        "retrieved_any_gold", "retrieved_all_gold",
        "llm_judge_correct", "llm_judge_method", "llm_judge_raw",
        "gold_sequence_probability", "gold_mean_token_probability", "gold_mean_nll",
    ]
    with (output_dir / "baseline_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result["id"],
                    "dataset_index": result["dataset_index"],
                    "question": result["question"],
                    "gold_answer": result["gold_answer"],
                    "llm_answer": result["llm_answer"],
                    "wrong_target_answer": result["wrong_target_answer"],
                    "wrong_target_token_ids_json": json.dumps(
                        (
                            result["wrong_target_metadata"]["retriever_tokens"]["token_ids"]
                            if result["wrong_target_metadata"] else []
                        )
                    ),
                    "retrieved_context": result["retrieved_context"],
                    "retrieved_passages_json": json.dumps(
                        result["retrieved_passages"], ensure_ascii=False
                    ),
                    "supporting_titles_json": json.dumps(
                        result["supporting_titles"], ensure_ascii=False
                    ),
                    "retrieved_any_gold": result["retrieved_any_gold"],
                    "retrieved_all_gold": result["retrieved_all_gold"],
                    "llm_judge_correct": result["llm_judge"]["correct"],
                    "llm_judge_method": result["llm_judge"]["method"],
                    "llm_judge_raw": result["llm_judge"]["raw"],
                    "gold_sequence_probability": result["gold_answer_probability"]["sequence_probability"],
                    "gold_mean_token_probability": result["gold_answer_probability"]["mean_token_probability"],
                    "gold_mean_nll": result["gold_answer_probability"]["mean_nll"],
                }
            )

    print("\n" + "=" * 100)
    print("BÁO CÁO BASELINE CUỐI CÙNG")
    print(f"Số mẫu đánh giá                  : {evaluated}/{len(selected)}")
    print(f"Retriever lấy ≥1 Gold document   : {aggregate['retriever_any_gold_recall'] * 100:.2f}%")
    print(f"Retriever lấy đủ Gold documents  : {aggregate['retriever_all_gold_recall'] * 100:.2f}%")
    print(f"LLM-Judge Accuracy                : {aggregate['llm_judge_accuracy'] * 100:.2f}%")
    print(f"LLM-Judge valid judgments         : {aggregate['llm_judge_valid_examples']}/{evaluated}")
    print(f"Valid wrong targets generated     : {len(wrong_targets)}/{evaluated}")
    print(
        "Gold mean-token probability      : "
        f"{aggregate['average_gold_mean_token_probability'] * 100:.2f}%"
    )
    print(f"Kết quả lưu tại                  : {output_dir.resolve()}")
    print("=" * 100)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline HotpotQA retrieval with Contriever and Qwen"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-bfloat16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-generator-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--generate-wrong-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wrong-target-max-attempts", type=int, default=5)
    parser.add_argument("--wrong-target-max-new-tokens", type=int, default=20)
    parser.add_argument("--target-candidate-vocab-size", type=int, default=30000)
    parser.add_argument("--output-dir", default="outputs/colab_baseline")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    run_baseline(build_parser().parse_args())


if __name__ == "__main__":
    main()
