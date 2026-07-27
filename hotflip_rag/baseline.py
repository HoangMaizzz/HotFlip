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

from .metrics import answer_metrics
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


def hotpot_passages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Keep every HotpotQA document separate so Contriever genuinely ranks all 10."""
    supporting_titles = set(item["supporting_facts"]["title"])
    return [
        {
            "title": title,
            "text": f"{title}: {' '.join(sentences)}",
            "source": "gold" if title in supporting_titles else "distractor",
        }
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        )
    ]


def load_models(args):
    repair_broken_optional_torchvision()
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
    print(
        f"ĐÁNH GIÁ      : EM={result['metrics']['em']:.0f} | "
        f"F1={result['metrics']['f1']:.4f}"
    )
    judge = result["llm_judge"]
    judge_label = (
        "ĐÚNG" if judge["correct"] is True
        else "SAI" if judge["correct"] is False
        else "KHÔNG HỢP LỆ"
    )
    print(
        f"HYBRID JUDGE  : {judge_label} | method={judge['method']} "
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
        f"\nRUNNING ACCURACY: EM={result['running_em_accuracy'] * 100:.2f}% | "
        f"F1={result['running_average_f1'] * 100:.2f}%"
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
    print("HOT POT QA — CONTRIEVER + QWEN 7B BASELINE")
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
    em_sum = f1_sum = 0.0
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
            metrics = answer_metrics(llm_answer, item["answer"])
            llm_judge = generator.judge_answer(
                item["question"], item["answer"], llm_answer
            )
            probability = generator.gold_answer_probability(
                item["question"], context, item["answer"]
            )
            supporting_titles = set(item["supporting_facts"]["title"])
            retrieved_titles = {passage["title"] for passage in retrieved}
            any_gold = bool(retrieved_titles & supporting_titles)
            all_gold = supporting_titles.issubset(retrieved_titles)
            em_sum += metrics["em"]
            f1_sum += metrics["f1"]
            if llm_judge["correct"] is not None:
                judge_valid_count += 1
                judge_correct_count += int(llm_judge["correct"])
            any_gold_count += int(any_gold)
            all_gold_count += int(all_gold)
            evaluated_so_far = len(results) + 1
            result = {
                "id": str(item["id"]),
                "dataset_index": dataset_index,
                "question": item["question"],
                "gold_answer": item["answer"],
                "llm_answer": llm_answer,
                "retrieved_context": context,
                "retrieved_passages": retrieved,
                "supporting_titles": sorted(supporting_titles),
                "retrieved_any_gold": any_gold,
                "retrieved_all_gold": all_gold,
                "metrics": metrics,
                "llm_judge": llm_judge,
                "gold_answer_probability": probability,
                "running_em_accuracy": 0.0,
                "running_average_f1": 0.0,
            }
            result["running_em_accuracy"] = em_sum / evaluated_so_far
            result["running_average_f1"] = f1_sum / evaluated_so_far
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
        "llm_exact_match_accuracy": em_sum / evaluated if evaluated else 0.0,
        "llm_average_f1": f1_sum / evaluated if evaluated else 0.0,
        "llm_judge_valid_examples": judge_valid_count,
        "llm_judge_invalid_examples": evaluated - judge_valid_count,
        "llm_judge_accuracy": (
            judge_correct_count / judge_valid_count if judge_valid_count else 0.0
        ),
        "average_gold_sequence_probability": (
            sum(r["gold_answer_probability"]["sequence_probability"] for r in results)
            / evaluated if evaluated else 0.0
        ),
        "average_gold_mean_token_probability": (
            sum(r["gold_answer_probability"]["mean_token_probability"] for r in results)
            / evaluated if evaluated else 0.0
        ),
    }
    aggregate["hybrid_judge_accuracy"] = aggregate["llm_judge_accuracy"]

    with (output_dir / "baseline_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "baseline_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    with (output_dir / "baseline_failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    config = {**vars(args), "environment": environment_metadata()}
    with (output_dir / "baseline_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    csv_columns = [
        "id", "question", "gold_answer", "llm_answer", "retrieved_context",
        "retrieved_any_gold", "retrieved_all_gold", "em", "f1",
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
                    "question": result["question"],
                    "gold_answer": result["gold_answer"],
                    "llm_answer": result["llm_answer"],
                    "retrieved_context": result["retrieved_context"],
                    "retrieved_any_gold": result["retrieved_any_gold"],
                    "retrieved_all_gold": result["retrieved_all_gold"],
                    "em": result["metrics"]["em"],
                    "f1": result["metrics"]["f1"],
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
    print(f"LLM Exact Match Accuracy          : {aggregate['llm_exact_match_accuracy'] * 100:.2f}%")
    print(f"LLM Average F1                    : {aggregate['llm_average_f1'] * 100:.2f}%")
    print(f"Hybrid Judge Accuracy             : {aggregate['hybrid_judge_accuracy'] * 100:.2f}%")
    print(f"Hybrid Judge valid judgments      : {aggregate['llm_judge_valid_examples']}/{evaluated}")
    print(
        "Gold mean-token probability      : "
        f"{aggregate['average_gold_mean_token_probability'] * 100:.2f}%"
    )
    print(f"Kết quả lưu tại                  : {output_dir.resolve()}")
    print("=" * 100)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline HotpotQA retrieval with Contriever and Qwen 7B"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-bfloat16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-generator-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/colab_baseline")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    run_baseline(build_parser().parse_args())


if __name__ == "__main__":
    main()
