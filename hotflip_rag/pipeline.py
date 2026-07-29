from __future__ import annotations

import csv
import difflib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .hotflip import ContrieverHotFlipAttacker, HotFlipConfig, mean_pool
from .metrics import normalize_answer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hotpot_documents(item: dict[str, Any]) -> list[dict[str, str]]:
    """Return the ten HotpotQA documents separately with stable document IDs."""
    titles = list(item["context"]["title"])
    sentence_groups = list(item["context"]["sentences"])
    supporting_titles = set(item["supporting_facts"]["title"])
    return [
        {
            "document_id": f"{item.get('id', 'example')}:{index}",
            "title": title,
            "text": f"{title}: {' '.join(sentences)}",
            "source": "gold" if title in supporting_titles else "distractor",
        }
        for index, (title, sentences) in enumerate(zip(titles, sentence_groups))
    ]


def select_gold_document(
    question: str,
    documents: list[dict[str, str]],
    retriever: "ContrieverRetriever",
) -> dict[str, Any]:
    """Select exactly one supporting document: the gold doc closest to the query."""
    gold_docs = [doc for doc in documents if doc["source"] == "gold"]
    if not gold_docs:
        raise ValueError("No supporting-fact document found")
    return retriever.retrieve(question, gold_docs, top_k=1)[0]


def hotpot_contexts(item: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Backward-compatible helper; new attack code should use ``hotpot_documents``."""
    documents = hotpot_documents(item)
    gold_docs = [doc for doc in documents if doc["source"] == "gold"]
    distractors = [doc for doc in documents if doc["source"] == "distractor"]
    gold = {
        "title": " + ".join(doc["title"] for doc in gold_docs),
        "text": "\n\n".join(doc["text"] for doc in gold_docs),
    }
    return gold, distractors


class ContrieverRetriever:
    def __init__(self, model, tokenizer, device: torch.device, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.calls = 0

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        output = self.model(**batch)
        return F.normalize(mean_pool(output.last_hidden_state, batch["attention_mask"]), dim=-1)

    @torch.no_grad()
    def retrieve(
        self, question: str, passages: list[dict[str, str]], top_k: int = 1
    ) -> list[dict[str, Any]]:
        self.calls += 1
        query = self._encode([question])
        contexts = self._encode([passage["text"] for passage in passages])
        scores = (contexts @ query.T).squeeze(1)
        top_k = min(top_k, len(passages))
        values, indices = torch.topk(scores, k=top_k)
        return [
            {**passages[int(index)], "score": float(value)}
            for value, index in zip(values, indices)
        ]


class QAGenerator:
    def __init__(self, model, tokenizer, max_input_tokens: int = 3072):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.device = next(model.parameters()).device

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        return (
            "You are an extractive question-answering engine.\n"
            "Use only the supplied context.\n"
            "Return exactly one shortest final-answer span.\n"
            "Do not explain, justify, reason, introduce the answer, repeat the "
            "question, or add any fact beyond the answer itself.\n"
            "A name, date, number, place, or yes/no must be returned by itself, "
            "not inside a sentence.\n"
            "Write the answer between <answer> and </answer>. Produce no text "
            "before <answer> or after </answer>.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "<answer>"
        )

    def _generation_prompt(self, question: str, context: str) -> str:
        """Use the instruction model's chat format and seed the answer tag."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an extractive question-answering engine. Your entire "
                    "response must be exactly one shortest final-answer span. Never "
                    "give an explanation, reasoning, an introductory phrase, a full "
                    "sentence around the answer, or any additional fact. Output the "
                    "answer only inside <answer>...</answer>."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use only the supplied context.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Return only the shortest answer. Examples of valid responses: "
                    "<answer>yes</answer>, <answer>1969–1974</answer>, "
                    "<answer>Richard Nixon</answer>."
                ),
            },
        ]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            formatted = apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted + "<answer>"
        return self.build_prompt(question, context)

    def _wrong_target_prompt(
        self,
        question: str,
        gold_answer: str,
        baseline_answer: str,
        attempt: int,
        rejected_answers: list[str],
    ) -> str:
        rejected = ", ".join(repr(answer) for answer in rejected_answers) or "(none)"
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate one concise, plausible, but definitely incorrect "
                    "answer for an adversarial QA experiment. The answer must have "
                    "the same semantic type as the reference (person, place, date, "
                    "yes/no, title, number, etc.), but it must not be the reference, "
                    "an alias, a paraphrase, or a partially correct answer. Return "
                    "only the false answer inside <answer>...</answer>."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Reference answer: {gold_answer}\n"
                    f"Baseline model answer: {baseline_answer}\n"
                    f"Generation attempt: {attempt}\n"
                    f"Previously rejected answers: {rejected}\n\n"
                    "Produce a different plausible false answer. Do not explain."
                ),
            },
        ]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            formatted = apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted + "<answer>"
        return (
            messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n<answer>"
        )

    @staticmethod
    def extract_final_answer(generated_text: str) -> str:
        """Strip formatting and discard anything after the closing answer tag."""
        answer = generated_text.strip()
        lowered = answer.lower()
        if "<answer>" in lowered:
            start = lowered.find("<answer>") + len("<answer>")
            answer = answer[start:]
            lowered = answer.lower()
        if "</answer>" in lowered:
            answer = answer[:lowered.find("</answer>")]
        answer = next(
            (line.strip() for line in answer.splitlines() if line.strip()),
            "",
        )
        for prefix in (
            "Final answer:", "Final Answer:", "Answer:", "Short Answer:"
        ):
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        return answer.strip().strip("`").strip()

    @torch.no_grad()
    def generate(self, question: str, context: str, max_new_tokens: int = 20) -> str:
        prompt = self._generation_prompt(question, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return self.extract_final_answer(answer)

    @torch.no_grad()
    def generate_wrong_target(
        self,
        question: str,
        gold_answer: str,
        baseline_answer: str,
        attempt: int,
        rejected_answers: list[str],
        max_new_tokens: int = 20,
    ) -> str:
        prompt = self._wrong_target_prompt(
            question, gold_answer, baseline_answer, attempt, rejected_answers
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return self.extract_final_answer(generated)

    @staticmethod
    def build_judge_prompt(question: str, gold_answer: str, predicted_answer: str) -> str:
        return (
            "You are a careful semantic answer evaluator.\n"
            "Judge the entire predicted answer against the reference answer in the "
            "context of the question, not merely whether reference words occur.\n"
            "Return YES when the prediction expresses the same complete core answer.\n"
            "Accept capitalization and punctuation differences, paraphrases, aliases, "
            "abbreviations, equivalent units, and equivalent date/range formats.\n"
            "Extra explanation is allowed only when every added claim is compatible "
            "with the reference and does not change or extend the answer.\n"
            "Important: merely containing the reference answer is NOT sufficient. "
            "Return NO if another clause adds a conflicting entity, date, date range, "
            "number, location, or mutually incompatible factual claim.\n"
            "Examples:\n"
            "- Reference '1969 until 1974'; prediction '1969-1974'.\n"
            "  Judgment: YES, because only the range format differs.\n"
            "- Reference '1969 until 1974'; prediction '1969-1974, 1974-1978'.\n"
            "  Judgment: NO, because the extra range extends/conflicts with the answer.\n"
            "- Reference 'Paris'; prediction 'Paris, the capital of France.'\n"
            "  Judgment: YES, because the added description is compatible.\n"
            "- Reference 'Paris'; prediction 'Paris, but the actual answer is London.'\n"
            "  Judgment: NO, because the later clause contradicts the contained answer.\n"
            "- Reference 'Richard Nixon'; prediction 'President Richard Nixon'.\n"
            "  Judgment: YES, because the title does not change the entity.\n"
            "When uncertain, decide whether a reader would learn the same complete "
            "answer to the question from the full prediction.\n"
            "Return exactly one token: YES or NO. Do not explain.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {gold_answer}\n"
            f"Predicted answer: {predicted_answer}\n"
            "Judgment:"
        )

    @torch.no_grad()
    def judge_answer(
        self, question: str, gold_answer: str, predicted_answer: str
    ) -> dict[str, Any]:
        normalized_gold = normalize_answer(gold_answer)
        normalized_prediction = normalize_answer(predicted_answer)
        if normalized_gold and normalized_gold == normalized_prediction:
            return {
                "correct": True,
                "raw": "EXACT_MATCH",
                "method": "normalized_exact_match",
                "normalized_gold": normalized_gold,
                "normalized_prediction": normalized_prediction,
            }
        prompt = self.build_judge_prompt(question, gold_answer, predicted_answer)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        normalized = raw.upper().strip()
        if normalized.startswith("YES"):
            correct: bool | None = True
        elif normalized.startswith("NO"):
            correct = False
        else:
            correct = None
        return {
            "correct": correct,
            "raw": raw,
            "method": "llm_judge",
            "normalized_gold": normalized_gold,
            "normalized_prediction": normalized_prediction,
        }

    @torch.no_grad()
    def gold_answer_probability(
        self, question: str, context: str, gold_answer: str
    ) -> dict[str, float]:
        """Teacher-forced probability of the gold answer for a decoder-only LM.

        ``sequence_probability`` is the product of answer-token probabilities
        and can be extremely small. ``mean_token_probability`` is exp(-mean NLL)
        and is the more readable confidence-like value.
        """
        if getattr(self.model.config, "is_encoder_decoder", False):
            raise NotImplementedError("The current baseline uses a decoder-only Qwen model")
        prompt = self._generation_prompt(question, context)
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids[0]
        answer_ids = self.tokenizer(
            gold_answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        available_prompt = self.max_input_tokens - len(answer_ids)
        if available_prompt < 1:
            raise ValueError("Gold answer alone exceeds the configured model input length")
        prompt_ids = prompt_ids[-available_prompt:]
        input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(self.device)
        labels = input_ids.clone()
        labels[:, : len(prompt_ids)] = -100
        output = self.model(input_ids=input_ids, labels=labels)
        mean_nll = float(output.loss)
        token_count = int(len(answer_ids))
        log_probability = -mean_nll * token_count
        sequence_probability = math.exp(log_probability) if log_probability > -745 else 0.0
        return {
            "mean_nll": mean_nll,
            "answer_token_count": token_count,
            "log_probability": log_probability,
            "sequence_probability": sequence_probability,
            "mean_token_probability": math.exp(-mean_nll),
        }


def context_diff(original: str, attacked: str) -> str:
    return " ".join(
        difflib.ndiff(original.split(), attacked.split())
    )


def load_targets(path: str | None, direct_target: str | None) -> dict[str, str] | list[str] | None:
    if direct_target:
        return {"*": direct_target}
    if not path:
        return None
    target_path = Path(path)
    if target_path.suffix.lower() == ".pkl":
        import pickle

        with target_path.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, (list, dict)):
            raise ValueError("Pickle target file must contain a list or dictionary")
        return value
    if target_path.suffix.lower() == ".jsonl":
        targets: dict[str, str] = {}
        with target_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                targets[str(row.get("id", len(targets)))] = str(
                    row.get("target_answer", row.get("answer"))
                )
        return targets
    with target_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, (list, dict)):
        raise ValueError("JSON target file must contain a list or dictionary")
    return value


def target_for_example(
    targets: dict[str, str] | list[str] | None, item: dict[str, Any], index: int
) -> str | None:
    if targets is None:
        return None
    if isinstance(targets, list):
        return str(targets[index]) if index < len(targets) else None
    return targets.get(str(item["id"]), targets.get(str(index), targets.get("*")))


def inspection_report(
    retriever_model, retriever_tokenizer, generator_model, generator_tokenizer,
    args, selected_count: int,
) -> dict[str, Any]:
    embedding = retriever_model.get_input_embeddings().weight
    generator_class = type(generator_model).__name__
    encoder_decoder = bool(getattr(generator_model.config, "is_encoder_decoder", False))
    report = {
        "retriever_model": args.retriever_model,
        "retriever_class": type(retriever_model).__name__,
        "retriever_tokenizer_class": type(retriever_tokenizer).__name__,
        "generator_model": args.generator_model,
        "generator_class": generator_class,
        "generator_tokenizer_class": type(generator_tokenizer).__name__,
        "generator_type": "encoder-decoder" if encoder_decoder else "decoder-only",
        "retriever_vocabulary_size": len(retriever_tokenizer),
        "retriever_embedding_shape": list(embedding.shape),
        "retriever_dtype": str(embedding.dtype),
        "retriever_device": str(embedding.device),
        "dataset": "hotpot_qa/distractor",
        "split": args.split,
        "selected_examples": selected_count,
        "retrieved_passages": args.top_k,
        "max_context_tokens": args.max_context_tokens,
        "attack_mode": args.attack_mode,
        "token_replacement_budget": args.max_token_changes,
        "attack_order": "Gold Context HotFlip -> Contriever retrieval -> generator",
    }
    print("\nREPOSITORY / RUN INSPECTION")
    for key, value in report.items():
        print(f"  {key}: {value}")
    return report


def run_pipeline(args) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model)
    retriever_model = AutoModel.from_pretrained(args.retriever_model).to(device).eval()
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    generator_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.generator_dtype]
    generator_model = AutoModelForCausalLM.from_pretrained(
        args.generator_model,
        dtype=generator_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        generator_model.to(device)
    generator_model.eval()

    dataset = load_dataset("hotpot_qa", "distractor", split=args.split)
    targets = load_targets(args.target_answer_file, args.target_answer)
    # A positional target list (including the existing 300-answer pickle) is
    # indexed by the original HotpotQA dataset position, never by shuffled order.
    selectable_count = (
        min(len(dataset), len(targets)) if isinstance(targets, list) else len(dataset)
    )
    indices = list(range(selectable_count))
    random.Random(args.seed).shuffle(indices)
    selected_indices = indices[: args.num_examples]
    report = inspection_report(
        retriever_model, retriever_tokenizer, generator_model, generator_tokenizer,
        args, len(selected_indices),
    )

    config = HotFlipConfig(
        attack_mode=args.attack_mode,
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
        untargeted_answer_weight=args.untargeted_answer_weight,
        score_chunk_size=args.score_chunk_size,
        max_context_tokens=args.max_context_tokens,
    )
    attacker = ContrieverHotFlipAttacker(
        retriever_model, retriever_tokenizer, config, device
    )
    retriever = ContrieverRetriever(
        retriever_model, retriever_tokenizer, device, args.max_context_tokens
    )
    generator = QAGenerator(generator_model, generator_tokenizer, args.max_generator_input_tokens)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset_index in selected_indices:
        item = dataset[dataset_index]
        started = time.perf_counter()
        try:
            clean_pool = hotpot_documents(item)
            selected_gold = select_gold_document(
                item["question"], clean_pool, retriever
            )
            clean_retrieved = retriever.retrieve(item["question"], clean_pool, args.top_k)
            clean_context = "\n\n".join(doc["text"] for doc in clean_retrieved)
            clean_answer = generator.generate(item["question"], clean_context)
            clean_judge = generator.judge_answer(
                item["question"], item["answer"], clean_answer
            )
            if args.only_clean_correct and clean_judge["correct"] is not True:
                failures.append({"id": str(item["id"]), "reason": "clean_answer_incorrect"})
                continue
            target_answer = target_for_example(targets, item, dataset_index)
            if args.attack_mode == "targeted" and not target_answer:
                failures.append({"id": str(item["id"]), "reason": "missing_target_answer"})
                continue

            attack_result = attacker.attack(
                item["question"],
                selected_gold["text"],
                target_answer=target_answer,
                avoid_answer=(
                    item["answer"] if args.attack_mode == "untargeted" else None
                ),
            )
            attacked_gold = {
                **selected_gold,
                "text": attack_result.attacked_text,
                "source": "attacked_gold",
            }
            attacked_pool = [
                attacked_gold
                if doc["document_id"] == selected_gold["document_id"]
                else doc
                for doc in clean_pool
            ]
            # Retrieval happens only after the selected gold document has been flipped.
            attacked_retrieved = retriever.retrieve(item["question"], attacked_pool, args.top_k)
            attacked_context = "\n\n".join(doc["text"] for doc in attacked_retrieved)
            attacked_answer = generator.generate(item["question"], attacked_context)
            attacked_judge = generator.judge_answer(
                item["question"], item["answer"], attacked_answer
            )
            attacked_vs_clean_judge = (
                generator.judge_answer(
                    item["question"], clean_answer, attacked_answer
                )
                if args.attack_mode == "untargeted"
                else None
            )
            target_judge = (
                generator.judge_answer(item["question"], target_answer, attacked_answer)
                if target_answer
                else None
            )
            clean_gold_selected = any(
                doc["document_id"] == selected_gold["document_id"]
                for doc in clean_retrieved
            )
            attacked_gold_selected = any(
                doc["document_id"] == selected_gold["document_id"]
                for doc in attacked_retrieved
            )
            if args.attack_mode == "untargeted":
                strict_success = (
                    clean_judge["correct"] is True
                    and attacked_judge["correct"] is False
                    and attacked_gold_selected
                )
                relaxed_success = (
                    attacked_judge["correct"] is False
                    and attacked_vs_clean_judge["correct"] is False
                    and attacked_gold_selected
                )
                success = strict_success
            else:
                success = bool(target_judge and target_judge["correct"] is True)
                strict_success = success
                relaxed_success = None
            retrieval_attack_success = attacked_gold_selected
            result = {
                "id": str(item["id"]),
                "dataset_index": dataset_index,
                "question": item["question"],
                "gold_answer": item["answer"],
                "target_answer": target_answer,
                "attack_mode": args.attack_mode,
                "attack_order": "flip_gold_before_retrieval",
                "attacked_document_id": selected_gold["document_id"],
                "attacked_document_title": selected_gold["title"],
                "original_attacked_document": selected_gold["text"],
                "modified_attacked_document": attack_result.attacked_text,
                "clean_retrieved_contexts": clean_retrieved,
                "attacked_retrieved_contexts": attacked_retrieved,
                "clean_gold_context": selected_gold["text"],
                "attacked_gold_context": attack_result.attacked_text,
                "clean_generated_answer": clean_answer,
                "attacked_generated_answer": attacked_answer,
                "clean_judge": clean_judge,
                "attacked_judge": attacked_judge,
                "attacked_vs_clean_judge": attacked_vs_clean_judge,
                "target_judge": target_judge,
                "retrieval_gold_selected_clean": clean_gold_selected,
                "retrieval_attacked_gold_selected": attacked_gold_selected,
                "retrieval_attack_success": retrieval_attack_success,
                "attack_success": success,
                "strict_attack_success": strict_success,
                "relaxed_attack_success": relaxed_success,
                "hotflip": attack_result.to_dict(),
                "context_diff": context_diff(
                    selected_gold["text"], attack_result.attacked_text
                ),
                "runtime_seconds": time.perf_counter() - started,
            }
            results.append(result)
            print(
                f"[{len(results)}/{args.num_examples}] {item['id']} "
                f"clean={clean_answer!r} attacked={attacked_answer!r} success={success}"
            )
        except Exception as error:
            failures.append({"id": str(item.get("id", dataset_index)), "reason": repr(error)})
            print(f"[FAILED] {item.get('id', dataset_index)}: {error}")
            if args.fail_fast:
                raise

    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)

    attempted = len(results)
    aggregate = {
        "selected": len(selected_indices),
        "attacked": attempted,
        "skipped_or_failed": len(failures),
        "attack_success_rate": (
            sum(result["attack_success"] for result in results) / attempted if attempted else 0.0
        ),
        "strict_attack_success_rate": (
            sum(result["strict_attack_success"] for result in results) / attempted
            if attempted else 0.0
        ),
        "relaxed_attack_success_rate": (
            sum(result["relaxed_attack_success"] is True for result in results) / attempted
            if attempted and args.attack_mode == "untargeted" else None
        ),
        "clean_gold_retrieval_rate": (
            sum(result["retrieval_gold_selected_clean"] for result in results) / attempted
            if attempted else 0.0
        ),
        "attacked_gold_retrieval_rate": (
            sum(result["retrieval_attacked_gold_selected"] for result in results) / attempted
            if attempted else 0.0
        ),
        "retrieval_attack_success_rate": (
            sum(result["retrieval_attack_success"] for result in results) / attempted
            if attempted else 0.0
        ),
        "average_token_changes": (
            sum(len(result["hotflip"]["changes"]) for result in results) / attempted
            if attempted else 0.0
        ),
        "average_runtime_seconds": (
            sum(result["runtime_seconds"] for result in results) / attempted if attempted else 0.0
        ),
    }
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    resolved_config = vars(args).copy()
    resolved_config["hotflip"] = asdict(config)
    resolved_config["inspection"] = report
    resolved_config["environment"] = environment_metadata()
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, ensure_ascii=False, indent=2, default=str)

    summary_columns = [
        "id", "question", "gold_answer", "target_answer", "clean_generated_answer",
        "attacked_generated_answer", "retrieval_gold_selected_clean",
        "retrieval_attacked_gold_selected", "retrieval_attack_success",
        "strict_attack_success", "relaxed_attack_success", "attack_success",
        "runtime_seconds",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in summary_columns})
    print(f"\nSaved results to {output_dir.resolve()}")
    print(json.dumps(aggregate, indent=2))
    return aggregate


def environment_metadata() -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_commit = None
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit,
    }
