from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonical_semantic_answer(text: str) -> str:
    """Normalize unambiguous surface variants before asking an LLM judge.

    This is intentionally conservative. It handles formatting differences such
    as ``1969 until 1974`` versus ``1969-1974`` without making fuzzy semantic
    decisions that belong to the judge model.
    """
    text = text.lower().strip()
    text = re.sub(r"[‐‑‒–—−-]", " to ", text)
    text = re.sub(r"\b(until|till|through|thru)\b", " to ", text)
    text = re.sub(r"\bfrom\s+(?=\d)", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(text.split())


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    em = float(pred == truth)
    pred_tokens, truth_tokens = pred.split(), truth.split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if not pred_tokens or not truth_tokens:
        f1 = float(pred_tokens == truth_tokens)
        precision = recall = f1
    elif overlap == 0:
        f1 = precision = recall = 0.0
    else:
        precision = overlap / len(pred_tokens)
        recall = overlap / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return {"em": em, "f1": f1, "precision": precision, "recall": recall}
