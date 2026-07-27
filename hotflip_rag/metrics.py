from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
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
