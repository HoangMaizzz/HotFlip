from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonical_answer(text: str) -> str:
    """Normalize safe surface variants without making semantic guesses."""
    text = text.lower().strip()
    # Treat common range notations as the same surface form.
    text = re.sub(r"[‐‑‒–—−-]", " to ", text)
    text = re.sub(r"\b(until|till|through|thru)\b", " to ", text)
    text = re.sub(r"\bfrom\s+(?=\d)", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(text.split())


def contains_complete_answer(prediction: str, gold: str) -> bool:
    """True when the complete canonical gold phrase occurs in the prediction."""
    canonical_prediction = canonical_answer(prediction)
    canonical_gold = canonical_answer(gold)
    if not canonical_gold:
        return False
    padded_prediction = f" {canonical_prediction} "
    padded_gold = f" {canonical_gold} "
    return padded_gold in padded_prediction


def contains_shortened_name(prediction: str, gold: str) -> bool:
    """Accept a conservative shortened proper name, e.g. dropping a first name/title."""
    original_tokens = re.findall(r"[A-Za-z]+", gold)
    if len(original_tokens) < 3 or sum(token[:1].isupper() for token in original_tokens) < 2:
        return False
    gold_tokens = canonical_answer(gold).split()
    prediction_text = f" {canonical_answer(prediction)} "
    if len(gold_tokens) < 3:
        return False
    # Only allow dropping one edge token. This accepts "Lee Hazlewood" for
    # "Barton Lee Hazlewood" without treating a lone surname as sufficient.
    aliases = (gold_tokens[1:], gold_tokens[:-1])
    return any(
        len(alias) >= 2 and f" {' '.join(alias)} " in prediction_text
        for alias in aliases
    )


def has_strong_conflict(prediction: str, answer: str) -> bool:
    """Detect explicit denial of an otherwise contained answer."""
    prediction_text = f" {canonical_answer(prediction)} "
    answer_text = canonical_answer(answer)
    if not answer_text:
        return False
    patterns = (
        f" not {answer_text} ",
        f" rather than {answer_text} ",
        f" instead of {answer_text} ",
        f" {answer_text} is wrong ",
        f" {answer_text} is incorrect ",
        f" {answer_text} is false ",
    )
    return any(pattern in prediction_text for pattern in patterns)


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
