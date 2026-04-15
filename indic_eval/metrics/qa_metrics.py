"""QA metrics — token-level F1 and Exact Match (same as SQuAD evaluation)."""

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def compute_f1(predictions: list[str], references: list[str]) -> dict:
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    return {
        "f1": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "exact_match": round(sum(em_scores) / len(em_scores) * 100, 2),
    }
