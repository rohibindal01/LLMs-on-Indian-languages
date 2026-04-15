"""Accuracy and Macro-F1 for classification tasks (NLI, sentiment)."""

from collections import Counter
import re


LABEL_ALIASES = {
    "entailment": "entailment", "yes": "entailment", "true": "entailment",
    "contradiction": "contradiction", "no": "contradiction", "false": "contradiction",
    "neutral": "neutral",
    "positive": "positive", "pos": "positive",
    "negative": "negative", "neg": "negative",
}


def normalize_label(text: str) -> str:
    text = text.lower().strip()
    first_word = re.split(r"[\s,.\n]", text)[0]
    return LABEL_ALIASES.get(first_word, text)


def compute_accuracy(predictions: list[str], references: list[str]) -> dict:
    norm_preds = [normalize_label(p) for p in predictions]
    norm_refs = [normalize_label(str(r)) for r in references]

    correct = sum(p == r for p, r in zip(norm_preds, norm_refs))
    accuracy = correct / len(norm_refs) if norm_refs else 0.0

    # Macro F1
    labels = list(set(norm_refs))
    f1_per_label = []
    for label in labels:
        tp = sum(p == label and r == label for p, r in zip(norm_preds, norm_refs))
        fp = sum(p == label and r != label for p, r in zip(norm_preds, norm_refs))
        fn = sum(p != label and r == label for p, r in zip(norm_preds, norm_refs))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_per_label.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    macro_f1 = sum(f1_per_label) / len(f1_per_label) if f1_per_label else 0.0
    return {
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
    }
