"""Metrics factory — returns the right scoring function."""

from indic_eval.metrics.qa_metrics import compute_f1
from indic_eval.metrics.rouge_metric import compute_rouge
from indic_eval.metrics.accuracy_metric import compute_accuracy
from indic_eval.metrics.bleu_metric import compute_bleu


METRIC_MAP = {
    "f1":       compute_f1,
    "rouge":    compute_rouge,
    "accuracy": compute_accuracy,
    "bleu":     compute_bleu,
}


def get_metric(metric_name: str):
    if metric_name not in METRIC_MAP:
        raise ValueError(f"Unknown metric '{metric_name}'. Available: {list(METRIC_MAP.keys())}")
    return METRIC_MAP[metric_name]
