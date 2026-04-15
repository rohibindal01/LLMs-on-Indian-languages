"""ROUGE-L metric for summarization tasks."""

from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": round(sum(r1) / len(r1), 4),
        "rouge2": round(sum(r2) / len(r2), 4),
        "rougeL": round(sum(rl) / len(rl), 4),
    }
