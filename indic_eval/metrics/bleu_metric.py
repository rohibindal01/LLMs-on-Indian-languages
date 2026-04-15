"""BLEU and chrF metrics for translation tasks."""

import sacrebleu


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
    }
