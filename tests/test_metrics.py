"""Tests for all metric functions."""

import pytest
from indic_eval.metrics.qa_metrics import compute_f1, normalize, token_f1
from indic_eval.metrics.rouge_metric import compute_rouge
from indic_eval.metrics.accuracy_metric import compute_accuracy, normalize_label
from indic_eval.metrics.bleu_metric import compute_bleu


class TestQAMetrics:
    def test_exact_match_identical(self):
        result = compute_f1(["नई दिल्ली"], ["नई दिल्ली"])
        assert result["exact_match"] == 100.0

    def test_exact_match_different(self):
        result = compute_f1(["मुंबई"], ["नई दिल्ली"])
        assert result["exact_match"] == 0.0

    def test_f1_partial_overlap(self):
        result = compute_f1(["the cat sat"], ["the cat"])
        assert 0 < result["f1"] < 100

    def test_f1_perfect(self):
        result = compute_f1(["नई दिल्ली"], ["नई दिल्ली"])
        assert result["f1"] == 100.0

    def test_normalize_strips_punctuation(self):
        assert normalize("Hello, World!") == "hello world"

    def test_multiple_samples(self):
        preds = ["दिल्ली", "मुंबई", "चेन्नई"]
        refs  = ["दिल्ली", "दिल्ली", "चेन्नई"]
        result = compute_f1(preds, refs)
        assert 0 < result["f1"] <= 100


class TestRougeMetric:
    def test_perfect_match(self):
        result = compute_rouge(["यह एक परीक्षण है"], ["यह एक परीक्षण है"])
        assert result["rougeL"] == 1.0

    def test_no_overlap(self):
        result = compute_rouge(["abc def"], ["xyz uvw"])
        assert result["rougeL"] == 0.0

    def test_keys_present(self):
        result = compute_rouge(["hello world"], ["hello there"])
        assert all(k in result for k in ["rouge1", "rouge2", "rougeL"])


class TestAccuracyMetric:
    def test_all_correct(self):
        result = compute_accuracy(["entailment", "neutral"], ["entailment", "neutral"])
        assert result["accuracy"] == 100.0

    def test_all_wrong(self):
        result = compute_accuracy(["entailment", "entailment"], ["neutral", "contradiction"])
        assert result["accuracy"] == 0.0

    def test_label_alias_yes(self):
        assert normalize_label("yes") == "entailment"

    def test_label_alias_positive(self):
        assert normalize_label("positive") == "positive"

    def test_label_alias_neg(self):
        assert normalize_label("neg") == "negative"

    def test_macro_f1_present(self):
        result = compute_accuracy(["positive", "negative"], ["positive", "positive"])
        assert "macro_f1" in result


class TestBleuMetric:
    def test_identical_sentences(self):
        result = compute_bleu(["यह एक परीक्षण है"], ["यह एक परीक्षण है"])
        assert result["bleu"] == 100.0

    def test_empty_overlap(self):
        result = compute_bleu(["abc def ghi"], ["xyz uvw rst"])
        assert result["bleu"] == 0.0

    def test_keys_present(self):
        result = compute_bleu(["test sentence"], ["test sentence"])
        assert "bleu" in result and "chrf" in result
