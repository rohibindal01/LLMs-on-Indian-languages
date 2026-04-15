"""Tests for prompt template functions."""

import pytest
from indic_eval.tasks.prompts import (
    qa_prompt, summarization_prompt, nli_prompt, sentiment_prompt, translation_prompt
)

QA_SAMPLE = {"context": "भारत की राजधानी नई दिल्ली है।", "question": "भारत की राजधानी क्या है?", "answer": "नई दिल्ली"}
SUMM_SAMPLE = {"document": "यह एक लंबा लेख है।", "answer": "यह एक लेख है।"}
NLI_SAMPLE = {"premise": "बिल्ली सोती है।", "hypothesis": "बिल्ली जाग रही है।", "answer": "contradiction"}
SENT_SAMPLE = {"text": "यह फिल्म बहुत अच्छी है।", "answer": "positive"}
TRANS_SAMPLE = {"source": "India is a great country.", "answer": "भारत एक महान देश है।"}


class TestPrompts:
    def test_qa_prompt_contains_context(self):
        prompt = qa_prompt(QA_SAMPLE, "hi")
        assert QA_SAMPLE["context"] in prompt

    def test_qa_prompt_contains_question(self):
        prompt = qa_prompt(QA_SAMPLE, "hi")
        assert QA_SAMPLE["question"] in prompt

    def test_qa_prompt_mentions_language(self):
        prompt = qa_prompt(QA_SAMPLE, "hi")
        assert "Hindi" in prompt

    def test_summarization_prompt_contains_document(self):
        prompt = summarization_prompt(SUMM_SAMPLE, "mr")
        assert SUMM_SAMPLE["document"] in prompt

    def test_nli_prompt_contains_premise(self):
        prompt = nli_prompt(NLI_SAMPLE, "ta")
        assert NLI_SAMPLE["premise"] in prompt

    def test_nli_prompt_contains_hypothesis(self):
        prompt = nli_prompt(NLI_SAMPLE, "ta")
        assert NLI_SAMPLE["hypothesis"] in prompt

    def test_sentiment_prompt_contains_text(self):
        prompt = sentiment_prompt(SENT_SAMPLE, "bn")
        assert SENT_SAMPLE["text"] in prompt

    def test_translation_prompt_contains_source(self):
        prompt = translation_prompt(TRANS_SAMPLE, "hi")
        assert TRANS_SAMPLE["source"] in prompt

    def test_unknown_lang_falls_back(self):
        prompt = qa_prompt(QA_SAMPLE, "xx")
        assert "xx" in prompt
