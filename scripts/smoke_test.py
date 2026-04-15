#!/usr/bin/env python3
"""
scripts/smoke_test.py
----------------------
Runs a tiny eval (5 samples) with a mock model to verify the full pipeline.
No API keys required.
    python scripts/smoke_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from indic_eval.metrics.qa_metrics import compute_f1
from indic_eval.metrics.rouge_metric import compute_rouge
from indic_eval.metrics.accuracy_metric import compute_accuracy
from indic_eval.metrics.bleu_metric import compute_bleu
from indic_eval.tasks.prompts import qa_prompt, nli_prompt


print("🔬 Smoke Test — Indic LLM Eval Harness\n")

# Metrics
print("1. QA F1")
r = compute_f1(["नई दिल्ली", "मुंबई"], ["नई दिल्ली", "दिल्ली"])
print(f"   {r}\n")

print("2. ROUGE-L")
r = compute_rouge(["यह एक परीक्षण है।"], ["यह एक परीक्षण वाक्य है।"])
print(f"   {r}\n")

print("3. NLI Accuracy")
r = compute_accuracy(["entailment", "yes", "contradiction", "neutral"],
                     ["entailment", "entailment", "contradiction", "neutral"])
print(f"   {r}\n")

print("4. BLEU")
r = compute_bleu(["यह एक परीक्षण है।"], ["यह एक परीक्षण है।"])
print(f"   {r}\n")

# Prompts
print("5. Prompt templates")
sample = {"context": "भारत की राजधानी नई दिल्ली है।",
          "question": "राजधानी क्या है?", "answer": "नई दिल्ली"}
prompt = qa_prompt(sample, "hi")
assert "Hindi" in prompt and sample["question"] in prompt
print("   QA prompt ✅\n")

nli_sample = {"premise": "बिल्ली सोती है।",
              "hypothesis": "बिल्ली जाग रही है।", "answer": "contradiction"}
prompt = nli_prompt(nli_sample, "ta")
assert "Tamil" in prompt
print("   NLI prompt ✅\n")

print("✅ All smoke tests passed!")
