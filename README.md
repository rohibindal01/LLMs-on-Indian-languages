# 🌐 Indic LLM Eval Harness

An open-source benchmarking toolkit to evaluate Large Language Models (LLMs) on **Indian languages** — Hindi, Marathi, Tamil, Bengali, Telugu, and more.

Most LLM eval frameworks (LM-Eval Harness, HELM) are heavily English-centric. This project fills that gap.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)

---

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/indic-llm-eval
cd indic-llm-eval
pip install -r requirements.txt

# Run eval on Mistral 7B in Hindi
python -m indic_eval.cli evaluate \
  --model mistral-7b \
  --lang hi \
  --tasks qa,summarization,nli
```

---

## 📦 Supported Tasks

| Task | Dataset | Metric |
|---|---|---|
| Question Answering | IndicQA (AI4Bharat) | Exact Match, F1 |
| Summarization | IndicWikisumm | ROUGE-L |
| NLI / Entailment | IndicXNLI | Accuracy |
| Sentiment Analysis | IndicSentiment | Macro F1 |
| Translation | IndicTrans2 | BLEU, chrF |

---

## 🤖 Supported Models

| Model | Backend | Cost |
|---|---|---|
| Mistral 7B, Gemma 2B | HuggingFace Inference API | Free tier |
| LLaMA 3 8B/70B | Groq API | Free tier |
| Gemini 1.5 Flash | Google AI Studio | Free tier |
| Any GGUF model | Ollama (local) | Free |

---

## 🌍 Supported Languages

`hi` Hindi · `mr` Marathi · `ta` Tamil · `bn` Bengali · `te` Telugu · `gu` Gujarati · `kn` Kannada · `ml` Malayalam · `pa` Punjabi · `ur` Urdu

---

## 📊 Leaderboard (Hindi)

| Model | QA (F1) | ROUGE-L | NLI Acc | Avg |
|---|---|---|---|---|
| Gemini 1.5 Flash | 81% | 0.44 | 78% | 79 |
| LLaMA 3 70B (Groq) | 76% | 0.39 | 74% | 75 |
| Mistral 7B | 61% | 0.31 | 58% | 60 |

> Results are auto-generated in `results/leaderboard.json` after each eval run.

---

## 🗂️ Project Structure

```
indic-llm-eval/
├── indic_eval/
│   ├── cli.py              # CLI entry point
│   ├── runner.py           # Core eval loop
│   ├── tasks/              # Task definitions
│   ├── models/             # Model backends
│   ├── datasets/           # Dataset loaders
│   └── metrics/            # Metric calculators
├── configs/                # YAML configs per model/lang
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── results/                # Eval outputs & leaderboard
└── docs/                   # Documentation
```

---

## ⚙️ Configuration

Copy and edit a config:

```bash
cp configs/default.yaml configs/my_run.yaml
python -m indic_eval.cli evaluate --config configs/my_run.yaml
```


