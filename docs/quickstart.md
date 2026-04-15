# Quickstart Guide

## 1. Clone and install

```bash
git clone https://github.com/your-username/indic-llm-eval
cd indic-llm-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and fill in the keys you need
```

| Backend | Where to get a free key |
|---|---|
| HuggingFace | https://huggingface.co/settings/tokens |
| Groq (LLaMA 3) | https://console.groq.com |
| Gemini | https://aistudio.google.com/app/apikey |
| Ollama | Install locally — no key needed |

## 3. Run your first evaluation

```bash
# Gemini Flash on Hindi QA
python -m indic_eval.cli evaluate --model gemini-flash --lang hi --tasks qa

# LLaMA 3 70B on Marathi (NLI + Sentiment) via Groq
python -m indic_eval.cli evaluate --model llama3-70b --lang mr --tasks nli,sentiment

# Local model via Ollama (no API key)
ollama pull llama3
python -m indic_eval.cli evaluate --model llama3-8b --lang ta --tasks qa,summarization

# Using a config file
python -m indic_eval.cli evaluate --config configs/groq_hindi.yaml
```

## 4. View the leaderboard

```bash
python -m indic_eval.cli leaderboard
```

## 5. Run smoke test (no API keys needed)

```bash
python scripts/smoke_test.py
```
