# Contributing to Indic LLM Eval Harness

Thank you for your interest in contributing! This project thrives on community contributions — new tasks, languages, model backends, and bug fixes are all welcome.

## Ways to contribute

- Add a new benchmark task
- Add a new language to an existing task
- Add a new model backend (e.g. Claude API, Cohere)
- Improve prompt templates for a language
- Fix bugs or improve test coverage
- Write documentation

## Development setup

```bash
git clone https://github.com/your-username/indic-llm-eval
cd indic-llm-eval
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Fill in your API keys
```

Run tests:
```bash
pytest tests/ -v
python scripts/smoke_test.py
```

## Adding a new task

1. Add a prompt function in `indic_eval/tasks/prompts.py`.
2. Register the task in `indic_eval/tasks/registry.py` with its dataset, metric, and prompt function.
3. Add tests in `tests/test_prompts.py`.

## Adding a new model backend

1. Create `indic_eval/models/your_model.py` extending `BaseModel`.
2. Implement the `generate(prompt, max_tokens)` method.
3. Register a shortname in `indic_eval/models/factory.py`.
4. Add a config example in `configs/`.

## Pull request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Smoke test passes (`python scripts/smoke_test.py`)
- [ ] No API keys committed
- [ ] New features have at least one test

## Code style

We use `ruff` for linting. Run `ruff check indic_eval/` before submitting.
