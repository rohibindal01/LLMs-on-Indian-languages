# Adding a New Task

This guide walks you through adding a new benchmark task end-to-end.

## Step 1 — Write a prompt function

Open `indic_eval/tasks/prompts.py` and add a function:

```python
def my_task_prompt(sample: dict, lang: str) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    return (
        f"Given the following {lang_name} text, do X.\n"
        f"Text: {sample['input']}\n"
        f"Answer:"
    )
```

Your function receives a `sample` dict (one row from the HuggingFace dataset) and the `lang` code.

## Step 2 — Register the task

Open `indic_eval/tasks/registry.py` and add an entry:

```python
from indic_eval.tasks.prompts import my_task_prompt

TASK_REGISTRY["my_task"] = {
    "dataset":     "huggingface-org/dataset-name",   # HF dataset ID
    "metric":      "accuracy",                        # f1 | rouge | accuracy | bleu
    "prompt_fn":   my_task_prompt,
    "description": "One line description",
}
```

## Step 3 — Check the dataset fields

Open `indic_eval/datasets/loader.py` and add a config for your dataset if it is new:

```python
"huggingface-org/dataset-name": {
    "subset_key": "language",    # column used to filter by language, or None
    "split": "test",
    "fields": {
        "input":  "text_column",   # maps sample["input"] → dataset["text_column"]
        "answer": "label_column",
    },
},
```

## Step 4 — Add tests

In `tests/test_prompts.py`, add a test for your prompt:

```python
MY_SAMPLE = {"input": "sample text", "answer": "label"}

def test_my_task_prompt_contains_input():
    prompt = my_task_prompt(MY_SAMPLE, "hi")
    assert MY_SAMPLE["input"] in prompt
```

## Step 5 — Run it

```bash
python -m indic_eval.cli evaluate --model gemini-flash --lang hi --tasks my_task
```
