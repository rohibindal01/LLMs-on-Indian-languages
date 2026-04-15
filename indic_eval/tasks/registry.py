"""Registry of all supported evaluation tasks."""

from indic_eval.tasks.prompts import qa_prompt, summarization_prompt, nli_prompt, sentiment_prompt, translation_prompt


TASK_REGISTRY = {
    "qa": {
        "dataset": "ai4bharat/IndicQA",
        "metric": "f1",
        "prompt_fn": qa_prompt,
        "description": "Extractive QA from IndicQA benchmark",
    },
    "summarization": {
        "dataset": "ai4bharat/IndicWikisumm",
        "metric": "rouge",
        "prompt_fn": summarization_prompt,
        "description": "News/wiki summarization with ROUGE-L",
    },
    "nli": {
        "dataset": "Divyanshu/indicxnli",
        "metric": "accuracy",
        "prompt_fn": nli_prompt,
        "description": "Natural language inference — entailment/contradiction/neutral",
    },
    "sentiment": {
        "dataset": "ai4bharat/IndicSentiment",
        "metric": "f1",
        "prompt_fn": sentiment_prompt,
        "description": "3-class sentiment classification",
    },
    "translation": {
        "dataset": "ai4bharat/IN22-Gen",
        "metric": "bleu",
        "prompt_fn": translation_prompt,
        "description": "English ↔ Indic translation (BLEU + chrF)",
    },
}
