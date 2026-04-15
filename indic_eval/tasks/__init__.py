from indic_eval.tasks.registry import TASK_REGISTRY
from indic_eval.tasks.prompts import qa_prompt, summarization_prompt, nli_prompt, sentiment_prompt, translation_prompt

__all__ = ["TASK_REGISTRY", "qa_prompt", "summarization_prompt",
           "nli_prompt", "sentiment_prompt", "translation_prompt"]
