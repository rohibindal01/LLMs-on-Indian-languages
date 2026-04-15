"""Prompt templates for each eval task."""

LANG_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
    "bn": "Bengali", "te": "Telugu", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
}

NLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


def qa_prompt(sample: dict, lang: str) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    return (
        f"You are an expert in {lang_name}. Answer the following question based on the context.\n"
        f"Context: {sample['context']}\n"
        f"Question: {sample['question']}\n"
        f"Answer (in {lang_name}, concise):"
    )


def summarization_prompt(sample: dict, lang: str) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    return (
        f"Summarize the following {lang_name} article in 2-3 sentences.\n\n"
        f"Article: {sample['document']}\n\n"
        f"Summary:"
    )


def nli_prompt(sample: dict, lang: str) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    return (
        f"Given the following premise and hypothesis in {lang_name}, "
        f"classify their relationship as one of: entailment, neutral, contradiction.\n"
        f"Premise: {sample['premise']}\n"
        f"Hypothesis: {sample['hypothesis']}\n"
        f"Relationship (one word):"
    )


def sentiment_prompt(sample: dict, lang: str) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    return (
        f"Classify the sentiment of the following {lang_name} text as: positive, negative, or neutral.\n"
        f"Text: {sample['text']}\n"
        f"Sentiment (one word):"
    )


def translation_prompt(sample: dict, lang: str) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    return (
        f"Translate the following English sentence to {lang_name}.\n"
        f"English: {sample['source']}\n"
        f"Translation:"
    )
