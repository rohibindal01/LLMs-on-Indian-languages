"""Dataset loader — pulls Indic datasets from HuggingFace."""

from datasets import load_dataset


DATASET_CONFIGS = {
    "ai4bharat/IndicQA": {
        "subset_key": "language",
        "split": "test",
        "fields": {"context": "context", "question": "question", "answer": "answers.text[0]"},
    },
    "ai4bharat/IndicWikisumm": {
        "subset_key": "lang",
        "split": "test",
        "fields": {"document": "article", "answer": "summary"},
    },
    "Divyanshu/indicxnli": {
        "subset_key": "language",
        "split": "test",
        "fields": {"premise": "premise", "hypothesis": "hypothesis", "answer": "label"},
    },
    "ai4bharat/IndicSentiment": {
        "subset_key": "language",
        "split": "test",
        "fields": {"text": "INDIC REVIEW", "answer": "LABEL"},
    },
    "ai4bharat/IN22-Gen": {
        "subset_key": None,
        "split": "gen",
        "fields": {"source": "sentence_en", "answer": "sentence_hi"},
    },
}

LANG_TO_HF = {
    "hi": "hi", "mr": "mr", "ta": "ta", "bn": "bn", "te": "te",
    "gu": "gu", "kn": "kn", "ml": "ml", "pa": "pa", "ur": "ur",
}


class DatasetLoader:
    def __init__(self, dataset_id: str, lang: str, num_samples: int = 100):
        self.dataset_id = dataset_id
        self.lang = LANG_TO_HF.get(lang, lang)
        self.num_samples = num_samples
        self.cfg = DATASET_CONFIGS.get(dataset_id, {})

    def load(self) -> list[dict]:
        try:
            subset = self.lang if self.cfg.get("subset_key") else None
            split = self.cfg.get("split", "test")
            ds = load_dataset(self.dataset_id, subset, split=split, trust_remote_code=True)
            ds = ds.select(range(min(self.num_samples, len(ds))))
            return [self._extract_fields(row) for row in ds]
        except Exception as e:
            print(f"   ⚠️  Could not load {self.dataset_id} ({self.lang}): {e}")
            return []

    def _extract_fields(self, row: dict) -> dict:
        fields = self.cfg.get("fields", {})
        result = {}
        for target_key, source_key in fields.items():
            # Support nested keys like "answers.text[0]"
            try:
                val = row
                for part in source_key.replace("[0]", ".0").split("."):
                    val = val[int(part)] if part.isdigit() else val[part]
                result[target_key] = val
            except (KeyError, IndexError, TypeError):
                result[target_key] = ""
        return result
