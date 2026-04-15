"""Model factory — returns the right backend based on model name."""

from indic_eval.models.hf_model import HuggingFaceModel
from indic_eval.models.groq_model import GroqModel
from indic_eval.models.gemini_model import GeminiModel
from indic_eval.models.ollama_model import OllamaModel


MODEL_MAP = {
    "mistral-7b":    ("hf",     "mistralai/Mistral-7B-Instruct-v0.2"),
    "gemma-2b":      ("hf",     "google/gemma-2b-it"),
    "llama3-8b":     ("ollama", "llama3"),
    "llama3-70b":    ("groq",   "llama3-70b-8192"),
    "gemini-flash":  ("gemini", "gemini-1.5-flash"),
    "gemini-pro":    ("gemini", "gemini-1.5-pro"),
}


def get_model(model_name: str):
    if model_name in MODEL_MAP:
        backend, model_id = MODEL_MAP[model_name]
    elif "/" in model_name:
        backend, model_id = "hf", model_name
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Known shortcuts: {list(MODEL_MAP.keys())}\n"
            f"Or pass a full HuggingFace model ID like 'mistralai/Mistral-7B-Instruct-v0.2'."
        )

    if backend == "hf":
        return HuggingFaceModel(model_id)
    elif backend == "groq":
        return GroqModel(model_id)
    elif backend == "gemini":
        return GeminiModel(model_id)
    elif backend == "ollama":
        return OllamaModel(model_id)
    else:
        raise ValueError(f"Unknown backend: {backend}")
