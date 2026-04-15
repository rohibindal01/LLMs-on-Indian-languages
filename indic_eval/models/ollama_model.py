"""Ollama backend — run models locally for free."""

from indic_eval.models.base import BaseModel


class OllamaModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        try:
            import ollama
            self.client = ollama
        except ImportError:
            raise ImportError("Run: pip install ollama  (and install Ollama from https://ollama.ai)")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            response = self.client.chat(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens, "temperature": 0},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""
