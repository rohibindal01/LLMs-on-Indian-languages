"""Google Gemini API backend — free tier via AI Studio."""

import os
from indic_eval.models.base import BaseModel


class GeminiModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel(model_id)
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")
        except KeyError:
            raise EnvironmentError("Set GEMINI_API_KEY in your .env file. Get one free at https://aistudio.google.com")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.0},
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""
