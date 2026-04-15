"""Groq API backend — free tier LLaMA 3 70B inference."""

import os
from indic_eval.models.base import BaseModel


class GroqModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        try:
            from groq import Groq
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        except ImportError:
            raise ImportError("Run: pip install groq")
        except KeyError:
            raise EnvironmentError("Set GROQ_API_KEY in your .env file. Get one free at https://console.groq.com")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API error: {e}")
            return ""
