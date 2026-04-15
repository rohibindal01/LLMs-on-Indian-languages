"""HuggingFace Inference API backend (free tier)."""

import os
import requests
from indic_eval.models.base import BaseModel

HF_API_URL = "https://api-inference.huggingface.co/models"


class HuggingFaceModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.api_key = os.getenv("HF_API_KEY", "")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        url = f"{HF_API_URL}/{self.model_id}"
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "return_full_text": False},
        }
        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            return ""
        except Exception as e:
            print(f"HF API error: {e}")
            return ""
