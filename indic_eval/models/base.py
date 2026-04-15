"""Base class for all model backends."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response for the given prompt."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"
