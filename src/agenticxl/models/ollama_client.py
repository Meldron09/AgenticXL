from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from agenticxl.config import ModelConfig

from .base import LLMClient


class OllamaClient(LLMClient):
    backend_name = "ollama"

    def __init__(self, config: ModelConfig):
        if not config.ollama_model:
            raise ValueError("OLLAMA_MODEL is required for the ollama backend")
        self._chat_model = ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        )

    @property
    def chat_model(self) -> BaseChatModel:
        return self._chat_model
