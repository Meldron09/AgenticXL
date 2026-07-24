from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from agenticxl.config import ModelConfig

from .base import LLMClient


class ClaudeClient(LLMClient):
    backend_name = "claude"

    def __init__(self, config: ModelConfig):
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the claude backend")
        self._chat_model = ChatAnthropic(
            model=config.claude_model,
            api_key=config.anthropic_api_key,
        )

    @property
    def chat_model(self) -> BaseChatModel:
        return self._chat_model
