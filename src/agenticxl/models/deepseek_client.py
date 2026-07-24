from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from agenticxl.config import ModelConfig

from .base import LLMClient


class DeepSeekClient(LLMClient):
    backend_name = "deepseek"

    def __init__(self, config: ModelConfig):
        if not config.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the deepseek backend")
        self._chat_model = ChatDeepSeek(
            model=config.deepseek_model,
            api_key=config.deepseek_api_key,
            # DeepSeek's v4 models default to "thinking mode" on, which the API
            # rejects when combined with the forced tool_choice that
            # with_structured_output relies on ("Thinking mode does not support
            # this tool_choice"). Every call this pipeline makes goes through
            # complete_structured, so thinking mode must always be off here --
            # not a user-configurable option.
            extra_body={"thinking": {"type": "disabled"}},
        )

    @property
    def chat_model(self) -> BaseChatModel:
        return self._chat_model
