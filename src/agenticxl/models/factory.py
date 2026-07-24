from __future__ import annotations

from agenticxl.config import ModelConfig

from .base import LLMClient


def get_llm_client(model_config: ModelConfig) -> LLMClient:
    """Single entry point for backend selection. Reads the already-resolved
    ModelConfig (itself built from AGENTICXL_MODEL_BACKEND) and fails fast if
    the selected backend is missing required configuration, rather than
    silently falling back to a different backend."""
    model_config.validate_backend_ready()

    if model_config.backend == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(model_config)

    if model_config.backend == "deepseek":
        from .deepseek_client import DeepSeekClient

        return DeepSeekClient(model_config)

    from .claude_client import ClaudeClient

    return ClaudeClient(model_config)
