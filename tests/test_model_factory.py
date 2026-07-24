from __future__ import annotations

import pytest

from agenticxl.config import ModelConfig
from agenticxl.models.claude_client import ClaudeClient
from agenticxl.models.deepseek_client import DeepSeekClient
from agenticxl.models.factory import get_llm_client
from agenticxl.models.ollama_client import OllamaClient


def test_backend_defaults_to_claude_when_unset():
    config = ModelConfig.from_env(env={})
    assert config.backend == "claude"


def test_backend_env_var_claude_explicit():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "claude", "ANTHROPIC_API_KEY": "sk-x"})
    client = get_llm_client(config)
    assert isinstance(client, ClaudeClient)
    assert client.backend_name == "claude"


def test_backend_env_var_ollama():
    config = ModelConfig.from_env(
        env={"AGENTICXL_MODEL_BACKEND": "ollama", "OLLAMA_MODEL": "llama3.1"}
    )
    client = get_llm_client(config)
    assert isinstance(client, OllamaClient)
    assert client.backend_name == "ollama"


def test_backend_env_var_deepseek():
    config = ModelConfig.from_env(
        env={"AGENTICXL_MODEL_BACKEND": "deepseek", "DEEPSEEK_API_KEY": "sk-x"}
    )
    client = get_llm_client(config)
    assert isinstance(client, DeepSeekClient)
    assert client.backend_name == "deepseek"


def test_deepseek_backend_without_api_key_fails_fast():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "deepseek"})
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_llm_client(config)


def test_deepseek_model_defaults():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "deepseek", "DEEPSEEK_API_KEY": "sk-x"})
    assert config.deepseek_model == "deepseek-v4-pro"


def test_deepseek_model_override():
    config = ModelConfig.from_env(
        env={
            "AGENTICXL_MODEL_BACKEND": "deepseek",
            "DEEPSEEK_API_KEY": "sk-x",
            "AGENTICXL_DEEPSEEK_MODEL": "deepseek-v4-flash",
        }
    )
    assert config.deepseek_model == "deepseek-v4-flash"


def test_backend_env_var_deepseek_case_insensitive():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "DeepSeek", "DEEPSEEK_API_KEY": "sk-x"})
    assert config.backend == "deepseek"


def test_backend_env_var_unrecognized_falls_back_to_claude():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "not-a-real-backend"})
    assert config.backend == "claude"


def test_backend_env_var_case_insensitive():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "OLLAMA", "OLLAMA_MODEL": "llama3.1"})
    assert config.backend == "ollama"


def test_claude_backend_without_api_key_fails_fast():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "claude"})
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_llm_client(config)


def test_ollama_backend_without_model_fails_fast():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "ollama"})
    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        get_llm_client(config)


def test_ollama_base_url_defaults():
    config = ModelConfig.from_env(env={"AGENTICXL_MODEL_BACKEND": "ollama", "OLLAMA_MODEL": "llama3.1"})
    assert config.ollama_base_url == "http://localhost:11434"


def test_ollama_base_url_override():
    config = ModelConfig.from_env(
        env={
            "AGENTICXL_MODEL_BACKEND": "ollama",
            "OLLAMA_MODEL": "llama3.1",
            "OLLAMA_BASE_URL": "http://custom-host:1234",
        }
    )
    assert config.ollama_base_url == "http://custom-host:1234"
