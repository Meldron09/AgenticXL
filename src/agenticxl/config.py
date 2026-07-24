"""Run configuration: loads config/run_config.yaml plus environment variables
into a single explicit RunConfig object. No hidden global state -- callers
build one RunConfig and pass it through the pipeline, which keeps tests able
to construct a config directly without touching real env/files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ModelBackend = Literal["claude", "ollama", "deepseek"]

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _parse_bool_env(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY_ENV_VALUES


class ModelConfig(BaseModel):
    """Resolved model-backend selection, read once from the environment."""

    backend: ModelBackend = "claude"
    anthropic_api_key: str | None = None
    claude_model: str = DEFAULT_CLAUDE_MODEL
    ollama_model: str | None = None
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    deepseek_api_key: str | None = None
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ModelConfig":
        env = env if env is not None else os.environ
        raw_backend = env.get("AGENTICXL_MODEL_BACKEND", "claude").strip().lower()
        backend: ModelBackend
        if raw_backend == "ollama":
            backend = "ollama"
        elif raw_backend == "deepseek":
            backend = "deepseek"
        else:
            backend = "claude"
        return cls(
            backend=backend,
            anthropic_api_key=env.get("ANTHROPIC_API_KEY"),
            claude_model=env.get("AGENTICXL_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            ollama_model=env.get("OLLAMA_MODEL"),
            ollama_base_url=env.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            deepseek_api_key=env.get("DEEPSEEK_API_KEY"),
            deepseek_model=env.get("AGENTICXL_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        )

    def validate_backend_ready(self) -> None:
        """Fail fast with a clear error if the selected backend is missing
        required configuration, rather than silently falling back."""
        if self.backend == "claude" and not self.anthropic_api_key:
            raise ValueError(
                "AGENTICXL_MODEL_BACKEND=claude requires ANTHROPIC_API_KEY to be set."
            )
        if self.backend == "ollama" and not self.ollama_model:
            raise ValueError(
                "AGENTICXL_MODEL_BACKEND=ollama requires OLLAMA_MODEL to be set."
            )
        if self.backend == "deepseek" and not self.deepseek_api_key:
            raise ValueError(
                "AGENTICXL_MODEL_BACKEND=deepseek requires DEEPSEEK_API_KEY to be set."
            )


class RunConfig(BaseModel):
    """Full pipeline run configuration: keywords (l), enrichment columns (c),
    batch sizes, paths, model backend, and Serper key."""

    project_root: Path
    input_dir: Path
    keywords: list[str] = Field(default_factory=list)
    enrichment_columns: list[str] = Field(default_factory=list)
    relevance_batch_size: int = 50
    relevance_concurrency: int = 10
    enrichment_batch_size: int = 25
    filtered_dir: Path = Path("data/filtered")
    consolidated_dir: Path = Path("data/consolidated")
    output_dir: Path = Path("data/output")
    logs_dir: Path = Path("logs")
    model: ModelConfig = Field(default_factory=ModelConfig)
    serper_api_key: str | None = None
    skip_enrichment: bool = False

    @field_validator("keywords", "enrichment_columns")
    @classmethod
    def _non_empty_strings(cls, v: list[str]) -> list[str]:
        return [item.strip() for item in v if item and item.strip()]

    def resolve(self, path: Path) -> Path:
        """Resolve a (possibly relative) path against the project root."""
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    @property
    def input_dir_resolved(self) -> Path:
        return self.resolve(self.input_dir)

    @property
    def filtered_dir_resolved(self) -> Path:
        return self.resolve(self.filtered_dir)

    @property
    def consolidated_dir_resolved(self) -> Path:
        return self.resolve(self.consolidated_dir)

    @property
    def output_dir_resolved(self) -> Path:
        return self.resolve(self.output_dir)

    @property
    def logs_dir_resolved(self) -> Path:
        return self.resolve(self.logs_dir)

    def ensure_directories(self) -> None:
        for d in (
            self.filtered_dir_resolved,
            self.consolidated_dir_resolved,
            self.output_dir_resolved,
            self.logs_dir_resolved,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        project_root: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> "RunConfig":
        """Load a RunConfig from a YAML file plus environment variables.

        `project_root` defaults to the config file's parent's parent (i.e.
        the project root when config lives at <root>/config/run_config.yaml),
        overridable for tests.
        """
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        root = Path(project_root) if project_root is not None else config_path.parent.parent
        env = env if env is not None else os.environ

        return cls(
            project_root=root,
            input_dir=raw.get("input_dir", "Documents"),
            keywords=raw.get("keywords", []),
            enrichment_columns=raw.get("enrichment_columns", []),
            relevance_batch_size=raw.get("relevance_batch_size", 50),
            relevance_concurrency=raw.get("relevance_concurrency", 10),
            enrichment_batch_size=raw.get("enrichment_batch_size", 25),
            filtered_dir=raw.get("filtered_dir", "data/filtered"),
            consolidated_dir=raw.get("consolidated_dir", "data/consolidated"),
            output_dir=raw.get("output_dir", "data/output"),
            logs_dir=raw.get("logs_dir", "logs"),
            model=ModelConfig.from_env(env),
            serper_api_key=env.get("SERPER_API_KEY"),
            skip_enrichment=_parse_bool_env(env.get("AGENTICXL_SKIP_ENRICHMENT")),
        )
