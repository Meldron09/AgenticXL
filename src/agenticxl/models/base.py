"""Backend-agnostic LLM client interface.

Every pipeline stage (filtering, enrichment) and every deepagents
agent/subagent goes through an `LLMClient` obtained from `factory.get_llm_client`,
so the single `AGENTICXL_MODEL_BACKEND` env var determines the model used
everywhere in a run -- no stage hardcodes a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel


class StructuredOutputError(RuntimeError):
    """Raised when a backend cannot produce parseable structured output
    after retries. Callers should treat this as a per-batch failure to
    isolate, not a reason to abort an entire run."""


class LLMClient(ABC):
    """Wraps a single LangChain BaseChatModel so it can be used both as the
    `model=` argument to deepagents' `create_deep_agent`/subagent definitions,
    and directly for structured completions inside custom tools."""

    backend_name: str

    @property
    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        """The underlying LangChain chat model, suitable for passing as the
        `model=` argument to deepagents' create_deep_agent / SubAgent."""

    def complete(self, prompt: str) -> str:
        result = self.chat_model.invoke(prompt)
        content = getattr(result, "content", result)
        return content if isinstance(content, str) else str(content)

    def complete_structured(self, prompt: str, schema: dict, max_retries: int = 2) -> dict:
        """Get a structured (JSON-schema-validated) response. Retries with
        the parse error fed back into the prompt -- local models (Ollama)
        are more likely to need this than Claude, but the retry logic is
        shared so both backends benefit."""
        structured = self.chat_model.with_structured_output(schema, include_raw=True)
        last_error = None
        current_prompt = prompt
        for _ in range(max_retries + 1):
            result = structured.invoke(current_prompt)
            parsing_error = result.get("parsing_error")
            parsed = result.get("parsed")
            if parsing_error is None and parsed is not None:
                return parsed
            last_error = parsing_error
            current_prompt = (
                f"{prompt}\n\n"
                "Your previous response could not be parsed as valid JSON matching "
                f"the required schema. Error: {last_error}. "
                "Respond again with ONLY valid JSON matching the schema, no other text."
            )
        raise StructuredOutputError(
            f"{self.backend_name}: failed to get structured output after "
            f"{max_retries + 1} attempts: {last_error}"
        )
