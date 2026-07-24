from __future__ import annotations

from pathlib import Path

from deepagents import SubAgent
from langchain.agents.structured_output import ToolStrategy

from agenticxl.config import RunConfig
from agenticxl.models import LLMClient

from .schemas import FileResult
from .tools import make_process_file_tool

FILE_SUBAGENT_NAME = "file_filter_agent"


def build_file_subagent(run_config: RunConfig, llm_client: LLMClient, run_logs_dir: Path) -> SubAgent:
    process_file_tool = make_process_file_tool(run_config, llm_client, run_logs_dir)
    return SubAgent(
        name=FILE_SUBAGENT_NAME,
        description=(
            "Filters a single raw Excel export file down to rows whose PRODUCT DETAILS "
            "text is semantically relevant to the configured keywords, writing the "
            "result to a filtered xlsx file. Invoke once per input file, passing its "
            "file path."
        ),
        system_prompt=(
            "You are given the path to exactly one raw Excel export file. Call the "
            "`process_file` tool exactly once with that path -- it performs header "
            "detection, per-row semantic relevance judgment against the configured "
            "keywords, and writes the filtered output itself; do not re-implement any "
            "of that logic yourself and do not call the tool more than once. Your "
            "final structured response must copy the tool's returned fields exactly, "
            "without altering any values."
        ),
        tools=[process_file_tool],
        model=llm_client.chat_model,
        # Force tool-calling-based structured output rather than letting
        # AutoStrategy pick a provider-native response_format mode -- some
        # backends (e.g. DeepSeek) reject that mode ("This response_format
        # type is unavailable now"), while tool-calling is already confirmed
        # to work uniformly across every supported backend.
        response_format=ToolStrategy(FileResult),
    )
