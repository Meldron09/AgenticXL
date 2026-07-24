"""The main orchestrator agent: a thin deepagents wrapper whose only job per
invocation is to call the `task` tool for the one subagent/file named in its
prompt. Fan-out across files and result aggregation is deliberately kept out
of the orchestrator's own LLM-driven decision loop -- it's deterministic
Python control flow in `pipeline.py` (one orchestrator invocation per file,
run concurrently), so a single file's outcome never depends on an LLM
correctly deciding to loop over N files itself. See the implementation plan
for the reasoning behind this split.
"""

from __future__ import annotations

import json
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.messages import BaseMessage, ToolMessage

from agenticxl.models import LLMClient

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are a data-pipeline orchestrator. You will be given a task naming exactly one "
    "input file and a subagent to use for it. Call the `task` tool to invoke that "
    "subagent with the file path, then report its result back as your final answer. "
    "Do not attempt to discover or process any other files yourself -- the calling "
    "system handles fan-out across files by invoking you separately, once per file."
)


def build_orchestrator(llm_client: LLMClient, subagents: list[SubAgent], system_prompt: str = ORCHESTRATOR_SYSTEM_PROMPT):
    return create_deep_agent(
        model=llm_client.chat_model,
        subagents=subagents,
        system_prompt=system_prompt,
    )


def extract_last_tool_result(messages: list[BaseMessage]) -> dict[str, Any] | None:
    """Scan messages in reverse for the last ToolMessage (the orchestrator's
    `task` tool call result) and parse its JSON content.

    Subagents given a `response_format` (see agents/schemas.py) have their
    structured final response JSON-serialized into this ToolMessage's
    content by deepagents -- parsing it directly is far more reliable than
    trying to extract data from the orchestrator's own free-text summary.
    """
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if not isinstance(content, str):
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            continue
    return None
