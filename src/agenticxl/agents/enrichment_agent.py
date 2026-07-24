from __future__ import annotations

from deepagents import SubAgent
from langchain.agents.structured_output import ToolStrategy

from agenticxl.config import RunConfig
from agenticxl.enrichment.web_search import SerperClient
from agenticxl.models import LLMClient

from .schemas import EnrichmentSummary
from .tools import make_enrich_columns_tool

ENRICHMENT_SUBAGENT_NAME = "enrichment_agent"


def build_enrichment_subagent(
    run_config: RunConfig,
    llm_client: LLMClient,
    serper_client: SerperClient | None,
) -> SubAgent:
    enrich_columns_tool = make_enrich_columns_tool(run_config, llm_client, serper_client)
    return SubAgent(
        name=ENRICHMENT_SUBAGENT_NAME,
        description=(
            "Populates the configured enrichment columns on the consolidated xlsx "
            "file, deriving values from existing row data first and falling back to "
            "Serper web search only when necessary, then writes the final enriched "
            "workbook. Invoke once with the consolidated file's path."
        ),
        system_prompt=(
            "You are given the path to the consolidated xlsx file. Call the "
            "`enrich_columns` tool exactly once with that path -- it performs the "
            "local-derivation-first, web-search-fallback enrichment and writes the "
            "final output itself; do not re-implement any of that logic yourself. "
            "Your final structured response must copy the tool's returned fields "
            "exactly, without altering any values."
        ),
        tools=[enrich_columns_tool],
        model=llm_client.chat_model,
        # See file_subagent.py: force tool-calling-based structured output
        # uniformly across backends rather than a provider-native mode that
        # not every backend supports.
        response_format=ToolStrategy(EnrichmentSummary),
    )
