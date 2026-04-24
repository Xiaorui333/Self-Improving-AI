"""Web Search Tool -- uses DuckDuckGo (no API key needed)."""

from __future__ import annotations

from typing import Any

from agentflow.tools.base import BaseTool


class WebSearchTool(BaseTool):
    def __init__(self, model_string: str = "", *, api_key: str | None = None) -> None:
        super().__init__(
            tool_name="Web_Search_Tool",
            tool_description=(
                "Searches the web using DuckDuckGo and returns the top results. "
                "Useful for finding up-to-date information, facts, and references."
            ),
            input_types={
                "query": "str - The search query.",
            },
            output_type="str - Concatenated search results with titles, URLs, and snippets.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="latest news on climate change")',
                    "description": "Search the web for recent information.",
                }
            ],
            user_metadata={
                "limitation": "Results may not always be relevant; rate-limited.",
                "best_practice": "Use specific, targeted queries for best results.",
            },
            model_string=model_string,
            api_key=api_key,
        )

    def execute(self, query: str, *, max_results: int = 5, **kwargs: Any) -> str:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return "No results found."

            parts: list[str] = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                href = r.get("href", "")
                body = r.get("body", "")
                parts.append(f"[{i}] {title}\n    URL: {href}\n    {body}")
            return "\n\n".join(parts)
        except ImportError:
            return "Error: duckduckgo-search package not installed."
        except Exception as exc:
            return f"Error during web search: {exc}"
