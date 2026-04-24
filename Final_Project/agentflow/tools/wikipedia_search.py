"""Wikipedia Search Tool -- retrieves article summaries via the wikipedia package."""

from __future__ import annotations

from typing import Any

from agentflow.tools.base import BaseTool


class WikipediaSearchTool(BaseTool):
    def __init__(self, model_string: str = "", *, api_key: str | None = None) -> None:
        super().__init__(
            tool_name="Wikipedia_Search_Tool",
            tool_description=(
                "Searches Wikipedia for a topic and returns the article summary. "
                "Useful for factual, encyclopaedic knowledge."
            ),
            input_types={
                "query": "str or list[str] - One or more search terms.",
            },
            output_type="str - Article summaries.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Photosynthesis")',
                    "description": "Retrieve the Wikipedia summary for Photosynthesis.",
                }
            ],
            user_metadata={
                "limitation": "Only returns summaries; may not cover very recent events.",
                "best_practice": "Use precise topic names for best results.",
            },
            model_string=model_string,
            api_key=api_key,
        )

    def execute(self, query: str | list[str], *, sentences: int = 5, **kwargs: Any) -> str:
        try:
            import wikipedia
        except ImportError:
            return "Error: wikipedia package not installed."

        queries = [query] if isinstance(query, str) else query
        parts: list[str] = []

        for q in queries:
            try:
                page = wikipedia.summary(q, sentences=sentences, auto_suggest=True)
                parts.append(f"## {q}\n{page}")
            except wikipedia.DisambiguationError as e:
                options = ", ".join(e.options[:5])
                parts.append(f"## {q}\nAmbiguous. Options: {options}")
            except wikipedia.PageError:
                parts.append(f"## {q}\nNo Wikipedia page found.")
            except Exception as exc:
                parts.append(f"## {q}\nError: {exc}")

        return "\n\n".join(parts)
