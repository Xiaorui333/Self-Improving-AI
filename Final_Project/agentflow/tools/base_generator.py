"""Base Generator Tool -- calls the LLM directly for reasoning-only answers."""

from __future__ import annotations

from typing import Any

from agentflow.engine.factory import create_llm_engine
from agentflow.tools.base import BaseTool


class BaseGeneratorTool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string: str = "", *, api_key: str | None = None) -> None:
        super().__init__(
            tool_name="Base_Generator_Tool",
            tool_description=(
                "A generalized tool that takes a query from the user and answers "
                "the question step by step to the best of its ability."
            ),
            input_types={
                "query": "str - The user query to answer.",
            },
            output_type="str - The generated response.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Summarize the following text")',
                    "description": "Generate a short summary.",
                }
            ],
            user_metadata={
                "limitation": "May provide hallucinated or incorrect responses.",
                "best_practice": "Use for general queries; verify important information.",
            },
            model_string=model_string,
            api_key=api_key,
        )
        if model_string and model_string != "dummy":
            self.llm = create_llm_engine(model_string, api_key=api_key, temperature=0.0)
        else:
            self.llm = None

    def execute(self, query: str, **kwargs: Any) -> str:
        if self.llm is None:
            return "Error: LLM engine not initialised."
        try:
            return self.llm(query)
        except Exception as exc:
            return f"Error generating response: {exc}"
