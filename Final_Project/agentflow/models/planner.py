"""Planner module -- generates query analysis, next-step actions, and final output.

Prompt templates are adapted from the official AgentFlow repository
(lupantech/AgentFlow) to preserve the original system's behaviour.
"""

from __future__ import annotations

import json
import re
from typing import Any, Tuple

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import NextStep, QueryAnalysis
from agentflow.models.memory import Memory


class Planner:
    def __init__(
        self,
        llm_engine_name: str,
        llm_engine_fixed_name: str | None = None,
        toolbox_metadata: dict | None = None,
        available_tools: list[str] | None = None,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name or llm_engine_name
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.query_analysis: str = ""

        self.llm_engine = create_llm_engine(
            llm_engine_name, api_key=api_key, temperature=temperature,
        )
        self.llm_engine_fixed = create_llm_engine(
            self.llm_engine_fixed_name, api_key=api_key, temperature=temperature,
        )

    # ------------------------------------------------------------------
    def generate_base_response(self, question: str, max_tokens: int = 2048) -> str:
        return self.llm_engine(question, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    def analyze_query(self, question: str) -> str:
        prompt = f"""Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be brief and precise with insight."""

        self.query_analysis = str(
            self.llm_engine_fixed(prompt, response_format=QueryAnalysis)
        ).strip()
        return self.query_analysis

    # ------------------------------------------------------------------
    def generate_next_step(
        self,
        question: str,
        query_analysis: str,
        memory: Memory,
        step_count: int,
        max_step_count: int,
        json_data: dict | None = None,
    ) -> Any:
        prompt = f"""Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1. **Justification:** Explain your choice of tool and sub-goal.
2. **Context:** Provide all necessary information for the tool.
3. **Sub-Goal:** State the specific objective for the tool.
4. **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content."""

        next_step = self.llm_engine(prompt, response_format=NextStep)

        if json_data is not None:
            json_data[f"action_predictor_{step_count}_prompt"] = prompt
            json_data[f"action_predictor_{step_count}_response"] = str(next_step)

        return next_step

    # ------------------------------------------------------------------
    def extract_context_subgoal_and_tool(
        self, response: Any,
    ) -> Tuple[str, str, str]:
        def _normalize(tool_name: str) -> str:
            canonical = re.sub(r"[ _]+", "_", tool_name).lower()
            for tool in self.available_tools:
                if re.sub(r"[ _]+", "_", tool).lower() == canonical:
                    return tool
            return tool_name

        try:
            if isinstance(response, str):
                try:
                    d = json.loads(response)
                    response = NextStep(**d)
                except Exception:
                    pass

            if isinstance(response, NextStep):
                return (
                    response.context.strip(),
                    response.sub_goal.strip(),
                    _normalize(response.tool_name.strip()),
                )

            text = str(response).replace("**", "")
            pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?:\n\n|\Z)"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                ctx, sg, tn = matches[-1]
                return ctx.strip(), sg.strip(), _normalize(tn.strip())
        except Exception as exc:
            print(f"Error extracting context/sub-goal/tool: {exc}")

        return "", "", ""

    # ------------------------------------------------------------------
    def generate_direct_output(self, question: str, memory: Memory) -> str:
        prompt = f"""Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {self.query_analysis}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1. **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2. **Answer:** A direct and concise final answer to the query."""

        return self.llm_engine_fixed(prompt)
