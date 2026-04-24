"""Executor module -- generates tool commands and executes them."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from typing import Any

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import ToolCommand


class Executor:
    def __init__(
        self,
        llm_engine_name: str,
        tool_instances: dict[str, Any] | None = None,
        root_cache_dir: str = "solver_cache",
        max_time: int = 120,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.llm_engine_name = llm_engine_name
        self.tool_instances = tool_instances or {}
        self.root_cache_dir = root_cache_dir
        self.query_cache_dir = root_cache_dir
        self.max_time = max_time

        self.llm = create_llm_engine(
            llm_engine_name, api_key=api_key, temperature=temperature,
        )

    def set_query_cache_dir(self, path: str) -> None:
        self.query_cache_dir = path
        os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------
    def generate_tool_command(
        self,
        question: str,
        context: str,
        sub_goal: str,
        tool_name: str,
        tool_metadata: dict[str, Any],
        step_count: int = 0,
        json_data: dict | None = None,
    ) -> Any:
        prompt = f"""Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1. Analyze the tool's required parameters from its metadata.
2. Construct valid Python code that addresses the sub-goal using the provided context and data.
3. The command must include at least one call to `tool.execute()`.
4. Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5. Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(query="Summarize the following problem: Isaac has 100 toys...")
```

Example2:
Generated Command:
```python
execution = tool.execute(query=["Methanol", "function of hyperbola"])
```"""

        result = self.llm(prompt, response_format=ToolCommand)
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt
            json_data[f"tool_commander_{step_count}_response"] = str(result)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def extract_explanation_and_command(response: Any) -> tuple[str, str, str]:
        analysis = explanation = "N/A"
        command = "No command found."

        def _normalise(code: str) -> str:
            return re.sub(r"^```python\s*", "", code).rstrip("```").strip()

        if isinstance(response, str):
            try:
                d = json.loads(response)
                response = ToolCommand(**d)
            except Exception:
                pass

        if isinstance(response, ToolCommand):
            return (
                response.analysis.strip(),
                response.explanation.strip(),
                _normalise(response.command.strip()),
            )

        text = str(response)
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            command = m.group(1).strip()
        return analysis, explanation, _normalise(command)

    # ------------------------------------------------------------------
    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        tool = self.tool_instances.get(tool_name)
        if tool is None:
            return f"Error: tool '{tool_name}' not found in instances."

        local_ctx: dict[str, Any] = {"tool": tool}
        result_box: dict[str, Any] = {"result": None, "exc": None, "done": False}

        def _run() -> None:
            try:
                exec(command, {}, local_ctx)  # noqa: S102
                result_box["result"] = local_ctx.get("execution")
                result_box["done"] = True
            except Exception as exc:
                result_box["exc"] = exc
                result_box["done"] = True

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self.max_time)

        if not result_box["done"]:
            return f"Execution timed out after {self.max_time}s"
        if result_box["exc"]:
            return f"Execution error: {result_box['exc']}"
        return result_box["result"]
