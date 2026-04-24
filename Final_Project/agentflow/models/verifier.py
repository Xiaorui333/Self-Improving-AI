"""Verifier module -- evaluates execution results and decides STOP / CONTINUE."""

from __future__ import annotations

import json
import re
from typing import Any, Tuple

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import MemoryVerification
from agentflow.models.memory import Memory


class Verifier:
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

        self.llm_engine_fixed = create_llm_engine(
            self.llm_engine_fixed_name, api_key=api_key, temperature=temperature,
        )

    # ------------------------------------------------------------------
    def verificate_context(
        self,
        question: str,
        query_analysis: str,
        memory: Memory,
        step_count: int = 0,
        json_data: dict | None = None,
    ) -> Any:
        prompt = f"""Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Initial Analysis:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}

Instructions:
1. Review the query, initial analysis, and memory.
2. Assess the completeness of the memory: Does it fully address all parts of the query?
3. Check for potential issues:
   - Are there any inconsistencies or contradictions?
   - Is any information ambiguous or in need of verification?
4. Determine if any unused tools could provide missing information.

Final Determination:
- If the memory is sufficient, explain why and conclude with "STOP".
- If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE"."""

        result = self.llm_engine_fixed(prompt, response_format=MemoryVerification)

        if json_data is not None:
            json_data[f"verifier_{step_count}_prompt"] = prompt
            json_data[f"verifier_{step_count}_response"] = str(result)

        return result

    # ------------------------------------------------------------------
    def extract_conclusion(self, response: Any) -> Tuple[str, str]:
        if isinstance(response, str):
            try:
                d = json.loads(response)
                response = MemoryVerification(**d)
            except Exception:
                pass

        if isinstance(response, MemoryVerification):
            return response.analysis, "STOP" if response.stop_signal else "CONTINUE"

        text = str(response)
        pattern = r"conclusion\**:?\s*\**\s*(\w+)"
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
        if matches:
            conclusion = matches[-1].group(1).upper()
            if conclusion in ("STOP", "CONTINUE"):
                return text, conclusion

        if "stop" in text.lower():
            return text, "STOP"
        if "continue" in text.lower():
            return text, "CONTINUE"

        return text, "CONTINUE"
