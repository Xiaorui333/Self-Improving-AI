"""Solver -- multi-turn orchestration loop for AgentFlow.

Coordinates Planner -> Executor -> Verifier -> Memory across turns until
the Verifier signals STOP or the step budget is exhausted, then produces
a final direct answer via the Planner/Generator.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentflow.config import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    PORTKEY_API_KEY,
)
from agentflow.models.executor import Executor
from agentflow.models.memory import Memory
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.tools import TOOL_NAMES, build_tools, get_toolbox_metadata


class Solver:
    """End-to-end AgentFlow solver."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str = PORTKEY_API_KEY,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_time: int = 300,
        temperature: float = DEFAULT_TEMPERATURE,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_steps = max_steps
        self.max_time = max_time
        self.verbose = verbose

        tool_instances = build_tools(model, api_key=api_key)
        toolbox_meta = {n: t.get_metadata() for n, t in tool_instances.items()}
        tool_names = list(tool_instances.keys())

        self.planner = Planner(
            llm_engine_name=model,
            llm_engine_fixed_name=model,
            toolbox_metadata=toolbox_meta,
            available_tools=tool_names,
            api_key=api_key,
            temperature=temperature,
        )
        self.verifier = Verifier(
            llm_engine_name=model,
            llm_engine_fixed_name=model,
            toolbox_metadata=toolbox_meta,
            available_tools=tool_names,
            api_key=api_key,
            temperature=temperature,
        )
        self.executor = Executor(
            llm_engine_name=model,
            tool_instances=tool_instances,
            api_key=api_key,
            temperature=temperature,
        )
        self.memory = Memory()

    # ------------------------------------------------------------------
    def solve(self, question: str) -> dict[str, Any]:
        """Run the full AgentFlow loop and return a results dict."""
        self.memory.reset()
        json_data: dict[str, Any] = {"query": question}

        if self.verbose:
            print(f"\n==> Query: {question}")

        # Step 0 -- analyse query
        t0 = time.time()
        query_analysis = self.planner.analyze_query(question)
        json_data["query_analysis"] = query_analysis
        if self.verbose:
            print(f"\n==> Step 0: Query Analysis ({time.time()-t0:.1f}s)")

        # Main loop
        step = 0
        loop_start = time.time()
        consecutive_tool_errors = 0
        last_error_tool: str | None = None
        while step < self.max_steps and (time.time() - loop_start) < self.max_time:
            step += 1
            t1 = time.time()

            # 1. Planner -> next step
            next_step = self.planner.generate_next_step(
                question, query_analysis, self.memory,
                step, self.max_steps, json_data,
            )
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            if self.verbose:
                print(f"\n==> Step {step}: Action ({tool_name}) -- {sub_goal[:80]}")

            if not tool_name or tool_name not in self.executor.tool_instances:
                command = "N/A"
                result: Any = f"Tool '{tool_name}' not available."
            else:
                # 2. Executor -> generate & run command
                tool_cmd = self.executor.generate_tool_command(
                    question, context, sub_goal, tool_name,
                    self.planner.toolbox_metadata.get(tool_name, {}),
                    step, json_data,
                )
                _, _, command = Executor.extract_explanation_and_command(tool_cmd)
                result = self.executor.execute_tool_command(tool_name, command)

            if self.verbose:
                res_str = str(result)[:200]
                print(f"    Result: {res_str}")

            self.memory.add_action(step, tool_name, sub_goal, command, result)

            # Short-circuit on repeated failures from the same tool.
            result_lc = str(result).lower()
            is_error = ("error" in result_lc[:60]) or "traceback" in result_lc
            if is_error and tool_name == last_error_tool:
                consecutive_tool_errors += 1
            else:
                consecutive_tool_errors = 1 if is_error else 0
            last_error_tool = tool_name if is_error else None
            if consecutive_tool_errors >= 2:
                if self.verbose:
                    print(f"    [guard] {tool_name} errored {consecutive_tool_errors}x; stopping loop.")
                break

            # 3. Verifier -> STOP / CONTINUE
            verification = self.verifier.verificate_context(
                question, query_analysis, self.memory, step, json_data,
            )
            _, conclusion = self.verifier.extract_conclusion(verification)
            if self.verbose:
                print(f"    Verifier: {conclusion} ({time.time()-t1:.1f}s)")
            if conclusion == "STOP":
                break

        # Final answer
        direct_output = self.planner.generate_direct_output(question, self.memory)
        json_data.update({
            "direct_output": direct_output,
            "memory": self.memory.get_actions(),
            "step_count": step,
            "execution_time": round(time.time() - loop_start, 2),
        })
        if self.verbose:
            print(f"\n==> Final Answer:\n{direct_output[:500]}")

        return json_data


# ---- convenience entry point for quick testing ----------------------------
if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "@deepinfra/Qwen/Qwen2.5-7B-Instruct"
    solver = Solver(model=model, max_steps=5, verbose=True)
    result = solver.solve("What is the capital of France?")
    print(json.dumps({"answer": result["direct_output"][:300]}, indent=2))
