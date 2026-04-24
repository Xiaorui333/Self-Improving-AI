"""Shared memory that persists across turns inside a single AgentFlow solve."""

from __future__ import annotations

from typing import Any


class Memory:
    def __init__(self) -> None:
        self.actions: dict[str, dict[str, Any]] = {}

    def add_action(
        self,
        step_count: int,
        tool_name: str,
        sub_goal: str,
        command: str,
        result: Any,
    ) -> None:
        self.actions[f"Action Step {step_count}"] = {
            "tool_name": tool_name,
            "sub_goal": sub_goal,
            "command": command,
            "result": result,
        }

    def get_actions(self) -> dict[str, dict[str, Any]]:
        return self.actions

    def reset(self) -> None:
        self.actions.clear()
