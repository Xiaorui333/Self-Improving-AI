"""Minimal base class for all AgentFlow tools."""

from __future__ import annotations

from typing import Any


class BaseTool:
    require_llm_engine: bool = False

    def __init__(
        self,
        tool_name: str = "",
        tool_description: str = "",
        input_types: dict | None = None,
        output_type: str = "",
        demo_commands: list | None = None,
        user_metadata: dict | None = None,
        model_string: str = "",
        api_key: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.input_types = input_types or {}
        self.output_type = output_type
        self.demo_commands = demo_commands or []
        self.user_metadata = user_metadata
        self.output_dir: str | None = None
        self.model_string = model_string
        self.api_key = api_key

    def get_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "demo_commands": self.demo_commands,
        }
        if self.user_metadata:
            meta["user_metadata"] = self.user_metadata
        return meta

    def set_custom_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def execute(self, **kwargs: Any) -> Any:
        raise NotImplementedError
