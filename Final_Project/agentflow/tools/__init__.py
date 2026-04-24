"""AgentFlow tool implementations."""

from agentflow.tools.base_generator import BaseGeneratorTool
from agentflow.tools.python_coder import PythonCoderTool
from agentflow.tools.web_search import WebSearchTool
from agentflow.tools.wikipedia_search import WikipediaSearchTool

TOOL_REGISTRY: dict[str, type] = {
    "Base_Generator_Tool": BaseGeneratorTool,
    "Python_Coder_Tool": PythonCoderTool,
    "Web_Search_Tool": WebSearchTool,
    "Wikipedia_Search_Tool": WikipediaSearchTool,
}

TOOL_NAMES = list(TOOL_REGISTRY.keys())


def build_tools(engine_model: str, *, api_key: str | None = None) -> dict:
    """Instantiate every registered tool and return ``{name: instance}``."""
    return {
        name: cls(model_string=engine_model, api_key=api_key)
        for name, cls in TOOL_REGISTRY.items()
    }


def get_toolbox_metadata(engine_model: str = "", *, api_key: str | None = None) -> dict:
    """Return ``{name: metadata_dict}`` for every tool."""
    tools = build_tools(engine_model or "dummy", api_key=api_key or "dummy")
    return {name: tool.get_metadata() for name, tool in tools.items()}
