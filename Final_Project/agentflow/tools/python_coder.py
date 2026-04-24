"""Python Coder Tool -- executes Python code in a subprocess sandbox."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from agentflow.tools.base import BaseTool


class PythonCoderTool(BaseTool):
    def __init__(self, model_string: str = "", *, api_key: str | None = None) -> None:
        super().__init__(
            tool_name="Python_Coder_Tool",
            tool_description=(
                "Executes a Python code snippet in a sandboxed subprocess and "
                "returns stdout/stderr. Useful for calculations, data processing, "
                "and symbolic math."
            ),
            input_types={
                "query": "str - Python code to execute (or a natural-language request converted to code by the executor).",
            },
            output_type="str - stdout from the executed code.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="print(2**10)")',
                    "description": "Run a simple Python expression.",
                }
            ],
            user_metadata={
                "limitation": "No network access; 60-second timeout.",
                "best_practice": "Print results explicitly; keep code self-contained.",
            },
            model_string=model_string,
            api_key=api_key,
        )

    def execute(self, query: str, **kwargs: Any) -> str:
        code = textwrap.dedent(query).strip()
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name
            # Use the running interpreter so we don't depend on a `python` binary.
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: code execution timed out (60s limit)."
        except Exception as exc:
            return f"Error executing code: {exc}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
