"""code_execute tool — local Python sandbox for computation and transforms."""

from __future__ import annotations

import io
import logging
import signal
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class CodeExecuteInput(BaseModel):
    code: str = Field(description="Python code to execute.")
    data: Optional[dict] = Field(default=None, description="Variables to inject.")
    timeout_s: int = Field(default=10, ge=1, le=60)


_DENY_RE = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import urllib",
    "import requests",
    "__import__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    ".system(",
    "subprocess",
    "socket",
]


class CodeExecuteTool(BaseAgentTool):
    name: str = "code_execute"
    description: str = (
        "Execute Python code in a restricted sandbox for computation, "
        "data transformation, or statistics. Use with data from extract_data or file_extract_table."
    )
    args_schema: type[BaseModel] = CodeExecuteInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: CodeExecuteInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        code = input_obj.code
        for bad in _DENY_RE:
            if bad in code:
                return {"ok": False, "result": {}, "error": f"Disallowed construct detected: {bad}", "tokens": 0}

        # Prepare globals
        globals_dict: dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "print": print,
                "range": range,
                "round": round,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            }
        }

        try:
            import numpy as np
            globals_dict["np"] = np
        except Exception:
            pass

        try:
            import pandas as pd
            globals_dict["pd"] = pd
        except Exception:
            pass

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            globals_dict["plt"] = plt
        except Exception:
            pass

        if input_obj.data:
            globals_dict.update(input_obj.data)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        def _alarm(signum, frame):
            raise TimeoutError("Code execution exceeded timeout")

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(input_obj.timeout_s)

        try:
            try:
                from RestrictedPython import compile_restricted
                compiled = compile_restricted(code, filename="<sandbox>", mode="exec")
                if compiled is None:
                    return {"ok": False, "result": {}, "error": "RestrictedPython refused to compile code.", "tokens": 0}
            except ImportError:
                return {"ok": False, "result": {}, "error": "RestrictedPython is not installed; code execution disabled.", "tokens": 0}
            except Exception as exc:
                return {"ok": False, "result": {}, "error": f"Code compilation failed: {exc}", "tokens": 0}

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, globals_dict)

            result = globals_dict.get("result")
            if result is None:
                # try last expression? exec doesn't return. Use result variable.
                result = ""

            signal.alarm(0)
        except TimeoutError as exc:
            return {"ok": False, "result": {}, "error": f"Timeout after {input_obj.timeout_s}s", "tokens": 0}
        except Exception as exc:
            signal.alarm(0)
            return {"ok": False, "result": {}, "error": f"{exc}\n{traceback.format_exc()}", "tokens": 0}

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        # Capture matplotlib figure as base64 if any
        plots: list[str] = []
        try:
            import base64
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plots.append(base64.b64encode(buf.getvalue()).decode())
                plt.close(fig)
        except Exception:
            pass

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "code_execute", input_obj.model_dump(), {"stdout_len": len(stdout)}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "stdout": stdout,
                "stderr": stderr,
                "result": result,
                "plots": plots,
            },
            "error": None,
            "tokens": len(code) // 4,
        }
