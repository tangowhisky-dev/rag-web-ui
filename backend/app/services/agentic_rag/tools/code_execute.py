"""code_execute tool — local Python sandbox for computation and transforms."""

from __future__ import annotations

import io
import logging
import operator
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
    code: str = Field(description="Python code to execute. Set the 'result' variable to return a value, or use print() to capture stdout.")
    data: Optional[dict] = Field(default=None, description="Variables to inject.")
    timeout_s: int = Field(default=30, ge=1, le=180)


_SAFE_IMPORT_TOP_LEVEL = frozenset({
    "numpy", "pandas", "matplotlib", "math", "statistics",
    "json", "re", "io", "collections", "itertools", "functools",
    "decimal", "datetime", "csv",
})


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__ — only allow whitelisted top-level packages."""
    top = name.split(".")[0]
    if top in _SAFE_IMPORT_TOP_LEVEL:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of {name!r} is not allowed in sandbox")


def _getiter_(ob):
    return ob


def _getitem_(ob, index):
    return ob[index]


def _write_(ob):
    return ob


_INPLACE_OPS = {
    "+=": operator.add, "-=": operator.sub, "*=": operator.mul,
    "/=": operator.truediv, "%=": operator.mod, "//=": operator.floordiv,
    "**=": operator.pow, "&=": operator.and_, "|=": operator.or_,
    "^=": operator.xor, "<<=": operator.lshift, ">>=": operator.rshift,
}


def _inplacevar_(op, x, y):
    return _INPLACE_OPS[op](x, y)


class CodeExecuteTool(BaseAgentTool):
    name: str = "code_execute"
    ui_label: str = "Executing Python code"
    description: str = (
        "Execute Python code in a restricted sandbox for computation, "
        "data transformation, or statistics. Use with data from extract_data or file_extract_table. "
        "Do NOT use this to build chart/ECharts options — use chart_generate for that.\n"
        "Sandbox details:\n"
        "- Available builtins: sum, min, max, len, print, list, dict, set, tuple, range, enumerate, "
        "sorted, reversed, all, any, abs, round, int, float, str, bool, map, filter, zip, type, isinstance.\n"
        "- Importable packages: math, statistics, json, re, io, collections, itertools, functools, "
        "decimal, datetime, csv, numpy (as np), pandas (as pd), matplotlib.pyplot (as plt).\n"
        "- Supported: list/dict/set comprehensions, for-loops, tuple unpacking, augmented assignment (+=, *=, etc.), "
        "f-strings, lambda, def, classes, try/except, with-statement.\n"
        "- NOT available: open(), eval(), exec(), getattr(), setattr(), globals(), locals(), __import__ of non-whitelisted packages, "
        "os, sys, subprocess, socket, pathlib, shutil, pickle, ctypes.\n"
        "- Output: set the 'result' variable to return a value, or use print() to capture stdout. "
        "Example: result = sum([1, 2, 3]) or print('hello')."
    )
    args_schema: type[BaseModel] = CodeExecuteInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: CodeExecuteInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        code = input_obj.code

        # Build a RestrictedPython-compatible globals dict.
        # safe_builtins provides a vetted subset of builtins (no open, eval,
        # exec, __import__, getattr). We add back the safe builtins the tool
        # needs (sum, min, max, print, list, dict, enumerate, all, any) and a
        # guarded __import__ that only allows whitelisted top-level packages.
        try:
            from RestrictedPython import safe_builtins
            from RestrictedPython.Guards import (
                safer_getattr,
                guarded_setattr,
                guarded_delattr,
                guarded_iter_unpack_sequence,
                guarded_unpack_sequence,
            )
            from RestrictedPython.PrintCollector import PrintCollector
        except ImportError:
            return {"ok": False, "result": {}, "error": "RestrictedPython is not installed; code execution disabled.", "tokens": 0}

        restricted_builtins = dict(safe_builtins)
        for name, fn in [
            ("sum", sum), ("min", min), ("max", max), ("print", print),
            ("list", list), ("dict", dict), ("set", set), ("enumerate", enumerate),
            ("all", all), ("any", any), ("reversed", reversed),
            ("map", map), ("filter", filter), ("type", type),
            ("__import__", _safe_import),
        ]:
            restricted_builtins[name] = fn

        globals_dict: dict[str, Any] = {
            "__builtins__": restricted_builtins,
            "_getattr_": safer_getattr,
            "_getiter_": _getiter_,
            "_getitem_": _getitem_,
            "_write_": _write_,
            "_print_": PrintCollector,
            "setattr": guarded_setattr,
            "delattr": guarded_delattr,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_inplacevar_": _inplacevar_,
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
            except Exception as exc:
                return {"ok": False, "result": {}, "error": f"Code compilation failed: {exc}", "tokens": 0}

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, globals_dict)

            result = globals_dict.get("result")
            if result is None:
                # try last expression? exec doesn't return. Use result variable.
                result = ""
        except TimeoutError as exc:
            return {"ok": False, "result": {}, "error": f"Timeout after {input_obj.timeout_s}s", "tokens": 0}
        except Exception as exc:
            return {"ok": False, "result": {}, "error": f"{exc}\n{traceback.format_exc()}", "tokens": 0}
        finally:
            # Always disarm the alarm on every exit path (including early
            # returns from compile failure) — an unarmed alarm otherwise
            # fires later inside the asyncio event loop and crashes the
            # whole worker process with an uncaught TimeoutError.
            signal.alarm(0)

        # RestrictedPython routes print() through a PrintCollector instance
        # bound to `_print` in globals_dict, not real stdout.
        print_collector = globals_dict.get("_print")
        stdout = stdout_buf.getvalue() + (print_collector() if print_collector else "")
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
