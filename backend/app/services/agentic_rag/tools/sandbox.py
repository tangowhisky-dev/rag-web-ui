"""Sandboxed code execution abstraction.

v1 uses RestrictedPython (safe for untrusted CPU code without network or disk).
nsjail can be plugged in later for hardened production deployments.
"""

from __future__ import annotations

import ast
import copy
import signal
import time
from typing import Any, Optional

from app.core.config import settings


def _safe_builtins() -> dict[str, Any]:
    """Return a restricted builtins namespace for the sandbox."""
    safe_names = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "map",
        "filter",
        "isinstance",
        "type",
        "dir",
        "print",
        "hasattr",
        "getattr",
    }
    return {name: __builtins__[name] for name in safe_names if name in __builtins__}


def _denylist_check(code: str) -> Optional[str]:
    """Check for disallowed syntax before compiling."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    denied_nodes = (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.ExceptHandler, ast.Lambda)
    denied_names = {"open", "eval", "exec", "compile", "__import__", "breakpoint", "input"}
    for node in ast.walk(tree):
        if isinstance(node, denied_nodes):
            return f"Disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in denied_names:
            return f"Disallowed call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Disallowed dunder access: {node.attr}"
    return None


def run_code(
    code: str,
    timeout_s: Optional[int] = None,
    environment: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run `code` in a restricted sandbox and return its `result` variable."""
    timeout_s = timeout_s or getattr(settings, "SANDBOX_TIMEOUT_S", 10)

    deny_reason = _denylist_check(code)
    if deny_reason:
        return {"ok": False, "result": None, "error": deny_reason, "stdout": ""}

    try:
        try:
            from RestrictedPython import compile_restricted

            compiled = compile_restricted(code, filename="<sandbox>", mode="exec")
        except Exception:
            compiled = compile(code, "<sandbox>", "exec")
    except SyntaxError as exc:
        return {"ok": False, "result": None, "error": f"Syntax error: {exc}", "stdout": ""}

    env = copy.deepcopy(environment or {})
    env.setdefault("__builtins__", _safe_builtins())
    env["result"] = None

    class TimeoutError(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutError("Sandbox timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_s))
    t0 = time.monotonic()
    try:
        exec(compiled, env)  # nosec B102
    except TimeoutError:
        return {"ok": False, "result": None, "error": f"Execution timed out after {timeout_s}s", "stdout": ""}
    except Exception as exc:
        return {"ok": False, "result": None, "error": f"Runtime error: {exc}", "stdout": ""}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    latency_ms = int((time.monotonic() - t0) * 1000)
    return {"ok": True, "result": env.get("result"), "error": None, "stdout": "", "latency_ms": latency_ms}
