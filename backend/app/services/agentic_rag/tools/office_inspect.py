"""office_inspect tool — inspect a generated Office document for quality.

Lets the LLM inspect a generated document's current state and run quality
checks. Supports structural checks (officecli's built-in), visual checks
(screenshot rendering for multimodal LLMs), and data extraction (get/query).
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)


class OfficeInspectInput(BaseModel):
    file_id: int = Field(description="ChatFile ID of the document to inspect")
    mode: str = Field(description=(
        "Inspection mode: outline | issues | annotated | text | screenshot | get | query | validate"
    ))
    path: Optional[str] = Field(default=None, description="Element path for get mode, e.g. /slide[2]/chart[1]")
    selector: Optional[str] = Field(default=None, description="CSS-like selector for query mode, e.g. shape:contains('TODO')")
    page: Optional[int] = Field(default=None, description="Slide/page number for screenshot mode (1-based)")
    issue_type: Optional[str] = Field(default=None, description="Filter for issues mode: format | content | structure")


class OfficeInspectTool(BaseAgentTool):
    name: str = "office_inspect"
    ui_label: str = "Inspecting Office document"
    description: str = (
        "Inspect a generated Office document for quality issues. Modes: "
        "outline (heading structure), issues (overflow/contrast/placeholders), "
        "annotated (text with formatting), text (raw text), screenshot (render to PNG for visual QA), "
        "get (read a specific element), query (search elements by selector), validate (schema check). "
        "Use after office_generate to verify quality. Use office_edit to fix any issues found."
    )
    args_schema: type[BaseModel] = OfficeInspectInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: OfficeInspectInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        # 1. Resolve file
        chat_file = ctx.db.query(ChatFile).filter(
            ChatFile.id == input_obj.file_id,
            ChatFile.chat_id == ctx.chat_id,
        ).first()
        if not chat_file or not chat_file.stored_path:
            return {"ok": False, "result": {}, "error": f"File not found: id={input_obj.file_id}", "tokens": 0}

        file_path = chat_file.stored_path
        if not os.path.exists(file_path):
            return {"ok": False, "result": {}, "error": "File no longer exists on disk.", "tokens": 0}

        # 2. Resolve binary
        binary = get_setting(ctx.db, "OFFICECLI_BINARY_PATH", ctx.org_id)

        mode = input_obj.mode.lower()

        # 3. Handle screenshot specially (needs rendering + optional vision)
        if mode == "screenshot":
            result = self._screenshot(file_path, input_obj.page, binary, ctx)
            latency_ms = round((time.monotonic() - t0) * 1000)
            write_audit(ctx, "office_inspect", {"file_id": input_obj.file_id, "mode": mode, "page": input_obj.page},
                        {"screenshot": True}, latency_ms=latency_ms, status="ok")
            return result

        # 4. Run officecli command
        try:
            cmd = [binary]
            if mode == "get":
                cmd.extend(["get", file_path, input_obj.path or "/"])
            elif mode == "query":
                cmd.extend(["query", file_path, input_obj.selector or "*"])
            elif mode == "validate":
                cmd.extend(["validate", file_path])
            elif mode in ("outline", "issues", "annotated", "text"):
                cmd.extend(["view", file_path, mode])
                if mode == "issues" and input_obj.issue_type:
                    cmd.extend(["--type", input_obj.issue_type])
            else:
                return {"ok": False, "result": {}, "error": f"Unknown mode: {mode}", "tokens": 0}

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = proc.stdout
            if proc.returncode != 0 and not output:
                return {"ok": False, "result": {}, "error": f"officecli error: {proc.stderr[:500]}", "tokens": 0}

        except subprocess.TimeoutExpired:
            return {"ok": False, "result": {}, "error": "officecli timed out (60s)", "tokens": 0}
        except FileNotFoundError:
            return {"ok": False, "result": {}, "error": f"officecli binary not found: {binary}", "tokens": 0}
        except Exception as exc:
            return {"ok": False, "result": {}, "error": f"Inspection failed: {exc}", "tokens": 0}

        # 5. Truncate very long outputs
        max_chars = 8000
        truncated = False
        if len(output) > max_chars:
            output = output[:max_chars] + "\n... (truncated)"
            truncated = True

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "office_inspect", {"file_id": input_obj.file_id, "mode": mode},
                    {"output_len": len(output), "truncated": truncated}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "output": output,
                "mode": mode,
                "truncated": truncated,
            },
            "error": None,
            "tokens": len(output) // 4,
        }

    def _screenshot(self, file_path: str, page: Optional[int], binary: str, ctx: ToolContext) -> dict:
        """Render a slide/page to PNG and return as base64 for multimodal LLMs."""
        work_dir = get_setting(ctx.db, "OFFICECLI_WORK_DIR", ctx.org_id)
        chat_work_dir = os.path.join(work_dir, str(ctx.chat_id or "default"))
        os.makedirs(chat_work_dir, exist_ok=True)

        png_name = f"screenshot_{page or 1}_{int(time.time())}.png"
        png_path = os.path.join(chat_work_dir, png_name)

        try:
            cmd = [binary, "view", file_path, "screenshot", "-o", png_path]
            if page:
                cmd.extend(["--page", str(page)])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                return {"ok": False, "result": {}, "error": f"Screenshot failed: {proc.stderr[:300]}", "tokens": 0}
        except subprocess.TimeoutExpired:
            return {"ok": False, "result": {}, "error": "Screenshot timed out", "tokens": 0}
        except Exception as exc:
            return {"ok": False, "result": {}, "error": f"Screenshot failed: {exc}", "tokens": 0}

        if not os.path.exists(png_path):
            return {"ok": False, "result": {}, "error": "Screenshot file not created.", "tokens": 0}

        try:
            with open(png_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
        finally:
            try:
                os.remove(png_path)
            except OSError:
                pass

        # Check if visual QA is enabled
        visual_qa = get_setting(ctx.db, "OFFICECLI_VISUAL_QA", ctx.org_id)
        if not visual_qa:
            return {
                "ok": True,
                "result": {
                    "mode": "screenshot",
                    "image_available": False,
                    "note": "Visual QA is disabled (OFFICECLI_VISUAL_QA=false). Use 'issues' or 'annotated' mode for text-based QA.",
                },
                "error": None,
                "tokens": 20,
            }

        return {
            "ok": True,
            "result": {
                "mode": "screenshot",
                "image_available": True,
                "image_base64": img_data,
                "page": page,
            },
            "error": None,
            "tokens": 1000,  # images cost more tokens
        }
