"""office_edit tool — refine an existing generated Office document.

Lets the LLM modify an existing generated document using OfficeCLI batch
commands. This is the iterative refinement tool used after office_inspect
finds issues.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, List

from pydantic import BaseModel, Field

from app.models.chat import ChatFile
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)


class OfficeEditInput(BaseModel):
    file_id: int = Field(description="ChatFile ID of the document to edit")
    commands: List[dict] = Field(description=(
        "OfficeCLI batch items to apply. Each item is a dict with 'command' "
        "(add/set/remove/move), 'path' or 'parent', 'type', and 'props'. "
        "Example: {\"command\": \"set\", \"path\": \"/slide[2]/shape[1]\", "
        "\"props\": {\"size\": \"36pt\"}}"
    ))


class OfficeEditTool(BaseAgentTool):
    name: str = "office_edit"
    ui_label: str = "Editing Office document"
    description: str = (
        "Edit an existing generated Office document with OfficeCLI batch commands. "
        "Use after office_inspect finds issues. Each command is a dict with "
        "'command' (add/set/remove/move), 'path' or 'parent', 'type', and 'props'. "
        "The file is modified in-place — no new file is created."
    )
    args_schema: type[BaseModel] = OfficeEditInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: OfficeEditInput) -> dict:
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

        # 3. Apply batch commands via OfficeCLI SDK
        try:
            import officecli
            with officecli.open(file_path, binary=binary, auto_install=False) as doc:
                # Batch in chunks of 50
                results = []
                for chunk_start in range(0, len(input_obj.commands), 50):
                    chunk = input_obj.commands[chunk_start:chunk_start + 50]
                    result = doc.batch(chunk)
                    if result:
                        results.append(result)
                doc.send({"command": "save"})
        except ImportError:
            return {"ok": False, "result": {}, "error": "officecli SDK not installed.", "tokens": 0}
        except Exception as exc:
            logger.warning("[office_edit] OfficeCLI failed: %s", exc)
            return {"ok": False, "result": {}, "error": f"OfficeCLI error: {exc}", "tokens": 0}

        # 4. Update file size
        try:
            chat_file.file_size = os.path.getsize(file_path)
            ctx.db.commit()
        except OSError:
            pass

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "office_edit", {"file_id": input_obj.file_id, "command_count": len(input_obj.commands)},
                    {"commands_applied": len(input_obj.commands)}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "file_id": input_obj.file_id,
                "commands_applied": len(input_obj.commands),
                "results": results if results else None,
            },
            "error": None,
            "tokens": len(str(results)) // 4 if results else 20,
        }
