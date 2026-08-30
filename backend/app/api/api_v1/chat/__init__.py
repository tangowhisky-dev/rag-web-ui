"""Chat API router — CRUD, messaging, branching, and clarification endpoints.

Assembles the single ``router`` used by ``app.api.api_v1.api`` and delegates
route registration to four submodules:

* ``chats``     — chat CRUD, search, and stream cancellation.
* ``messages``  — message creation (text + file), pagination, editing, and
  branch sibling listing.
* ``branching`` — active-branch selection and the clarification state machine
  that drives the agentic RAG pipeline's interrupt/resume flow.
* ``exports``   — single-message and whole-chat export (PDF, Word, PNG, MD).

Each submodule imports ``router`` from this package and registers its own
``@router`` decorators at import time.
"""
import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# Importing the submodules registers their route handlers on ``router``.
from . import chats, messages, branching, exports  # noqa: E402, F401
