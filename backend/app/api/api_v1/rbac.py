"""Shared RBAC helpers for chat/message resource access.

Every chat and message is owned by a single user (user_id).
Access is strictly user-level — org membership does not grant cross-user visibility.
"""

from app.models.chat import Chat


def chat_owner_filter(current_user):
    """Return SQLAlchemy filter clause scoping Chat access.

    Users can only access chats they personally own (user_id).
    """
    return Chat.user_id == current_user.id
