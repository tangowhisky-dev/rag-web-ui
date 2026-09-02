import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.chat import Chat, Folder
from app.models.user import User
from app.api.api_v1.rbac import chat_owner_filter as _chat_owner_filter
from app.schemas.chat import FolderCreate, FolderResponse, FolderUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=FolderResponse)
def create_folder(
    *,
    db: Session = Depends(get_db),
    folder_in: FolderCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    folder = Folder(name=folder_in.name, user_id=current_user.id)
    db.add(folder)
    db.commit()
    db.refresh(folder)
    logger.debug("[FOLDER] action=create folder_id=%s user_id=%s", folder.id, current_user.id)
    return folder


@router.get("", response_model=List[FolderResponse])
def list_folders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    return db.query(Folder).filter(Folder.user_id == current_user.id).all()


@router.patch("/{folder_id}", response_model=FolderResponse)
def rename_folder(
    folder_id: int,
    folder_in: FolderUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    folder = db.query(Folder).filter(
        Folder.id == folder_id, Folder.user_id == current_user.id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    if folder_in.name is not None:
        folder.name = folder_in.name
    db.commit()
    db.refresh(folder)
    logger.debug("[FOLDER] action=rename folder_id=%s user_id=%s", folder.id, current_user.id)
    return folder


@router.delete("/{folder_id}", status_code=204)
def delete_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    folder = db.query(Folder).filter(
        Folder.id == folder_id, Folder.user_id == current_user.id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    # Unassign chats from this folder
    db.query(Chat).filter(
        Chat.folder_id == folder_id, _chat_owner_filter(current_user)
    ).update({"folder_id": None})
    db.delete(folder)
    db.commit()
    logger.debug("[FOLDER] action=delete folder_id=%s user_id=%s", folder_id, current_user.id)


@router.patch("/{folder_id}/chats/{chat_id}", response_model=dict)
def assign_chat(
    folder_id: int,
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    folder = db.query(Folder).filter(
        Folder.id == folder_id, Folder.user_id == current_user.id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    chat = db.query(Chat).filter(
        Chat.id == chat_id, _chat_owner_filter(current_user)
    ).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat.folder_id = folder_id
    db.commit()
    logger.info(
        "[FOLDER] action=assign folder_id=%s chat_id=%s user_id=%s",
        folder_id, chat_id, current_user.id,
    )
    return {"folder_id": folder_id, "chat_id": chat_id}


@router.delete("/{folder_id}/chats/{chat_id}", status_code=204)
def unassign_chat(
    folder_id: int,
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.folder_id == folder_id,
        _chat_owner_filter(current_user),
    ).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not in folder or not found")
    chat.folder_id = None
    db.commit()
    logger.info(
        "[FOLDER] action=unassign folder_id=%s chat_id=%s user_id=%s",
        folder_id, chat_id, current_user.id,
    )
