"use client";

import { useState } from "react";
import { useDroppable } from "@dnd-kit/core";
import { ChevronRight, ChevronDown, Folder, FolderOpen, Pencil, Trash2, MessageSquare } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface Chat {
  id: number;
  title: string;
  pinned: boolean;
  folder_id?: number | null;
}

interface FolderItemProps {
  id: number;
  name: string;
  chats: Chat[];
  onRename: (id: number, newName: string) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onCloseNav?: () => void;
}

export default function FolderItem({
  id,
  name,
  chats,
  onRename,
  onDelete,
  onCloseNav,
}: FolderItemProps) {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState(true);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(name);

  const { isOver, setNodeRef } = useDroppable({ id: `folder-${id}` });

  const commitRename = async () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== name) {
      await onRename(id, trimmed).catch(() => {});
    } else {
      setEditValue(name);
    }
    setEditing(false);
  };

  return (
    <div
      ref={setNodeRef}
      className={[
        "rounded-lg transition-colors",
        isOver ? "bg-accent/40 ring-1 ring-primary/40" : "",
      ].join(" ")}
    >
      {/* Folder header row */}
      <div className="group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm hover:bg-accent/60 cursor-pointer transition-colors">
        <button
          onClick={() => setExpanded((p) => !p)}
          className="flex items-center gap-1.5 flex-1 min-w-0 text-left"
          aria-label={expanded ? "Collapse folder" : "Expand folder"}
        >
          <span className="shrink-0 text-muted-foreground">
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </span>
          <span className="shrink-0 text-muted-foreground">
            {expanded ? (
              <FolderOpen className="h-3.5 w-3.5" />
            ) : (
              <Folder className="h-3.5 w-3.5" />
            )}
          </span>
          {editing ? (
            <input
              autoFocus
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitRename();
                if (e.key === "Escape") { setEditValue(name); setEditing(false); }
              }}
              onClick={(e) => e.stopPropagation()}
              className="flex-1 min-w-0 bg-transparent border-b border-primary outline-none text-sm px-0.5"
            />
          ) : (
            <span className="truncate font-medium">{name}</span>
          )}
        </button>

        {!editing && (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
            <button
              onClick={(e) => { e.stopPropagation(); setEditing(true); setEditValue(name); }}
              className="p-1 rounded hover:bg-muted-foreground/20 transition-colors"
              aria-label="Rename folder"
            >
              <Pencil className="h-3 w-3" />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(id).catch(() => {}); }}
              className="p-1 rounded hover:bg-destructive/20 hover:text-destructive transition-colors"
              aria-label="Delete folder"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>

      {/* Child chats */}
      {expanded && chats.length > 0 && (
        <div className="ml-4 space-y-0.5">
          {chats.map((chat) => {
            const isActive = pathname === `/dashboard/chat/${chat.id}`;
            return (
              <Link
                key={chat.id}
                href={`/dashboard/chat/${chat.id}`}
                onClick={onCloseNav}
                className={[
                  "flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors",
                  isActive
                    ? "bg-accent text-accent-foreground"
                    : "hover:bg-accent/60 text-foreground",
                ].join(" ")}
              >
                <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-50" />
                <span className="truncate">{chat.title}</span>
              </Link>
            );
          })}
        </div>
      )}

      {expanded && chats.length === 0 && (
        <p className="ml-8 text-xs text-muted-foreground py-1 italic">
          {isOver ? "Drop here" : "Empty folder"}
        </p>
      )}
    </div>
  );
}
