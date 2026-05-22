"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import {
  Plus, Pencil, Trash2, X, MessageSquare, Pin, Search,
  Settings, Download, PanelLeftClose, PanelLeftOpen,
} from "lucide-react";
import { useChatContext } from "@/contexts/chat-context";

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ChatSidebar({ isOpen, onClose }: ChatSidebarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { chatList, activeChat, renameChat, deleteChat, patchChat } = useChatContext();

  // Always start uncollapsed on SSR; sync from localStorage after mount to avoid hydration mismatch
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    setCollapsed(localStorage.getItem("chat-sidebar-collapsed") === "true");
  }, []);

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId !== null) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editingId]);

  const toggleCollapse = () => {
    setCollapsed((prev) => {
      localStorage.setItem("chat-sidebar-collapsed", String(!prev));
      return !prev;
    });
  };

  const startEdit = (id: number, title: string) => {
    setEditingId(id);
    setEditingValue(title);
  };

  const commitEdit = async (id: number) => {
    const trimmed = editingValue.trim();
    if (trimmed && trimmed !== chatList.find((c) => c.id === id)?.title) {
      await renameChat(id, trimmed).catch(() => {});
    }
    setEditingId(null);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent, id: number) => {
    if (e.key === "Enter") commitEdit(id);
    if (e.key === "Escape") setEditingId(null);
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm("Delete this chat?")) return;
    await deleteChat(id).catch(() => {});
    if (pathname === `/dashboard/chat/${id}`) {
      router.push("/dashboard/chat");
    }
  };

  const handleTogglePin = async (id: number, pinned: boolean) => {
    await patchChat(id, { pinned: !pinned }).catch(() => {});
  };

  const handleExport = async (id: number) => {
    try {
      const res = await fetch(`/api/chat/${id}/export`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token") ?? ""}` },
      });
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chat-${id}.md`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed", e);
    }
  };

  const filtered = chatList.filter((c) => {
    if (!searchQuery.trim()) return true;
    return c.title.toLowerCase().includes(searchQuery.toLowerCase());
  });
  const pinned = filtered.filter((c) => c.pinned);
  const unpinned = filtered.filter((c) => !c.pinned);

  const renderChatItem = (chat: (typeof chatList)[0]) => {
    const isActive =
      activeChat === chat.id || pathname === `/dashboard/chat/${chat.id}`;
    return (
      <div
        key={chat.id}
        data-testid={`chat-item-${chat.id}`}
        className={[
          "group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm transition-colors cursor-pointer",
          isActive
            ? "bg-accent text-accent-foreground"
            : "hover:bg-accent/60 text-foreground",
        ].join(" ")}
      >
        {editingId === chat.id ? (
          <input
            ref={inputRef}
            value={editingValue}
            onChange={(e) => setEditingValue(e.target.value)}
            onBlur={() => commitEdit(chat.id)}
            onKeyDown={(e) => handleEditKeyDown(e, chat.id)}
            className="flex-1 min-w-0 bg-transparent border-b border-primary outline-none text-sm px-0.5"
          />
        ) : (
          <Link
            href={`/dashboard/chat/${chat.id}`}
            onClick={onClose}
            className="flex items-center gap-2 flex-1 min-w-0"
          >
            <MessageSquare className="h-3.5 w-3.5 shrink-0 opacity-50" />
            <span className="truncate">{chat.title}</span>
          </Link>
        )}

        {editingId !== chat.id && (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
            <button
              onClick={() => handleTogglePin(chat.id, chat.pinned)}
              className={[
                "p-1 rounded transition-colors",
                chat.pinned ? "text-primary" : "hover:bg-muted-foreground/20",
              ].join(" ")}
              aria-label={chat.pinned ? "Unpin" : "Pin"}
            >
              <Pin className="h-3 w-3" />
            </button>
            <button
              onClick={(e) => { e.preventDefault(); handleExport(chat.id); }}
              className="p-1 rounded hover:bg-muted-foreground/20 transition-colors"
              aria-label="Download chat"
            >
              <Download className="h-3 w-3" />
            </button>
            <button
              onClick={() => startEdit(chat.id, chat.title)}
              className="p-1 rounded hover:bg-muted-foreground/20 transition-colors"
              aria-label="Rename"
            >
              <Pencil className="h-3 w-3" />
            </button>
            <button
              onClick={() => handleDelete(chat.id)}
              className="p-1 rounded hover:bg-destructive/20 hover:text-destructive transition-colors"
              aria-label="Delete"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={[
          "fixed inset-y-0 left-0 z-30 bg-card border-r flex flex-col",
          "transition-all duration-200 ease-in-out",
          "lg:relative lg:inset-auto lg:translate-x-0 lg:z-auto lg:h-full",
          // Width: collapsed = icon rail, expanded = full
          collapsed ? "w-12" : "w-64",
          // Mobile show/hide
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        ].join(" ")}
      >
        {/* ── Collapsed view ─────────────────────────────────────────── */}
        {collapsed && (
          <div className="flex flex-col items-center gap-2 py-3 flex-1">
            {/* Expand button */}
            <button
              onClick={toggleCollapse}
              className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              aria-label="Expand sidebar"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
            <div className="flex-1" />
            {/* Settings */}
            <Link
              href="/dashboard/settings"
              className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              aria-label="Settings"
            >
              <Settings className="h-4 w-4" />
            </Link>
          </div>
        )}

        {/* ── Expanded view ──────────────────────────────────────────── */}
        {!collapsed && (
          <>
            {/* Collapse + New Chat row */}
            <div className="flex items-center gap-1 px-3 pt-3 pb-2 shrink-0">
              {/* New Chat */}
              <Link
                href="/dashboard/chat/new"
                onClick={onClose}
                data-testid="new-chat-button"
                className="flex items-center gap-2 flex-1 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent transition-colors"
              >
                <Plus className="h-4 w-4 shrink-0" />
                New Chat
              </Link>
              {/* Desktop: collapse button */}
              <button
                onClick={toggleCollapse}
                className="hidden lg:flex p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0"
                aria-label="Collapse sidebar"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
              {/* Mobile: close button */}
              <button
                onClick={onClose}
                className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0"
                aria-label="Close sidebar"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Search */}
            <div className="px-3 pb-3 shrink-0">
              <div className="flex items-center gap-1.5 rounded-lg border bg-muted/40 px-2.5 py-1.5">
                <Search className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <input
                  data-testid="chat-search"
                  type="text"
                  placeholder="Search chats…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                />
              </div>
            </div>

            {/* Chat list */}
            <nav
              data-testid="chat-list"
              className="flex-1 overflow-y-auto px-2 space-y-0.5"
            >
              {filtered.length === 0 && (
                <p className="text-xs text-muted-foreground text-center mt-6 px-4">
                  {searchQuery ? "No matching chats." : "No conversations yet."}
                </p>
              )}
              {pinned.length > 0 && (
                <p className="text-xs text-muted-foreground px-2 pt-1 pb-0.5 uppercase tracking-wide">
                  Pinned
                </p>
              )}
              {pinned.map(renderChatItem)}
              {pinned.length > 0 && unpinned.length > 0 && (
                <p className="text-xs text-muted-foreground px-2 pt-3 pb-0.5 uppercase tracking-wide">
                  Recent
                </p>
              )}
              {unpinned.map(renderChatItem)}
            </nav>

            {/* Bottom: Settings */}
            <div className="border-t px-3 py-3 shrink-0">
              <Link
                href="/dashboard/settings"
                className="flex items-center gap-2 w-full rounded-lg px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              >
                <Settings className="h-4 w-4 shrink-0" />
                Settings
              </Link>
            </div>
          </>
        )}
      </aside>
    </>
  );
}
