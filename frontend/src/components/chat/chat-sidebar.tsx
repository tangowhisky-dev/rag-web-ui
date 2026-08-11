"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Plus, Pencil, Trash2, X, MessageSquare, Pin, Search,
  Settings, Download, PanelLeftClose, PanelLeftOpen, FolderPlus,
} from "lucide-react";
import { DndContext, DragEndEvent, useDraggable } from "@dnd-kit/core";
import { CSS } from "@dnd-kit/utilities";
import { useChatContext } from "@/contexts/chat-context";
import FolderItem from "@/components/chat/folder-item";
import { useSidebarCollapse, exportChatToMarkdown } from "@/lib/hooks";
import { api, ApiError } from "@/lib/api";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

// Draggable chat item wrapper — drag handle is the icon+title area only,
// so action buttons (pin, rename, delete, export) receive clicks normally.
function DraggableChatItem({
  chat,
  children,
}: {
  chat: { id: number };
  children: (dragHandleProps: React.HTMLAttributes<HTMLElement>) => React.ReactNode;
}) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: String(chat.id),
  });
  const style = {
    transform: CSS.Translate.toString(transform),
    opacity: isDragging ? 0.5 : 1,
  };
  // Pass drag listeners down so only the handle area activates drag
  const dragHandleProps = { ...listeners, ...attributes, style: { cursor: isDragging ? "grabbing" : "grab" } };
  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      layout
      transition={{ type: "spring", stiffness: 500, damping: 35, mass: 0.3 }}
    >
      {children(dragHandleProps)}
    </motion.div>
  );
}

export default function ChatSidebar({ isOpen, onClose }: ChatSidebarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const {
    chatList,
    activeChat,
    renameChat,
    deleteChat,
    patchChat,
    folderList,
    createFolder,
    renameFolder,
    deleteFolder,
    assignChatToFolder,
    chatListLoaded,
  } = useChatContext();

  const { toast } = useToast();

  const { collapsed, toggleCollapse } = useSidebarCollapse("chat-sidebar-collapsed");

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Message search state ───────────────────────────────────────────
  interface SearchResult {
    chat_id: number;
    chat_title: string;
    snippet: string;
  }
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runSearch = useCallback(async (q: string) => {
    setSearchLoading(true);
    setSearchError(false);
    console.debug("[SEARCH] query=%s", q);
    try {
      const data: SearchResult[] = await api.get(`/api/chat/search?q=${encodeURIComponent(q)}`);
      console.debug("[SEARCH] result_count=%d", data.length);
      setSearchResults(data);
    } catch {
      setSearchError(true);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  }, []);

  // useDebounce pattern (300 ms): call backend when query >= 4 chars, clear results otherwise
  useEffect(() => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    if (searchQuery.length >= 4) {
      searchTimerRef.current = setTimeout(() => runSearch(searchQuery), 300);
    } else {
      setSearchResults([]);
      setSearchError(false);
    }
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [searchQuery, runSearch]);

  // Highlight matching phrase inside snippet
  const highlightSnippet = (snippet: string, query: string) => {
    const idx = snippet.toLowerCase().indexOf(query.toLowerCase());
    if (idx === -1) return <span>{snippet}</span>;
    return (
      <>
        {snippet.slice(0, idx)}
        <mark className="bg-yellow-200 dark:bg-yellow-800 rounded px-0.5">
          {snippet.slice(idx, idx + query.length)}
        </mark>
        {snippet.slice(idx + query.length)}
      </>
    );
  };

  useEffect(() => {
    if (editingId !== null) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editingId]);

  const startEdit = (id: number, title: string) => {
    setEditingId(id);
    setEditingValue(title);
  };

  const commitEdit = async (id: number) => {
    const trimmed = editingValue.trim();
    if (trimmed && trimmed !== chatList.find((c) => c.id === id)?.title) {
      await renameChat(id, trimmed).catch((e) => {
        toast({ title: "Rename failed", description: e instanceof ApiError ? e.message : "Please try again", variant: "destructive" });
      });
    }
    setEditingId(null);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent, id: number) => {
    if (e.key === "Enter") commitEdit(id);
    if (e.key === "Escape") setEditingId(null);
  };

  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");

  const handleDelete = async (id: number) => {
    await deleteChat(id).catch((e) => {
      toast({ title: "Delete failed", description: e instanceof ApiError ? e.message : "Please try again", variant: "destructive" });
    });
    if (pathname === `/dashboard/chat/${id}`) {
      router.push("/dashboard/chat");
    }
  };

  const handleTogglePin = async (id: number, pinned: boolean) => {
    await patchChat(id, { pinned: !pinned }).catch((e) => {
      toast({ title: "Pin failed", description: e instanceof ApiError ? e.message : "Please try again", variant: "destructive" });
    });
  };

  const handleExport = async (id: number) => {
    try {
      await exportChatToMarkdown(id);
    } catch (e) {
      console.error("Export failed", e);
    }
  };

  const handleNewFolder = async () => {
    const name = newFolderName.trim();
    if (!name) return;
    await createFolder(name).catch((e) => {
      toast({ title: "Folder creation failed", description: e instanceof ApiError ? e.message : "Please try again", variant: "destructive" });
    });
    setNewFolderName("");
    setNewFolderOpen(false);
  };

  // DnD: drag a chat (id=String(chatId)) onto a folder (droppable id=`folder-{folderId}`)
  const onDragEnd = async (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over) return;
    const chatId = Number(active.id);
    const overStr = String(over.id);
    if (!overStr.startsWith("folder-")) return;
    const folderId = Number(overStr.replace("folder-", ""));
    if (!chatId || !folderId) return;
    console.debug("[DnD] chat_id=%d → folder_id=%d", chatId, folderId);
    await assignChatToFolder(chatId, folderId).catch((e) => {
      toast({ title: "Move failed", description: e instanceof ApiError ? e.message : "Could not move chat to folder", variant: "destructive" });
    });
  };

  // Filtered lists
  const filtered = chatList.filter((c) => {
    if (!searchQuery.trim()) return true;
    return c.title.toLowerCase().includes(searchQuery.toLowerCase());
  });

  // Chats without a folder (or not matching a known folder)
  const knownFolderIds = new Set(folderList.map((f) => f.id));
  const freeChats = filtered.filter(
    (c) => !c.folder_id || !knownFolderIds.has(c.folder_id)
  );
  const pinned = freeChats.filter((c) => c.pinned);
  const unpinned = freeChats.filter((c) => !c.pinned);

  const renderChatItem = (chat: (typeof chatList)[0]) => {
    const isActive =
      activeChat === chat.id || pathname === `/dashboard/chat/${chat.id}`;
    return (
      <DraggableChatItem key={chat.id} chat={chat}>
        {(dragHandleProps) => (
        <div
          data-testid={`chat-item-${chat.id}`}
          className={[
            "group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm transition-colors",
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
              {...dragHandleProps}
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
                onClick={() => setConfirmDeleteId(chat.id)}
                className="p-1 rounded hover:bg-destructive/20 hover:text-destructive transition-colors"
                aria-label="Delete"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          )}
        </div>
        )}
      </DraggableChatItem>
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
          collapsed ? "w-12" : "w-64",
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        ].join(" ")}
      >
        {/* ── Collapsed view ─────────────────────────────────────────── */}
        {collapsed && (
          <div className="flex flex-col items-center gap-2 py-3 flex-1">
            <button
              onClick={toggleCollapse}
              className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              aria-label="Expand sidebar"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
            <div className="flex-1" />
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
          <DndContext onDragEnd={onDragEnd}>
            {/* Header row */}
            <div className="flex items-center gap-1 px-3 pt-3 pb-2 shrink-0">
              <Link
                href="/dashboard/chat/new"
                onClick={onClose}
                data-testid="new-chat-button"
                className="flex items-center gap-2 flex-1 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent transition-colors"
              >
                <Plus className="h-4 w-4 shrink-0" />
                New Chat
              </Link>
              <button
                onClick={() => setNewFolderOpen(true)}
                className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0"
                aria-label="New Folder"
                title="New Folder"
              >
                <FolderPlus className="h-4 w-4" />
              </button>
              <button
                onClick={toggleCollapse}
                className="hidden lg:flex p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground shrink-0"
                aria-label="Collapse sidebar"
              >
                <PanelLeftClose className="h-4 w-4" />
              </button>
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
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery("")}
                    className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                    aria-label="Clear search"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Chat list */}
            <nav
              data-testid="chat-list"
              className="flex-1 overflow-y-auto px-2 space-y-0.5"
            >
              {/* ── Message search results ─────────────────────────── */}
              {searchQuery.length >= 4 && (
                <div data-testid="search-results" className="mb-2">
                  <p className="text-xs text-muted-foreground px-2 pt-1 pb-1 uppercase tracking-wide">
                    Message search
                  </p>
                  {searchLoading && (
                    <p className="text-xs text-muted-foreground px-2 py-1 animate-pulse">
                      Searching messages…
                    </p>
                  )}
                  {searchError && !searchLoading && (
                    <p className="text-xs text-destructive px-2 py-1">
                      Search unavailable
                    </p>
                  )}
                  {!searchLoading && !searchError && searchResults.length === 0 && (
                    <p className="text-xs text-muted-foreground px-2 py-1">
                      No message matches found.
                    </p>
                  )}
                  {!searchLoading && !searchError && searchResults.map((r) => (
                    <button
                      key={r.chat_id}
                      onClick={() => { router.push(`/dashboard/chat/${r.chat_id}`); onClose(); }}
                      className="w-full text-left rounded-lg px-2 py-1.5 hover:bg-accent/60 transition-colors"
                    >
                      <p className="text-sm font-semibold truncate">{r.chat_title}</p>
                      <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                        {highlightSnippet(r.snippet, searchQuery)}
                      </p>
                    </button>
                  ))}
                  <div className="border-b my-2 mx-2 opacity-30" />
                </div>
              )}
              {/* Folders section */}
              {folderList.length > 0 && (
                <>
                  <p className="text-xs text-muted-foreground px-2 pt-1 pb-0.5 uppercase tracking-wide">
                    Folders
                  </p>
                  {folderList.map((folder) => (
                    <FolderItem
                      key={folder.id}
                      id={folder.id}
                      name={folder.name}
                      chats={filtered.filter((c) => c.folder_id === folder.id)}
                      onRename={renameFolder}
                      onDelete={deleteFolder}
                      onCloseNav={onClose}
                    />
                  ))}
                  {(pinned.length > 0 || unpinned.length > 0) && (
                    <p className="text-xs text-muted-foreground px-2 pt-3 pb-0.5 uppercase tracking-wide">
                      Chats
                    </p>
                  )}
                </>
              )}

              {filtered.length === 0 && folderList.length === 0 && (
                <p className="text-xs text-muted-foreground text-center mt-6 px-4">
                  {searchQuery
                    ? "No matching chats."
                    : chatListLoaded
                    ? "No conversations yet."
                    : "Loading chats…"}
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
          </DndContext>
        )}
      </aside>

      <ConfirmDialog
        open={confirmDeleteId !== null}
        title="Delete chat"
        description="Delete this chat?"
        confirmText="Delete"
        destructive
        onConfirm={() => {
          if (confirmDeleteId !== null) handleDelete(confirmDeleteId);
          setConfirmDeleteId(null);
        }}
        onCancel={() => setConfirmDeleteId(null)}
      />

      <Dialog open={newFolderOpen} onOpenChange={(o) => { setNewFolderOpen(o); if (!o) setNewFolderName(""); }}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>New folder</DialogTitle>
          </DialogHeader>
          <Input
            autoFocus
            placeholder="Folder name"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleNewFolder();
              if (e.key === "Escape") { setNewFolderOpen(false); setNewFolderName(""); }
            }}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => { setNewFolderOpen(false); setNewFolderName(""); }}>
              Cancel
            </Button>
            <Button onClick={handleNewFolder} disabled={!newFolderName.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
