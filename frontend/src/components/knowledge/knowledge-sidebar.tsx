"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import {
  Plus, Pencil, Trash2, X, Database,
  Search, PanelLeftClose, PanelLeftOpen,
} from "lucide-react";
import { useKnowledgeContext } from "@/contexts/knowledge-context";
import type { KnowledgeBase } from "@/contexts/knowledge-context";
import { useSidebarCollapse } from "@/lib/hooks";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useToast } from "@/components/ui/use-toast";
import { ApiError } from "@/lib/api";

interface KnowledgeSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

function filterKbList(kbList: KnowledgeBase[], search: string): KnowledgeBase[] {
  const q = search.trim().toLowerCase();
  if (!q) return kbList;
  return kbList.filter((kb) =>
    kb.name.toLowerCase().includes(q) ||
    (kb.description ?? "").toLowerCase().includes(q)
  );
}

interface KbListItemProps {
  kb: KnowledgeBase;
  isActive: boolean;
  isEditing: boolean;
  editingValue: string;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onEditingValueChange: (value: string) => void;
  onCommitEdit: (id: number) => void;
  onEditKeyDown: (e: React.KeyboardEvent, id: number) => void;
  onStartEdit: (id: number, name: string) => void;
  onSetConfirmId: (id: number) => void;
  onClose: () => void;
}

function KbListItem({
  kb, isActive, isEditing, editingValue, inputRef,
  onEditingValueChange, onCommitEdit, onEditKeyDown,
  onStartEdit, onSetConfirmId, onClose,
}: KbListItemProps) {
  return (
    <div
      className={[
        "group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm transition-colors",
        isActive
          ? "bg-accent text-accent-foreground"
          : "hover:bg-accent/60 text-foreground",
      ].join(" ")}
    >
      {isEditing ? (
        <input
          ref={inputRef}
          value={editingValue}
          onChange={(e) => onEditingValueChange(e.target.value)}
          onBlur={() => onCommitEdit(kb.id)}
          onKeyDown={(e) => onEditKeyDown(e, kb.id)}
          className="flex-1 min-w-0 bg-transparent border-b border-primary outline-none text-sm px-0.5"
        />
      ) : (
        <Link
          href={`/dashboard/knowledge/${kb.id}`}
          onClick={onClose}
          className="flex items-center gap-2 flex-1 min-w-0"
        >
          <Database className="h-3.5 w-3.5 shrink-0 opacity-50" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{kb.name}</p>
            <p className="text-[10px] text-muted-foreground">
              {kb.documents?.length ?? 0} doc{(kb.documents?.length ?? 0) !== 1 ? "s" : ""} • {kb.data_source_count ?? 0} data store{(kb.data_source_count ?? 0) !== 1 ? "s" : ""}
            </p>
          </div>
        </Link>
      )}

      {!isEditing && (
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
          <button
            onClick={() => onStartEdit(kb.id, kb.name)}
            className="p-1 rounded hover:bg-muted-foreground/20 transition-colors"
            aria-label="Rename"
          >
            <Pencil className="h-3 w-3" />
          </button>
          <button
            onClick={() => onSetConfirmId(kb.id)}
            className="p-1 rounded hover:bg-destructive/20 hover:text-destructive transition-colors"
            aria-label="Delete"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}

export default function KnowledgeSidebar({ isOpen, onClose }: KnowledgeSidebarProps) {
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const { kbList, activeKbId, renameKb, deleteKb } = useKnowledgeContext();
  const { toast } = useToast();

  const { collapsed, toggleCollapse } = useSidebarCollapse("kb-sidebar-collapsed");

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId !== null) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editingId]);

  const handleDelete = useCallback(async (id: number) => {
    await deleteKb(id).catch((e) => {
      toast({ title: "Delete failed", description: e instanceof ApiError ? e.message : "Please try again", variant: "destructive" });
    });
    if (pathname === `/dashboard/knowledge/${id}`) {
      router.push("/dashboard/knowledge");
    }
  }, [deleteKb, toast, pathname, router]);

  const startEdit = useCallback((id: number, name: string) => {
    setEditingId(id);
    setEditingValue(name);
  }, [setEditingId, setEditingValue]);

  const commitEdit = useCallback(async (id: number) => {
    const trimmed = editingValue.trim();
    if (trimmed && trimmed !== kbList.find((k) => k.id === id)?.name) {
      await renameKb(id, trimmed).catch((e) => {
        toast({ title: "Rename failed", description: e instanceof ApiError ? e.message : "Please try again", variant: "destructive" });
      });
    }
    setEditingId(null);
  }, [editingValue, kbList, renameKb, toast, setEditingId]);

  const handleEditKeyDown = useCallback((e: React.KeyboardEvent, id: number) => {
    if (e.key === "Enter") commitEdit(id);
    if (e.key === "Escape") setEditingId(null);
  }, [commitEdit, setEditingId]);

  const filtered = filterKbList(kbList, searchQuery);

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
        {/* ── Collapsed ─── */}
        {collapsed && (
          <div className="flex flex-col items-center gap-2 py-3 flex-1">
            <button
              onClick={toggleCollapse}
              className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              aria-label="Expand sidebar"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* ── Expanded ─── */}
        {!collapsed && (
          <>
            {/* Header */}
            <div className="flex items-center gap-1 px-3 pt-3 pb-2 shrink-0">
              <Link
                href="/dashboard/knowledge/new"
                onClick={onClose}
                className="flex items-center gap-2 flex-1 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent transition-colors"
              >
                <Plus className="h-4 w-4 shrink-0" />
                New Knowledge Base
              </Link>
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
                  type="text"
                  placeholder="Search knowledge bases…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                />
              </div>
            </div>

            {/* List */}
            <nav className="flex-1 overflow-y-auto px-2 space-y-0.5">
              {filtered.length === 0 && (
                <p className="text-xs text-muted-foreground text-center mt-6 px-4">
                  {searchQuery ? "No matching knowledge bases." : "No knowledge bases yet."}
                </p>
              )}

              {filtered.map((kb) => {
                const isActive =
                  activeKbId === kb.id ||
                  pathname === `/dashboard/knowledge/${kb.id}` ||
                  pathname.startsWith(`/dashboard/knowledge/${kb.id}/`);

                return (
                  <KbListItem
                    key={kb.id}
                    kb={kb}
                    isActive={isActive}
                    isEditing={editingId === kb.id}
                    editingValue={editingValue}
                    inputRef={inputRef}
                    onEditingValueChange={setEditingValue}
                    onCommitEdit={commitEdit}
                    onEditKeyDown={handleEditKeyDown}
                    onStartEdit={startEdit}
                    onSetConfirmId={setConfirmId}
                    onClose={onClose}
                  />
                );
              })}
            </nav>
          </>
        )}
      </aside>
      <ConfirmDialog
        open={confirmId !== null}
        title="Delete knowledge base"
        description="Delete this knowledge base and all its documents?"
        confirmText="Delete"
        destructive
        onConfirm={() => {
          if (confirmId !== null) handleDelete(confirmId);
          setConfirmId(null);
        }}
        onCancel={() => setConfirmId(null)}
      />
    </>
  );
}
