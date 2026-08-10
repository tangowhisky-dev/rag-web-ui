"use client";

import { useState, useRef, useEffect } from "react";
import { ChevronLeft, ChevronRight, Pencil, Check, X } from "lucide-react";
import { api } from "@/lib/api";

interface SiblingPair {
  user: { id: number; content: string; branch_index: number };
  assistant: { id: number; content: string; [key: string]: unknown } | null;
}

interface BranchPickerProps {
  /** The DB message id for the user message being shown */
  messageId: string;
  chatId: string;
  /** Current displayed content */
  content: string;
  /**
   * Called when the user submits an edit.
   * Receives the new branch message id and content so the parent can
   * swap the message in its local state and stream an assistant reply.
   */
  onBranch: (newMessageId: string, newContent: string) => void;
  /** Called when the user navigates to a different sibling branch.
   *  Receives the target user message, its paired assistant (or null),
   *  the current user message id, and the shared parent message id. */
  onNavigate: (
    targetUser: { id: string; content: string },
    targetAssistant: { id: string; content: string; [key: string]: unknown } | null,
    currentUserMsgId: string,
    parentMessageId: string,
  ) => void;
  disabled?: boolean;
}

/**
 * BranchPicker renders inline under a user message bubble.
 *
 * Idle state   → shows a pencil (edit) button.
 * Edit state   → shows an inline textarea + confirm/cancel.
 * After branch → shows ◀ N / M ▶ navigation when siblings exist.
 */
export function BranchPicker({
  messageId,
  chatId,
  content,
  onBranch,
  onNavigate,
  disabled = false,
}: BranchPickerProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(content);
  const [isSaving, setIsSaving] = useState(false);
  const [siblings, setSiblings] = useState<SiblingPair[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Keep draft in sync if content prop changes (e.g. after navigation)
  useEffect(() => {
    setDraft(content);
  }, [content]);

  // Fetch siblings on mount so navigation arrows appear for branched messages
  useEffect(() => {
    fetchSiblings(messageId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messageId]);

  // Auto-focus textarea when entering edit mode
  useEffect(() => {
    if (isEditing) {
      textareaRef.current?.focus();
      const len = draft.length;
      textareaRef.current?.setSelectionRange(len, len);
    }
  }, [isEditing]);

  const fetchSiblings = async (forMessageId: string) => {
    try {
      const data: SiblingPair[] = await api.get(
        `/api/chat/${chatId}/messages/${forMessageId}/siblings`
      );
      if (data.length > 1) {
        setSiblings(data);
        // Determine which sibling is currently displayed
        const idx = data.findIndex((s) => s.user.id.toString() === forMessageId);
        setCurrentIndex(idx >= 0 ? idx : data.length - 1);
      }
    } catch {
      // Silently ignore — siblings are optional UX enhancement
    }
  };

  const handleEdit = () => {
    setDraft(content);
    setIsEditing(true);
  };

  const handleCancel = () => {
    setDraft(content);
    setIsEditing(false);
  };

  const handleConfirm = async () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === content || isSaving) return;

    setIsSaving(true);
    try {
      const newMsg: { id: number; content: string; branch_index: number } =
        await api.patch(`/api/chat/${chatId}/messages/${messageId}`, {
          content: trimmed,
        });
      setIsEditing(false);
      onBranch(newMsg.id.toString(), newMsg.content);
      // Load siblings so navigation becomes available
      await fetchSiblings(newMsg.id.toString());
    } catch (err) {
      console.error("[BranchPicker] branch creation failed", err);
    } finally {
      setIsSaving(false);
    }
  };

  const handlePrev = () => {
    if (currentIndex <= 0) return;
    const prev = siblings[currentIndex - 1];
    setCurrentIndex(currentIndex - 1);
    const parentId = siblings[0].user.id.toString();
    onNavigate(
      { id: prev.user.id.toString(), content: prev.user.content },
      prev.assistant
        ? { ...prev.assistant, id: prev.assistant.id.toString(), content: prev.assistant.content }
        : null,
      messageId,
      parentId,
    );
  };

  const handleNext = () => {
    if (currentIndex >= siblings.length - 1) return;
    const next = siblings[currentIndex + 1];
    setCurrentIndex(currentIndex + 1);
    const parentId = siblings[0].user.id.toString();
    onNavigate(
      { id: next.user.id.toString(), content: next.user.content },
      next.assistant
        ? { ...next.assistant, id: next.assistant.id.toString(), content: next.assistant.content }
        : null,
      messageId,
      parentId,
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleConfirm();
    }
    if (e.key === "Escape") {
      handleCancel();
    }
  };

  if (isEditing) {
    return (
      <div className="mt-1 w-full max-w-[70%] ml-auto flex flex-col gap-1">
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={Math.max(2, draft.split("\n").length)}
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
          disabled={isSaving}
        />
        <div className="flex items-center justify-end gap-1">
          <button
            onClick={handleCancel}
            disabled={isSaving}
            title="Cancel (Esc)"
            className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-50"
          >
            <X className="h-3 w-3" />
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={isSaving || !draft.trim() || draft.trim() === content}
            title="Save & branch (⌘↵)"
            className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <Check className="h-3 w-3" />
            {isSaving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 mt-0.5">
      {/* Edit trigger — always visible in this area (parent controls hover) */}
      <button
        onClick={handleEdit}
        disabled={disabled}
        title="Edit message"
        className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-50"
      >
        <Pencil className="h-3 w-3" />
        Edit
      </button>

      {/* Branch navigation — only shown when siblings exist */}
      {siblings.length > 1 && (
        <>
          <button
            onClick={handlePrev}
            disabled={currentIndex <= 0}
            title="Previous branch"
            className="p-0.5 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-30"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <span className="text-xs text-muted-foreground tabular-nums">
            {currentIndex + 1} / {siblings.length}
          </span>
          <button
            onClick={handleNext}
            disabled={currentIndex >= siblings.length - 1}
            title="Next branch"
            className="p-0.5 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-30"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </>
      )}
    </div>
  );
}
