"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HelpCircle, Search, Scissors, MessageSquarePlus } from "lucide-react";

/**
 * SelectionActions — floating toolbar that appears when the user selects
 * text inside the answer content. Offers contextual actions that feed
 * the selected passage back into the chat as a follow-up query.
 *
 * No backend changes required: the parent provides an `onAction` callback
 * that receives the selected text + action type. The parent decides how
 * to route it (typically: set input value + auto-submit).
 *
 * The toolbar positions itself above the selection using the browser's
 * getBoundingClientRect. It dismisses on selection clear, scroll, or
 * click outside.
 */

type ActionType = "explain" | "find_sources" | "summarize" | "follow_up";

const ACTIONS: Array<{ type: ActionType; label: string; icon: typeof HelpCircle }> = [
  { type: "explain", label: "Explain", icon: HelpCircle },
  { type: "find_sources", label: "Find sources", icon: Search },
  { type: "summarize", label: "Summarize", icon: Scissors },
  { type: "follow_up", label: "Ask", icon: MessageSquarePlus },
];

const ACTION_PROMPTS: Record<ActionType, (text: string) => string> = {
  explain: (t) => `Explain this in more detail:\n\n> ${t}`,
  find_sources: (t) => `Find sources relevant to this passage:\n\n> ${t}`,
  summarize: (t) => `Summarize this concisely:\n\n> ${t}`,
  follow_up: (t) => t,
};

export function SelectionActions({
  containerRef,
  onAction,
  disabled,
}: {
  containerRef: React.RefObject<HTMLElement>;
  onAction: (query: string) => void;
  disabled?: boolean;
}) {
  const [selection, setSelection] = useState<{ text: string; rect: DOMRect } | null>(null);
  const toolbarRef = useRef<HTMLDivElement>(null);

  const handleSelectionChange = useCallback(() => {
    if (disabled) {
      setSelection(null);
      return;
    }

    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      setSelection(null);
      return;
    }

    const text = sel.toString().trim();
    if (text.length < 3) {
      setSelection(null);
      return;
    }

    // Only react if the selection is within the container
    const container = containerRef.current;
    if (!container) return;
    const range = sel.getRangeAt(0);
    if (!container.contains(range.commonAncestorContainer)) {
      setSelection(null);
      return;
    }

    const rect = range.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) {
      setSelection(null);
      return;
    }

    setSelection({ text, rect });
  }, [containerRef, disabled]);

  // Check selection on mouseup + keyup (selection can change via keyboard)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onMouseUp = () => {
      // Small delay to let the browser finalize the selection
      setTimeout(handleSelectionChange, 10);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.shiftKey || e.key === "Shift") {
        setTimeout(handleSelectionChange, 10);
      }
    };

    container.addEventListener("mouseup", onMouseUp);
    container.addEventListener("keyup", onKeyUp);
    return () => {
      container.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("keyup", onKeyUp);
    };
  }, [containerRef, handleSelectionChange]);

  // Dismiss on scroll or resize
  useEffect(() => {
    if (!selection) return;
    const dismiss = () => setSelection(null);
    const scrollContainer = containerRef.current?.closest(".overflow-y-auto");
    if (scrollContainer) {
      scrollContainer.addEventListener("scroll", dismiss, { passive: true });
    }
    window.addEventListener("scroll", dismiss, { passive: true });
    window.addEventListener("resize", dismiss);
    return () => {
      if (scrollContainer) {
        scrollContainer.removeEventListener("scroll", dismiss);
      }
      window.removeEventListener("scroll", dismiss);
      window.removeEventListener("resize", dismiss);
    };
  }, [selection, containerRef]);

  // Dismiss on click outside the toolbar
  useEffect(() => {
    if (!selection) return;
    const onPointerDown = (e: PointerEvent) => {
      if (toolbarRef.current && !toolbarRef.current.contains(e.target as Node)) {
        setSelection(null);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [selection]);

  const handleAction = (type: ActionType) => {
    if (!selection) return;
    const prompt = ACTION_PROMPTS[type](selection.text);
    onAction(prompt);
    // Clear the browser selection
    window.getSelection()?.removeAllRanges();
    setSelection(null);
  };

  if (!selection) return null;

  // Position the toolbar above the selection, centered horizontally
  const top = selection.rect.top + window.scrollY - 44;
  const left = selection.rect.left + window.scrollX + selection.rect.width / 2;

  return (
    <AnimatePresence>
      <motion.div
        ref={toolbarRef}
        initial={{ opacity: 0, y: 4, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 4, scale: 0.96 }}
        transition={{ duration: 0.15, ease: "easeOut" }}
        style={{
          position: "absolute",
          top: `${top}px`,
          left: `${left}px`,
          transform: "translateX(-50%)",
          zIndex: 50,
        }}
        className="flex items-center gap-0.5 rounded-lg border border-border bg-popover shadow-md py-1 px-1"
        onPointerDown={(e) => e.stopPropagation()}
      >
        {ACTIONS.map(({ type, label, icon: Icon }) => (
          <button
            key={type}
            onClick={() => handleAction(type)}
            title={label}
            className="flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <Icon className="h-3 w-3" />
            {label}
          </button>
        ))}
      </motion.div>
    </AnimatePresence>
  );
}
