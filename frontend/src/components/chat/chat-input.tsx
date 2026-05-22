"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Loader2, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { FileAttachButton, FileChip, useFileDropzone } from "./file-attachment";

interface KnowledgeBase {
  id: number;
  name: string;
  description?: string;
}

interface InputBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  placeholder?: string;
  /** Currently attached file (controlled from parent) */
  file?: File | null;
  /** Called when the user picks or clears a file */
  onFileChange?: (file: File | null) => void;
  /** Inline error message (oversized / unsupported type) */
  fileError?: string;
  /** Setter for fileError (parent can clear it) */
  onFileError?: (msg: string) => void;
}

const LINES_MIN = 1;
const LINES_MAX = 5;
const LINE_HEIGHT_PX = 24; // matches textarea line-height

function useAutoResize(value: string) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const hiddenSpanRef = useRef<HTMLSpanElement>(null);

  const resize = useCallback(() => {
    const ta = textareaRef.current;
    const span = hiddenSpanRef.current;
    if (!ta || !span) return;

    // Keep textarea and span in sync for measurement
    span.style.fontSize = getComputedStyle(ta).fontSize;
    span.style.fontFamily = getComputedStyle(ta).fontFamily;
    span.style.padding = getComputedStyle(ta).padding;
    span.style.boxSizing = getComputedStyle(ta).boxSizing;
    span.style.width = getComputedStyle(ta).width;
    span.textContent = value || "\u00A0"; // non-breaking space preserves height when empty

    const neededLines = Math.ceil(span.scrollHeight / LINE_HEIGHT_PX);
    const clampedLines = Math.max(LINES_MIN, Math.min(neededLines, LINES_MAX));
    ta.style.height = `${clampedLines * LINE_HEIGHT_PX}px`;

    if (neededLines > LINES_MAX) {
      ta.style.overflowY = "auto";
      ta.style.maxHeight = `${LINES_MAX * LINE_HEIGHT_PX}px`;
    } else {
      ta.style.overflowY = "hidden";
      ta.style.maxHeight = "";
    }
  }, [value]);

  useEffect(() => {
    resize();
  }, [resize]);

  return { textareaRef, hiddenSpanRef };
}

export function InputBar({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder = "Type your message...",
  file = null,
  onFileChange,
  fileError = "",
  onFileError,
}: InputBarProps) {
  const [kbOpen, setKbOpen] = useState(false);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKbIds, setSelectedKbIds] = useState<number[]>([]);
  const { textareaRef, hiddenSpanRef } = useAutoResize(value);

  // Internal file state when used uncontrolled
  const [internalFile, setInternalFile] = useState<File | null>(null);
  const [internalError, setInternalError] = useState("");

  const activeFile = onFileChange !== undefined ? file : internalFile;
  const activeError = onFileError !== undefined ? fileError : internalError;

  const handleFileAccepted = (f: File) => {
    if (onFileChange) onFileChange(f);
    else setInternalFile(f);
    if (onFileError) onFileError("");
    else setInternalError("");
  };

  const handleFileError = (msg: string) => {
    if (onFileError) onFileError(msg);
    else setInternalError(msg);
  };

  const handleFileRemove = () => {
    if (onFileChange) onFileChange(null);
    else setInternalFile(null);
    if (onFileError) onFileError("");
    else setInternalError("");
  };

  // Drag-and-drop on the whole input bar container
  const { getRootProps, isDragActive } = useFileDropzone({
    onFileAccepted: handleFileAccepted,
    onError: handleFileError,
    disabled,
  });

  // Fetch KBs on mount
  useEffect(() => {
    let cancelled = false;
    const fetchKbs = async () => {
      try {
        const data = await api.get("/api/knowledge-base");
        if (!cancelled) {
          const kbs = Array.isArray(data) ? data : data.items || [];
          setKnowledgeBases(kbs);
          // Default: select all KBs
          setSelectedKbIds(kbs.map((kb: KnowledgeBase) => kb.id));
        }
      } catch {
        // KB endpoint may not exist yet in dev — silently ignore
        if (!cancelled) {
          setKnowledgeBases([]);
          setSelectedKbIds([]);
        }
      }
    };
    fetchKbs();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (value.trim()) {
        onSubmit();
      }
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };

  const handleKbToggle = (kbId: number, checked: boolean) => {
    if (checked) {
      setSelectedKbIds((prev) => [...prev, kbId]);
    } else {
      setSelectedKbIds((prev) => prev.filter((id) => id !== kbId));
    }
  };

  return (
    <div className="flex flex-col gap-2" data-testid="chat-input-container">
      {/* Hidden span for textarea measurement */}
      <span
        ref={hiddenSpanRef}
        className="absolute invisible whitespace-pre-wrap break-words"
        style={{
          width: "100%",
          lineHeight: `${LINE_HEIGHT_PX}px`,
        }}
      />

      {/* File chip row */}
      {activeFile && (
        <div className="px-1">
          <FileChip file={activeFile} onRemove={handleFileRemove} />
        </div>
      )}

      <div
        {...getRootProps()}
        className={cn(
          "flex items-end gap-2",
          isDragActive && "ring-2 ring-primary ring-offset-1 rounded-md"
        )}
      >
        {/* KB Selector */}
        {knowledgeBases.length > 0 && (
          <Select open={kbOpen} onOpenChange={setKbOpen}>
            <SelectTrigger
              className="h-8 w-[140px] rounded-md text-xs shrink-0"
              data-testid="chat-input-kb-selector"
            >
              <SelectValue
                placeholder="Select KBs"
              />
            </SelectTrigger>
            <SelectContent>
              {knowledgeBases.map((kb) => (
                <SelectItem
                  key={kb.id}
                  value={String(kb.id)}
                  onSelect={(e) => {
                    e.preventDefault();
                    handleKbToggle(kb.id, !selectedKbIds.includes(kb.id));
                  }}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={selectedKbIds.includes(kb.id)}
                      onChange={() => {}}
                      className="h-3 w-3"
                    />
                    <span className="truncate max-w-[160px]">{kb.name}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {/* File Attachment Button */}
        <FileAttachButton
          onFileAccepted={handleFileAccepted}
          onError={handleFileError}
          disabled={disabled}
          isDragActive={isDragActive}
        />

        {/* Textarea + Send Button Container */}
        <div className="flex-1 flex items-end gap-2 rounded-md border border-input bg-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={LINES_MIN}
            className={cn(
              "flex-1 min-w-0 resize-none rounded-l-md border-0 bg-transparent px-3 py-2 text-sm",
              "focus:outline-none focus:ring-0 disabled:cursor-not-allowed disabled:opacity-50",
              "placeholder:text-muted-foreground"
            )}
            style={{
              height: `${LINES_MIN * LINE_HEIGHT_PX}px`,
              lineHeight: `${LINE_HEIGHT_PX}px`,
            }}
            data-testid="chat-input-textarea"
          />

          {/* Send Button */}
          <button
            type="button"
            onClick={onSubmit}
            disabled={disabled || !value.trim()}
            className={cn(
              "h-10 rounded-r-md px-3 flex items-center justify-center transition-all",
              "bg-primary text-primary-foreground hover:bg-primary/90",
              "disabled:pointer-events-none disabled:opacity-50"
            )}
            data-testid="chat-input-send-button"
          >
            {disabled ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {/* File error */}
      {activeError && (
        <p className="text-xs text-destructive px-1" data-testid="file-error">
          {activeError}
        </p>
      )}

      {/* Selected KBs indicator */}
      {knowledgeBases.length > 0 && selectedKbIds.length > 0 && (
        <div className="flex flex-wrap gap-1 px-1">
          {selectedKbIds.map((id) => {
            const kb = knowledgeBases.find((k) => k.id === id);
            if (!kb) return null;
            return (
              <span
                key={id}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-secondary text-secondary-foreground"
              >
                {kb.name}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
