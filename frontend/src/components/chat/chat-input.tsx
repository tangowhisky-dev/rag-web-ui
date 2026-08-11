"use client";

import { useState, useRef, useEffect } from "react";
import { Paperclip, ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { FileChip, useFileDropzone, type UploadedFile } from "./file-attachment";

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
  /** Uploaded file record (processing/ready/error) */
  uploadedFile?: UploadedFile | null;
  /** Called when user selects a file to upload */
  onFileAccepted?: (file: File) => void;
  /** Called when user removes the file chip */
  onFileRemove?: () => void;
  /** Inline error message */
  fileError?: string;
  /** Setter for fileError */
  onFileError?: (msg: string) => void;
  /** Called when user clicks stop during generation */
  onStop?: () => void;
}

const LINE_HEIGHT_PX = 24;
const MIN_HEIGHT_PX = 2 * LINE_HEIGHT_PX;  // 2 lines default
const MAX_HEIGHT_PX = 10 * LINE_HEIGHT_PX; // 10 lines max

function useAutoResize(ref: React.RefObject<HTMLTextAreaElement>, value: string) {
  useEffect(() => {
    const ta = ref.current;
    if (!ta) return;
    const rafId = requestAnimationFrame(() => {
      ta.style.height = "auto";
      ta.style.height = `${Math.min(Math.max(ta.scrollHeight, MIN_HEIGHT_PX), MAX_HEIGHT_PX)}px`;
    });
    return () => cancelAnimationFrame(rafId);
  }, [value, ref]);
}

export function InputBar({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder = "Type your message...",
  uploadedFile = null,
  onFileAccepted: onFileAcceptedProp,
  onFileRemove: onFileRemoveProp,
  fileError = "",
  onFileError,
  onStop,
}: InputBarProps) {
  const [kbOpen, setKbOpen] = useState(false);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKbIds, setSelectedKbIds] = useState<number[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useAutoResize(textareaRef, value);

  const [internalError, setInternalError] = useState("");
  const activeError = onFileError !== undefined ? fileError : internalError;

  const handleFileAccepted = (f: File) => {
    if (onFileAcceptedProp) onFileAcceptedProp(f);
    if (onFileError) onFileError(""); else setInternalError("");
  };

  const handleFileError = (msg: string) => {
    if (onFileError) onFileError(msg); else setInternalError(msg);
  };

  const handleFileRemove = () => {
    if (onFileRemoveProp) onFileRemoveProp();
    if (onFileError) onFileError(""); else setInternalError("");
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
    <div
      {...getRootProps()}
      className={cn(
        "flex flex-col rounded-2xl border border-border bg-background/80 backdrop-blur-sm shadow-md",
        isDragActive && "ring-2 ring-primary ring-offset-1"
      )}
      data-testid="chat-input-container"
    >
      {uploadedFile && (
        <div className="px-3 pt-3">
          <FileChip uploadedFile={uploadedFile} onRemove={handleFileRemove} />
        </div>
      )}

      {/* Textarea + send button — right-center aligned */}
      <div className="flex items-center px-3 pt-3 pb-1 gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className={cn(
            "flex-1 resize-none border-0 bg-transparent text-sm",
            "focus:outline-none focus:ring-0 disabled:cursor-not-allowed disabled:opacity-50",
            "placeholder:text-muted-foreground overflow-y-auto"
          )}
          style={{ lineHeight: `${LINE_HEIGHT_PX}px`, height: `${LINE_HEIGHT_PX}px` }}
          data-testid="chat-input-textarea"
        />
        {/* Send / Stop button — right-center of input */}
        {disabled ? (
          <button
            type="button"
            onClick={onStop}
            className="h-8 w-8 rounded-full flex items-center justify-center transition-all shrink-0 bg-destructive text-destructive-foreground hover:bg-destructive/90"
            aria-label="Stop generation"
            data-testid="chat-input-stop-button"
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={onSubmit}
            disabled={!value.trim() || uploadedFile?.status === "processing"}
            className={cn(
              "h-8 w-8 rounded-full flex items-center justify-center transition-all shrink-0",
              !value.trim()
                ? "bg-muted text-muted-foreground"
                : "bg-primary text-primary-foreground hover:bg-primary/90",
              "disabled:pointer-events-none"
            )}
            data-testid="chat-input-send-button"
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Bottom row: attach + KB chips */}
      <div className="flex items-center justify-start px-2 pb-2 gap-2">
        {/* File attach button */}
        <label
          className={cn(
            "p-1.5 rounded-lg cursor-pointer text-muted-foreground hover:text-foreground hover:bg-muted transition-colors",
            disabled && "pointer-events-none opacity-50"
          )}
          aria-label="Attach file"
          data-testid="chat-input-file-button"
        >
          <Paperclip className="h-4 w-4" />
          <input
            type="file"
            className="hidden"
            disabled={disabled}
            accept=".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.txt,.md,.html,.htm,.csv,.json,.xml,.eml,.epub,.jpg,.jpeg,.png,.gif,.zip"
            data-testid="file-input"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                handleFileAccepted(file);
              }
              e.target.value = "";  // reset so same file can be re-selected
            }}
          />
        </label>

        {/* KB selector pills */}
        {knowledgeBases.length > 0 && (
          <Select open={kbOpen} onOpenChange={setKbOpen}>
            <SelectTrigger
              className="h-7 w-auto gap-1 rounded-full border px-2.5 text-xs shrink-0"
              data-testid="chat-input-kb-selector"
            >
              <SelectValue placeholder="KBs" />
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
      </div>

      {/* File error */}
      {activeError && (
        <p className="text-xs text-destructive px-3 pb-2" data-testid="file-error">
          {activeError}
        </p>
      )}
    </div>
  );
}
