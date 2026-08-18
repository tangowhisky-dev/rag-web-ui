"use client";

import { useState, useRef, useEffect } from "react";
import { Paperclip, ArrowUp, Square, Plus, Mic } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { FileChip, useFileDropzone, type UploadedFile } from "./file-attachment";
import { useVoiceInput } from "./use-voice-input";

interface KbPill {
  id: number;
  name: string;
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
  /** KBs associated with this chat (from ChatResponse) */
  knowledgeBases?: KbPill[];
  /** Currently selected KB IDs (controlled from parent) */
  selectedKbIds?: number[];
  /** Called when user toggles a KB pill */
  onKbToggle?: (kbId: number) => void;
  /** When true, KB pills are disabled (PATCH in-flight) */
  kbToggling?: boolean;
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
  knowledgeBases = [],
  selectedKbIds = [],
  onKbToggle,
  kbToggling = false,
}: InputBarProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useAutoResize(textareaRef, value);

  const [internalError, setInternalError] = useState("");
  const activeError = onFileError !== undefined ? fileError : internalError;

  const { isListening, isSupported: voiceSupported, interim, start: startVoice, stop: stopVoice } = useVoiceInput(
    (text) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      onChange(value ? `${value} ${trimmed}` : trimmed);
    }
  );

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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isListening) { stopVoice(); return; }
      if (value.trim()) {
        onSubmit();
      }
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    // While listening, textarea shows value+interim; user edits should
    // only affect the committed value, not the interim preview.
    if (isListening) {
      onChange(e.target.value.slice(0, e.target.value.length - interim.length));
    } else {
      onChange(e.target.value);
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

      {/* Textarea + mic + send button — right-center aligned */}
      <div className="flex items-center px-3 pt-3 pb-1 gap-2">
        <textarea
          ref={textareaRef}
          value={isListening ? value + interim : value}
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
        {/* Mic button — voice-to-text (Chrome/Edge only) */}
        {voiceSupported && !disabled && (
          <button
            type="button"
            onClick={isListening ? stopVoice : startVoice}
            className={cn(
              "h-8 w-8 rounded-full flex items-center justify-center transition-all shrink-0",
              isListening
                ? "bg-red-500 text-white animate-pulse"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            )}
            aria-label={isListening ? "Stop voice input" : "Start voice input"}
            data-testid="chat-input-mic-button"
          >
            <Mic className="h-4 w-4" />
          </button>
        )}
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

      {/* Bottom row: attach + KB pills */}
      <div className="flex items-center justify-start px-2 pb-2 gap-2 flex-wrap">
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

        {/* KB pills — colored if associated, greyed if not; click to toggle */}
        {knowledgeBases.map((kb) => {
          const associated = selectedKbIds.includes(kb.id);
          return (
            <button
              key={kb.id}
              type="button"
              onClick={() => onKbToggle?.(kb.id)}
              disabled={kbToggling}
              className={cn(
                "h-7 rounded-full border px-2.5 text-xs shrink-0 transition-colors",
                associated
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-muted/40 text-muted-foreground/60 hover:bg-muted hover:text-muted-foreground",
                kbToggling && "animate-pulse cursor-not-allowed opacity-60"
              )}
              data-testid="chat-input-kb-pill"
            >
              {kb.name}
            </button>
          );
        })}

        {/* + button — link to KB creation page */}
        <Link
          href="/dashboard/knowledge"
          className="h-7 w-7 rounded-full border border-dashed border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-foreground transition-colors shrink-0"
          aria-label="Create new knowledge base"
          data-testid="chat-input-kb-add"
        >
          <Plus className="h-3.5 w-3.5" />
        </Link>
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
