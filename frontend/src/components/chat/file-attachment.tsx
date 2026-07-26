"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Paperclip, X, FileText, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export const MAX_FILE_SIZE = 10 * 1024 * 1024;

export const SUPPORTED_TYPES: Record<string, string[]> = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/msword": [".doc"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
  "application/vnd.ms-powerpoint": [".ppt"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-excel": [".xls"],
  "text/plain": [".txt"],
  "text/markdown": [".md"],
  "text/html": [".html", ".htm"],
  "text/csv": [".csv"],
  "application/json": [".json"],
  "application/xml": [".xml"],
  "text/xml": [".xml"],
  "message/rfc822": [".eml"],
  "application/epub+zip": [".epub"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/gif": [".gif"],
  "application/zip": [".zip"],
};

export interface UploadedFile {
  id: number;
  file_name: string;
  file_size: number;
  status: "processing" | "ready" | "error";
  error_message?: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function truncateFilename(name: string, maxLen = 28): string {
  if (name.length <= maxLen) return name;
  const ext = name.lastIndexOf(".");
  if (ext > 0) {
    const base = name.slice(0, ext);
    const suffix = name.slice(ext);
    return base.slice(0, maxLen - suffix.length - 1) + "…" + suffix;
  }
  return name.slice(0, maxLen - 1) + "…";
}

// ── FileChip: shows upload status ────────────────────────────────────────────

interface FileChipProps {
  uploadedFile: UploadedFile;
  onRemove: () => void;
}

export function FileChip({ uploadedFile, onRemove }: FileChipProps) {
  const { status, file_name, file_size, error_message } = uploadedFile;
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs max-w-[280px]",
        status === "error"
          ? "bg-destructive/10 text-destructive border border-destructive/30"
          : "bg-secondary text-secondary-foreground"
      )}
      data-testid="file-chip"
      title={error_message || file_name}
    >
      {status === "processing" && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-muted-foreground" />}
      {status === "ready" && <CheckCircle2 className="h-3 w-3 shrink-0 text-green-500" />}
      {status === "error" && <AlertCircle className="h-3 w-3 shrink-0" />}
      {status === "processing" || status === "ready" ? (
        <FileText className="h-3 w-3 shrink-0 text-muted-foreground" />
      ) : null}
      <span className="truncate">{truncateFilename(file_name)}</span>
      <span className="text-muted-foreground shrink-0">({formatBytes(file_size)})</span>
      <button
        type="button"
        onClick={onRemove}
        className="ml-0.5 shrink-0 rounded-sm hover:bg-muted p-0.5 transition-colors"
        aria-label={`Remove ${file_name}`}
        data-testid="file-chip-remove"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

// ── MessageFileChip: clickable download icon shown under a sent message ─────

interface MessageFileChipProps {
  fileName: string;
  fileId: number;
  chatId: string | number;
}

export function MessageFileChip({ fileName, fileId, chatId }: MessageFileChipProps) {
  const handleDownload = async () => {
    const res = await fetch(`/api/chat/${chatId}/files/${fileId}/download`, {
      credentials: "include",
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      type="button"
      onClick={handleDownload}
      title={fileName}
      className="group inline-flex items-center gap-1 mt-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
    >
      <span className="max-w-[200px] truncate opacity-0 group-hover:opacity-100 transition-opacity">
        {fileName}
      </span>
      <Paperclip className="h-3 w-3 shrink-0 group-hover:text-primary transition-colors" />
    </button>
  );
}

// ── FileDropzone hook ─────────────────────────────────────────────────────────

interface UseFileDropzoneOptions {
  onFileAccepted: (file: File) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

export function useFileDropzone({ onFileAccepted, onError, disabled = false }: UseFileDropzoneOptions) {
  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: { file: File; errors: ReadonlyArray<{ code: string }> }[]) => {
      if (rejectedFiles.length > 0) {
        const err = rejectedFiles[0].errors[0];
        if (err.code === "file-too-large") onError("File exceeds 10 MB limit.");
        else if (err.code === "file-invalid-type") onError("Unsupported file type.");
        else onError(`File rejected: ${err.code}`);
        return;
      }
      if (acceptedFiles.length > 0) {
        onError("");
        onFileAccepted(acceptedFiles[0]);
      }
    },
    [onFileAccepted, onError]
  );

  return useDropzone({
    onDrop, accept: SUPPORTED_TYPES, maxFiles: 1, maxSize: MAX_FILE_SIZE,
    disabled, noClick: true, noKeyboard: true,
  });
}

// ── FileAttachButton ──────────────────────────────────────────────────────────

interface FileAttachButtonProps {
  onFileAccepted: (file: File) => void;
  onError: (message: string) => void;
  disabled?: boolean;
  isDragActive?: boolean;
}

export function FileAttachButton({ onFileAccepted, onError, disabled = false, isDragActive = false }: FileAttachButtonProps) {
  const { getInputProps, open } = useFileDropzone({ onFileAccepted, onError, disabled });
  return (
    <>
      <input {...getInputProps()} data-testid="file-input" />
      <button
        type="button" onClick={open} disabled={disabled} title="Attach a file"
        className={cn(
          "h-8 w-8 flex items-center justify-center rounded-md transition-colors shrink-0",
          "text-muted-foreground hover:text-foreground hover:bg-accent",
          "disabled:cursor-not-allowed disabled:opacity-40",
          isDragActive && "text-primary bg-accent"
        )}
        data-testid="chat-input-file-button"
      >
        <Paperclip className="h-4 w-4" />
      </button>
    </>
  );
}
