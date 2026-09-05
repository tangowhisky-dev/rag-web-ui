"use client";

import { FileText, Presentation, Sheet, Download } from "lucide-react";
import { api } from "@/lib/api";

interface GeneratedFileChipProps {
  file: {
    file_id: number;
    file_name: string;
    format: string;
    title?: string;
    summary?: string;
    slide_count?: number;
    chart_count?: number;
  };
  chatId: string | number;
}

const FORMAT_ICONS: Record<string, typeof FileText> = {
  pptx: Presentation,
  docx: FileText,
  xlsx: Sheet,
};

const FORMAT_LABELS: Record<string, string> = {
  pptx: "PowerPoint",
  docx: "Word",
  xlsx: "Excel",
};

export function GeneratedFileChip({ file, chatId }: GeneratedFileChipProps) {
  const Icon = FORMAT_ICONS[file.format] ?? FileText;
  const label = FORMAT_LABELS[file.format] ?? file.format.toUpperCase();

  const handleDownload = async () => {
    try {
      const res = await api.getRaw(`/api/chat/${chatId}/files/${file.file_id}/download`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.file_name;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // best-effort download
    }
  };

  const meta: string[] = [label];
  if (file.slide_count) meta.push(`${file.slide_count} slides`);
  if (file.chart_count) meta.push(`${file.chart_count} charts`);

  return (
    <button
      type="button"
      onClick={handleDownload}
      title={file.summary || file.file_name}
      className="group inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm hover:bg-accent transition-colors"
    >
      <Icon className="h-4 w-4 shrink-0 text-primary" />
      <div className="flex flex-col items-start text-left">
        <span className="font-medium leading-tight">{file.title || file.file_name}</span>
        <span className="text-xs text-muted-foreground leading-tight">{meta.join(" · ")}</span>
      </div>
      <Download className="h-4 w-4 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
    </button>
  );
}
