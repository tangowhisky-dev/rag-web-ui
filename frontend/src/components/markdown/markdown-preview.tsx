"use client";

import { memo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import { cn } from "@/lib/utils";

// NOTE: rehype-raw is intentionally NOT used. Admin-edited markdown can
// contain arbitrary HTML; we do not render it unsanitized. GFM + math +
// code highlighting are sufficient for the editor preview.

interface MarkdownPreviewProps {
  markdown: string;
  className?: string;
}

export const MarkdownPreview = memo(function MarkdownPreview({
  markdown,
  className,
}: MarkdownPreviewProps) {
  return (
    <Markdown
      className={cn("prose prose-sm dark:prose-invert max-w-none", className)}
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeHighlight, rehypeKatex]}
    >
      {markdown}
    </Markdown>
  );
});
