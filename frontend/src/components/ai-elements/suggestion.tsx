"use client";

import { cn } from "@/lib/utils";
import type { ComponentProps } from "react";
import { useCallback } from "react";

export type SuggestionsProps = ComponentProps<"div">;

export const Suggestions = ({
  className,
  children,
  ...props
}: SuggestionsProps) => (
  <div className={cn("flex flex-wrap items-center gap-2", className)} {...props}>
    {children}
  </div>
);

export type SuggestionProps = Omit<ComponentProps<"button">, "onClick"> & {
  suggestion: string;
  onClick?: (suggestion: string) => void;
};

export const Suggestion = ({
  suggestion,
  onClick,
  className,
  children,
  ...props
}: SuggestionProps) => {
  const handleClick = useCallback(() => {
    onClick?.(suggestion);
  }, [onClick, suggestion]);

  return (
    <button
      className={cn(
        "text-xs text-muted-foreground hover:text-foreground",
        "border border-border/60 hover:border-border",
        "rounded-lg px-3 py-1.5 transition-colors text-left cursor-pointer",
        className,
      )}
      onClick={handleClick}
      type="button"
      {...props}
    >
      {children || suggestion}
    </button>
  );
};
