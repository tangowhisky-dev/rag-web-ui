"use client";

import { useEffect, useRef, useState } from "react";

/**
 * LoadingDots — three-dot shimmer loader with optional elapsed-time counter.
 *
 * Used as a drop-in replacement for CSS border-spinners. The shimmer
 * animation is GPU-only (opacity + scale), honors prefers-reduced-motion
 * via the global rule in globals.css, and the elapsed counter is opt-in.
 */
export function LoadingDots({
  label,
  showElapsed = false,
  size = "md",
}: {
  label?: string;
  showElapsed?: boolean;
  size?: "sm" | "md";
}) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    if (!showElapsed) return;
    const interval = setInterval(() => {
      setElapsed(Date.now() - startRef.current);
    }, 100);
    return () => clearInterval(interval);
  }, [showElapsed]);

  const dotSize = size === "sm" ? "w-1 h-1" : "w-1.5 h-1.5";
  const gap = size === "sm" ? "gap-1" : "gap-1.5";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className={`flex items-center ${gap}`}>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={`${dotSize} rounded-full bg-muted-foreground/60`}
            style={{
              animation: `loading-dot-pulse 1.4s ease-in-out ${i * 0.16}s infinite`,
            }}
          />
        ))}
      </div>
      {label && (
        <p className="text-xs text-muted-foreground tabular-nums">
          {label}
          {showElapsed && elapsed > 0 && (
            <span className="ml-1 text-muted-foreground/60">
              {elapsed < 1000 ? `${elapsed}ms` : `${(elapsed / 1000).toFixed(1)}s`}
            </span>
          )}
        </p>
      )}
    </div>
  );
}
