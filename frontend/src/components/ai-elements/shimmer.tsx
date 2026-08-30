"use client";

import { cn } from "@/lib/utils";
import type { MotionProps } from "framer-motion";
import { motion } from "framer-motion";
import type { CSSProperties, ElementType } from "react";
import { memo, useMemo } from "react";

type MotionHTMLProps = MotionProps & Record<string, unknown>;

// Pre-create motion components at module level to avoid creating during render.
// framer-motion's `motion.p`, `motion.span`, etc. are already created — use them
// directly when possible. For custom element types, fall back to motion.create
// at module scope.
const motionP = motion.p;
const motionSpan = motion.span;
const motionDiv = motion.div;
const motionH1 = motion.h1;
const motionH2 = motion.h2;
const motionH3 = motion.h3;

const PRESET_MOTION: Partial<Record<string, React.ComponentType<MotionHTMLProps>>> = {
  p: motionP,
  span: motionSpan,
  div: motionDiv,
  h1: motionH1,
  h2: motionH2,
  h3: motionH3,
};

export interface TextShimmerProps {
  children: string;
  as?: ElementType;
  className?: string;
  duration?: number;
  spread?: number;
}

const ShimmerComponent = ({
  children,
  as: Component = "p",
  className,
  duration = 2,
  spread = 2,
}: TextShimmerProps) => {
  const MotionComponent = useMemo(() => {
    const tag = (typeof Component === "string" ? Component : "p") as string;
    return PRESET_MOTION[tag] ?? motionP;
  }, [Component]);

  const dynamicSpread = useMemo(
    () => (children?.length ?? 0) * spread,
    [children, spread]
  );

  return (
    <MotionComponent
      animate={{ backgroundPosition: "0% center" }}
      className={cn(
        "relative inline-block bg-[length:250%_100%,auto] bg-clip-text text-transparent",
        "[--bg:linear-gradient(90deg,#0000_calc(50%-var(--spread)),var(--color-background),#0000_calc(50%+var(--spread)))] [background-repeat:no-repeat,padding-box]",
        className
      )}
      initial={{ backgroundPosition: "100% center" }}
      style={
        {
          "--spread": `${dynamicSpread}px`,
          backgroundImage:
            "var(--bg), linear-gradient(var(--color-muted-foreground), var(--color-muted-foreground))",
        } as CSSProperties
      }
      transition={{
        duration,
        ease: "linear",
        repeat: Number.POSITIVE_INFINITY,
      }}
    >
      {children}
    </MotionComponent>
  );
};

ShimmerComponent.displayName = "Shimmer";

export const Shimmer = memo(ShimmerComponent);
