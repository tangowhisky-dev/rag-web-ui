# Open WebUI Pattern Migration Plan

> Date: 2026-07-09 (updated)
> Source: open-webui (`/Users/tango16/code/open-webui`)
> Target: rag-web-ui (this project)
> Scope: Skeleton cursor, flushSync removal, and compact progress badges.

---

## Overview

This plan migrates three UI/UX patterns from Open WebUI into rag-web-ui. Phases 2 (word-level fade-in) and 4 (Socket.IO dual transport) were removed after codebase audit — they add complexity without proportional value.

| # | Pattern | What changes | Impact |
|---|---------|-------------|--------|
| 1 | **Skeleton cursor** | Replace 3 bouncing dots with pulsing concentric circles | UI only, no API change |
| 2 | **flushSync removal** | Remove `flushSync` wrapper on `1:`/`2:` SSE events | Performance win, enables natural React batching |
| 3 | **Compact progress badges** | Add Open WebUI-style compact status badges with shimmer to `AgentTimeline` | UI only, reuses existing SSE `4:` events |

---

## Phase 1 — Skeleton Cursor

### 1.1 Goal

Replace the 3 bouncing dots with a pulsing concentric-circle skeleton that matches Open WebUI's two-ring pattern.

### 1.2 Current code

**File:** `frontend/src/app/dashboard/chat/[id]/page.tsx` lines 1040-1045

```tsx
{isLoading && !message.content && !message.rewrittenQuery ? (
  <div className="flex items-center gap-1 py-2">
    <div className="w-2 h-2 rounded-full bg-primary animate-bounce" />
    <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.2s]" />
    <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.4s]" />
  </div>
) : (
```

### 1.3 Design

Two concentric circles:
- **Outer ring:** `animate-pulse` — opacity 0.25→0.75, scale 0.75→1, 2s ease-in-out infinite
- **Inner ring:** `animate-size` — scale 1→1.25, 1.5s ease-in-out infinite
- Dark mode variants with muted ring colors

### 1.4 Implementation steps

1. **Add keyframe to `globals.css`:**
   ```css
   @keyframes skeleton-pulse {
     0%, 100% { opacity: 0.25; transform: scale(0.75); }
     50% { opacity: 0.75; transform: scale(1); }
   }
   @keyframes skeleton-size {
     0%, 100% { transform: scale(1); }
     50% { transform: scale(1.25); }
   }
   ```

2. **Replace inline JSX** in `page.tsx` lines 1040-1045:

   ```tsx
   {isLoading && !message.content && !message.rewrittenQuery ? (
     <div className="flex items-center justify-center py-2" aria-label="Generating response…">
       <div className="relative w-5 h-5">
         <div className="absolute inset-0 rounded-full bg-primary animate-pulse dark:bg-muted-foreground/40" />
         <div className="absolute inset-0 rounded-full bg-primary/60 dark:bg-muted-foreground/30"
              style={{ animation: 'skeleton-size 1.5s ease-in-out infinite' }} />
       </div>
     </div>
   ) : (
   ```

3. **Remove the old `animate-bounce` dots.**

### 1.5 Verification

- Page loads, type a message, observe the pulsing circles appear instead of bouncing dots
- Dark mode: circles are visible against dark background
- Reduced motion preference (`prefers-reduced-motion: reduce`) disables the animation (already handled by `globals.css`)

---

## Phase 2 — flushSync Removal

### 2.1 Goal

Remove `flushSync` wrapper on `1:`/`2:` SSE events to enable natural React batching, improving streaming smoothness.

### 2.2 Current code

**File:** `frontend/src/app/dashboard/chat/[id]/page.tsx` line 728

```tsx
flushSync(() => processStreamLine(line, assistantId));
```

### 2.3 Implementation

1. **Remove `flushSync` wrapper.**

   Change:
   ```tsx
   if (t.startsWith("1:") || t.startsWith("2:")) {
     flushSync(() => processStreamLine(line, assistantId));
   }
   ```

   To:
   ```tsx
   if (t.startsWith("1:") || t.startsWith("2:")) {
     processStreamLine(line, assistantId);
   }
   ```

2. **Remove unused `flushSync` import.**

   Remove: `import { flushSync } from "react-dom";` from line 16 (if no longer used elsewhere).

3. **Keep `flushToBrowser` scroll** — already uses `requestAnimationFrame` (line 407-421), no change needed.

### 2.4 Verification

- Stream a response — tokens still appear in correct order
- Page layout is smooth (no jank) during fast streaming
- Agent steps (`4:`) and context (`1:`, `2:`) render correctly
- Auto-scroll still works via `requestAnimationFrame` throttle

### 3.1 Goal

Add Open WebUI-style compact progress badges with shimmer animation to `AgentTimeline`. Badges appear during streaming, persist after completion with status indicators. Expandable detail rows stay below for rich metadata (rewritten query, retrieval context, tool traces).

### 3.2 Current code

**File:** `frontend/src/components/chat/agent-timeline.tsx` (514 lines)

- Rich `TimelineStepRow` with expandable detail panels
- `NODE_META` mapping with 15+ node types
- Icons, latency badges, active/done/error colors
- Already receives live SSE `4:` events and post-stream metadata

No new files needed. Badges render inline in the existing component.

### 3.3 Design

Compact single-line badges (not expandable) for high-level phases. Expandable detail rows remain below for rich metadata.

```
┌──────────────────────────────┐
│ 🔍 Searching knowledge bases… │  ← compact badge (shimmer)
│ ✅ Query rewritten            │  ← compact badge (done)
│ 🧠 Drafting answer…           │  ← compact badge (active)
│ ──────────────────────────── │
│ ┌──────────────────────────┐ │
│ │ Expanded detail section  │ │  ← expandable row
│ └──────────────────────────┘ │
└──────────────────────────────┘
```

### 3.4 Implementation steps

1. **Add shimmer keyframe to `globals.css`:**
   ```css
   @keyframes shimmer {
     0% { background-position: 200% 0; }
     100% { background-position: -200% 0; }
   }
   .status-shimmer {
     background-image: linear-gradient(
       90deg,
       rgba(255,255,255,0) 0%,
       rgba(255,255,255,0.08) 20%,
       rgba(255,255,255,0.16) 60%,
       rgba(255,255,255,0)
     );
     background-size: 200% 100%;
     background-repeat: no-repeat;
     animation: shimmer 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
   }
   ```

2. **Add compact badge rows inline in `AgentTimeline`.** Badge styles:
   - `flex items-center gap-2 px-3 py-1.5 rounded-md text-sm`
   - Blue background + shimmer when `active`
   - Green background + check icon when `done`
   - Red background when `error`
   - `opacity-0 → opacity-1` fade-in (200ms)

3. **Derive badges from existing `agentSteps`** in `AgentTimeline` using `useMemo`:
   - Map each `step` to a compact badge for the node types defined in `NODE_META`
   - Badge `status` derives from `step.status` (`active` → shimmer, `done` → green, `error` → red)
   - Render badges **above** existing expandable rows

4. **No new component files needed.** Add rendering inline in `AgentTimeline`'s return block, before the expandable section.

### 3.5 Verification

- Start streaming — badges appear as steps activate, shimmer while active
- After completion — badges persist with green/done state, shimmer stops
- Expandable rows still function below badges
- Dark mode: badges visible with appropriate color variants

---

## Phase 6 — Integration & E2E Verification

### 6.1 Checklist

| Test | How |
|------|-----|
| Skeleton pulse shows on new message | Observe pulsing circles during generation |
| flushSync removed | No forced layout on stream lines |
| Agent steps appear as compact badges | Watch badges shimmer during active nodes |
| Expandable rows still work | Click to see rewritten query, retrieval details |
| SSE still streams tokens | Content appears normally over SSE |
| Dark mode | All animations visible in dark mode |
| Reduced motion | Animations disabled when `prefers-reduced-motion: reduce` |
| Mobile layout | Pulsing skeleton fits on narrow screens |

### 6.2 Performance targets

| Metric | Target |
|--------|--------|
| FPS during streaming | ≥ 55 (measured with Chrome DevTools) |
| Main thread blocking | < 50ms per frame (no `flushSync` or forced layout) |
| SSE first token | Unchanged (no impact on initial latency) |

---

## Summary: Changes by file

| File | Change | Lines affected |
|------|--------|---------------|
| `globals.css` | Add `skeleton-pulse`, `skeleton-size`, `shimmer` keyframes | +20 |
| `frontend/src/app/dashboard/chat/[id]/page.tsx` | Replace bouncing dots, kill `flushSync` | ~10 |
| `frontend/src/components/chat/agent-timeline.tsx` | Add compact badge rows, add shimmer to active steps | ~40 |

**Total new files: 0**
**Total modified files: 3**
**Total estimated new lines: ~70**