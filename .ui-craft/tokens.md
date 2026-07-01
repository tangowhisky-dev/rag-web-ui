# Design Tokens

## Colors

- **Neutral ramp:** HSL 222.2/84%/4.9 (foreground) → 210/40%/96.1% (muted) → 0/0%/100% (background). Pure grays in dark mode (0/0% palette).
- **Accent:** `--primary` = 222.2/47.4%/11.2% (near-black). One accent only — used for CTAs, active states, links.
- **Semantic:**
  - Success: `--chart-2` = 173/58%/39% (light), 160/50%/45% (dark)
  - Warning: `--chart-3` = 197/37%/24% (light), 30/70%/55% (dark)
  - Error: `--destructive` = 0/84.2%/60.2% (light), 0/62.8%/40% (dark)
  - Info: `--chart-1` = 12/76%/61% (light), 220/60%/55% (dark)
- **Dark palette:** Pure neutral grey (Open WebUI style). Background 0/0%/12%, card 0/0%/16%, border 0/0%/25%. No blue tint.

## Typography

- **Body:** System sans-serif stack, 14px/1.5, weight 400 (Tailwind default).
- **Display:** System sans-serif, weight 600-700, tracking-tight above 24px.
- **Mono:** JetBrains Mono for code blocks (via `hljs` CSS class).
- **Scale:** text-sm (14px), text-base (16px), text-lg (18px), text-xl (20px), text-2xl (24px), text-3xl (30px), text-4xl (36px), text-5xl (48px).

## Spacing

- **4px grid.** Component padding: 12px (sm), 16px (md), 24px (lg), 32px (xl).
- **Container:** center-aligned, 2rem padding, max 1400px (2xl).
- **Section gaps:** 6 (24px), 8 (32px), 12 (48px), 16 (64px).

## Radius

- **Inputs:** 0.5rem (8px) — `--radius` token.
- **Cards:** 0.5rem (8px) for body cards, 1rem (16px) for hero/banner cards.
- **Buttons:** 9999px (pill) for CTAs on landing/dashboard.
- **Badges:** 9999px (pill) for status badges.
- **Modals:** 0.75rem (12px).
- **Code blocks:** 0.5rem (8px).

## Shadows

- **sm:** `0 1px 2px rgba(0,0,0,0.05)` (hover cards).
- **md:** `0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05)` (elevated cards).
- **lg:** `0 10px 15px rgba(0,0,0,0.1), 0 4px 6px rgba(0,0,0,0.05)` (login card on landing).
- **Dark mode:** Use tinted surfaces over shadows for elevation (gray-850 > gray-900 > gray-950).
