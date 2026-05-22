---
estimated_steps: 16
estimated_files: 2
skills_used: []
---

# T02: Create InputBar component with auto-resize textarea and KB selector

Replace the bare <input> with an auto-resize InputBar featuring Shift+Enter, animated send button, and KB selector dropdown.

Steps:
1. Create frontend/src/components/chat/chat-input.tsx
2. Build InputBar FC with props: value, onChange, onSubmit, disabled, placeholder
3. Implement auto-resize textarea: starts at 1 line, grows to max 5 lines, then scrolls; uses a hidden span for measurement
4. Handle Shift+Enter for newline (preventDefault on form submit when Shift held), Enter alone submits
5. Animated send button: Lucide Send icon that transitions to Loader2 spinner when disabled (isLoading)
6. KB selector dropdown using @radix-ui/react-select: fetch KBs from GET /api/knowledge-base, render as compact dropdown, multi-select mode
7. File attachment button slot (placeholder for S03): Paperclip icon button, disabled with tooltip "File attachment coming soon"
8. Create chat-input.test.tsx with Jest tests: auto-resize behavior, Shift+Enter vs Enter, KB selector renders, submit disabled when empty

Must-Haves:
- InputBar textarea auto-resizes from 1 to 5 lines
- Shift+Enter inserts newline, Enter submits
- Send button animates to spinner when disabled
- KB selector dropdown lists available KBs
- Test file passes with Jest

## Inputs

- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/components/ui/select.tsx`
- `frontend/src/components/ui/input.tsx`
- `frontend/src/lib/api.ts`

## Expected Output

- `frontend/src/components/chat/chat-input.tsx`
- `frontend/src/components/chat/__tests__/chat-input.test.tsx`

## Verification

npm test -- --testPathPattern=chat-input.test.tsx --no-coverage 2>&1 | tail -20
