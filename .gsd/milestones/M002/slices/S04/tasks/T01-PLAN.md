---
estimated_steps: 3
estimated_files: 2
skills_used: []
---

# T01: Install KaTeX and Mermaid npm packages

1. cd frontend && npm install remark-math rehype-katex katex mermaid
2. npm install --save-dev @types/katex (if available)
3. Verify the four packages appear in package.json dependencies.

## Inputs

- `frontend/package.json`

## Expected Output

- `frontend/package.json`
- `frontend/package-lock.json`

## Verification

grep -q '"remark-math"' frontend/package.json && grep -q '"rehype-katex"' frontend/package.json && grep -q '"mermaid"' frontend/package.json
