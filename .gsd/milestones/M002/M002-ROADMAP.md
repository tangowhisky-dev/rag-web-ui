# M002: Modern Chat UI + Agentic Workflows

**Vision:** Transform the RAG Web UI chat experience into a polished, ChatGPT/Open WebUI-style interface with persistent sidebar, animated AgentTimeline, rich rendering, and a clean floating input bar.

## Slices

- [ ] **S01: Chat Sidebar and Layout Foundation** `risk:high` `depends:[]`
  > After this: Sidebar lists chats, New Chat works, rename/delete, mobile drawer

- [ ] **S02: AgentTimeline and Streaming** `risk:medium` `depends:[S01]`
  > After this: AgentTimeline animates step by step during streaming

- [ ] **S03: File Attachment and Multipart SSE** `risk:medium` `depends:[S02]`
  > After this: Drag PDF, file chip appears, submit includes file context

- [ ] **S04: Rich Rendering and Citations** `risk:medium` `depends:[S02]`
  > After this: LaTeX renders, Mermaid renders, citation popovers show score

- [ ] **S05: Per-Chat Settings and Controls** `risk:low` `depends:[S02]`
  > After this: Settings panel opens, model/temp/legs configurable, pin/export/search work

- [ ] **S06: Chat UI Redesign to Open WebUI Style** `risk:medium` `depends:[S01,S02,S03,S04,S05]`
  > After this: Full-height sidebar, floating rounded input, clean bubbles, welcome screen — visually matching Open WebUI

## Boundary Map

Not provided.
