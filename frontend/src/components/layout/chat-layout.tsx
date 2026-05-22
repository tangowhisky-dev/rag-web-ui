"use client";

import { useState } from "react";
import Link from "next/link";
import { Menu, Home, Book } from "lucide-react";
import ChatSidebar from "@/components/chat/chat-sidebar";
import { ChatProvider } from "@/contexts/chat-context";

interface ChatLayoutProps {
  children: React.ReactNode;
  pageTitle?: string;
  graphRagActive?: boolean;
}

export default function ChatLayout({
  children,
  pageTitle,
  graphRagActive,
}: ChatLayoutProps) {
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(false);

  return (
    <ChatProvider>
      <div className="flex flex-col h-screen bg-background">
        {/* Top nav bar */}
        <header className="h-14 border-b flex items-center gap-3 px-4 shrink-0 bg-card">
          {/* Mobile hamburger */}
          <button
            onClick={() => setIsChatSidebarOpen(true)}
            className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors"
            aria-label="Open chat sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          {/* Logo + title */}
          <Link
            href="/dashboard"
            className="flex items-center gap-2 font-semibold text-base hover:text-primary transition-colors"
          >
            <img src="/logo.svg" alt="Logo" className="h-7 w-7 rounded" />
            InsightCore
          </Link>

          {/* GraphRAG badge */}
          {graphRagActive && (
            <span className="inline-flex items-center rounded-full border border-violet-400/50 bg-violet-500/10 px-2 py-0.5 text-xs font-medium text-violet-400">
              ⬡ GraphRAG
            </span>
          )}

          {/* Page title */}
          {pageTitle && (
            <span className="hidden sm:block text-sm text-muted-foreground truncate max-w-xs">
              {pageTitle}
            </span>
          )}

          {/* Right side nav */}
          <nav className="ml-auto flex items-center gap-1">
            <Link
              href="/dashboard"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              aria-label="Home"
            >
              <Home className="h-4 w-4" />
              <span className="hidden sm:inline">Home</span>
            </Link>
            <Link
              href="/dashboard/knowledge"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              <Book className="h-4 w-4" />
              <span className="hidden sm:inline">Knowledge Base</span>
            </Link>
          </nav>
        </header>

        {/* Body: sidebar + main */}
        <div className="flex flex-1 min-h-0">
          <ChatSidebar
            isOpen={isChatSidebarOpen}
            onClose={() => setIsChatSidebarOpen(false)}
          />
          <main className="flex-1 min-w-0 overflow-auto">{children}</main>
        </div>
      </div>
    </ChatProvider>
  );
}
