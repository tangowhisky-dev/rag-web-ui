"use client";

import { useState } from "react";
import { Menu, CircuitBoard } from "lucide-react";
import ChatSidebar from "@/components/chat/chat-sidebar";
import { ChatProvider, useChatContext } from "@/contexts/chat-context";
import Breadcrumb from "@/components/ui/breadcrumb";
import { NavActions } from "./nav-actions";

interface ChatLayoutProps {
  children: React.ReactNode;
}

function ChatLayoutInner({ children }: ChatLayoutProps) {
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(false);
  const { chatList, activeChat, graphRagActive } = useChatContext();

  const chatTitle = activeChat
    ? (chatList.find((c) => c.id === activeChat)?.title ?? undefined)
    : undefined;

  return (
    <div className="relative h-screen bg-background overflow-hidden">
      <header className="absolute top-0 left-0 right-0 z-30 border-b bg-card/80 backdrop-blur-sm">
        <div className="flex h-12 items-center gap-2 px-4">
          <button
            onClick={() => setIsChatSidebarOpen(true)}
            className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors shrink-0"
            aria-label="Open sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          <Breadcrumb overrideLastLabel={chatTitle} />

          {graphRagActive && (
            <span className="inline-flex items-center gap-1 rounded-full border border-primary/20 bg-primary/5 px-2 py-0.5 text-xs font-medium text-muted-foreground shrink-0" title="GraphRAG active">
              <CircuitBoard className="h-3 w-3" />
              GraphRAG
            </span>
          )}

          <div className="ml-auto flex items-center gap-1">
            <NavActions />
          </div>
        </div>
      </header>

      <div className="absolute inset-0 flex">
        <div className="pt-12 flex-shrink-0 h-full">
          <ChatSidebar
            isOpen={isChatSidebarOpen}
            onClose={() => setIsChatSidebarOpen(false)}
          />
        </div>
        <main className="flex-1 min-w-0 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}

export default function ChatLayout({ children }: ChatLayoutProps) {
  return (
    <ChatProvider>
      <ChatLayoutInner>{children}</ChatLayoutInner>
    </ChatProvider>
  );
}
