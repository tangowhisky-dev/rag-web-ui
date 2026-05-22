"use client";

import { useState } from "react";
import { Menu } from "lucide-react";
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
      <div className="flex h-screen bg-background overflow-hidden">
        <ChatSidebar
          isOpen={isChatSidebarOpen}
          onClose={() => setIsChatSidebarOpen(false)}
        />

        {/* Main area */}
        <div className="flex-1 flex flex-col min-w-0 relative">
          {/* Mobile hamburger — only visible on small screens */}
          <button
            onClick={() => setIsChatSidebarOpen(true)}
            className="lg:hidden absolute top-3 left-3 z-10 p-1.5 rounded-lg hover:bg-muted transition-colors"
            aria-label="Open sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          {children}
        </div>
      </div>
    </ChatProvider>
  );
}
