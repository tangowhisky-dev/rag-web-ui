"use client";

import { useState } from "react";
import { Menu } from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
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
      <DashboardLayout pageTitle={pageTitle} graphRagActive={graphRagActive}>
        <div className="flex h-full relative">
          <ChatSidebar
            isOpen={isChatSidebarOpen}
            onClose={() => setIsChatSidebarOpen(false)}
          />

          {/* Hamburger — mobile only */}
          <button
            onClick={() => setIsChatSidebarOpen(true)}
            className="lg:hidden absolute top-3 left-3 z-10 p-2 rounded-lg bg-background border shadow-sm hover:bg-muted transition-colors"
            aria-label="Open chat sidebar"
          >
            <Menu className="h-4 w-4" />
          </button>

          <main className="flex-1 min-w-0 overflow-auto">{children}</main>
        </div>
      </DashboardLayout>
    </ChatProvider>
  );
}
