"use client";

import { useState } from "react";
import { Menu, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import ChatSidebar from "@/components/chat/chat-sidebar";
import { ChatProvider } from "@/contexts/chat-context";
import Breadcrumb from "@/components/ui/breadcrumb";

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
  const router = useRouter();

  const handleLogout = () => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/login");
  };

  return (
    <ChatProvider>
      <div className="flex flex-col h-screen bg-background overflow-hidden">
        {/* Full-width breadcrumb bar */}
        <header className="w-full border-b bg-card shrink-0 z-30">
          <div className="flex h-12 items-center gap-2 px-4">
            {/* Mobile hamburger */}
            <button
              onClick={() => setIsChatSidebarOpen(true)}
              className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors shrink-0"
              aria-label="Open sidebar"
            >
              <Menu className="h-5 w-5" />
            </button>

            <Breadcrumb overrideLastLabel={pageTitle} />

            {graphRagActive && (
              <span className="inline-flex items-center rounded-full border border-violet-400/50 bg-violet-500/10 px-2 py-0.5 text-xs font-medium text-violet-400 shrink-0">
                ⬡ GraphRAG
              </span>
            )}

            <div className="ml-auto">
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Sign out</span>
              </button>
            </div>
          </div>
        </header>

        {/* Sidebar + content row */}
        <div className="flex flex-1 min-h-0">
          <ChatSidebar
            isOpen={isChatSidebarOpen}
            onClose={() => setIsChatSidebarOpen(false)}
          />
          <main className="flex-1 min-w-0 overflow-hidden">
            {children}
          </main>
        </div>
      </div>
    </ChatProvider>
  );
}
