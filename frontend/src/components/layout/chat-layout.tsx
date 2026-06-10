"use client";

import { useState } from "react";
import { Menu, LogOut, Share2 } from "lucide-react";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { useRouter } from "next/navigation";
import ChatSidebar from "@/components/chat/chat-sidebar";
import { ChatProvider, useChatContext } from "@/contexts/chat-context";
import Breadcrumb from "@/components/ui/breadcrumb";
import { ChangePasswordDialog } from "@/components/ui/change-password-dialog";
import { UserName } from "./user-name";

interface ChatLayoutProps {
  children: React.ReactNode;
}

function ChatLayoutInner({ children }: ChatLayoutProps) {
  const [isChatSidebarOpen, setIsChatSidebarOpen] = useState(false);
  const router = useRouter();
  const { chatList, activeChat, graphRagActive } = useChatContext();

  const chatTitle = activeChat
    ? (chatList.find((c) => c.id === activeChat)?.title ?? undefined)
    : undefined;

  const handleLogout = () => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/");
  };
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);

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
            <span className="inline-flex items-center rounded-full border border-violet-400/50 bg-violet-500/10 p-1 text-violet-400 shrink-0" title="GraphRAG active">
              <Share2 className="h-3 w-3" />
            </span>
          )}

          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={() => setPasswordDialogOpen(true)}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
            >
              <span className="hidden sm:inline">Change Password</span>
              <span className="sm:hidden">Password</span>
            </button>
            <ThemeToggle />
            <UserName />
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

      {/* Change Password Dialog */}
      <ChangePasswordDialog
        open={passwordDialogOpen}
        onOpenChange={setPasswordDialogOpen}
      />

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
