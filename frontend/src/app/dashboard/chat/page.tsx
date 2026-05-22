"use client";

import Link from "next/link";
import { Plus, MessageSquare } from "lucide-react";
import ChatLayout from "@/components/layout/chat-layout";

export default function ChatPage() {
  return (
    <ChatLayout>
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center px-6">
        <div className="bg-primary/10 rounded-full p-4 mb-4">
          <MessageSquare className="h-8 w-8 text-primary" />
        </div>
        <h2 className="text-2xl font-semibold mb-2">No chat selected</h2>
        <p className="text-muted-foreground mb-6 max-w-sm">
          Select a conversation from the sidebar, or start a new one.
        </p>
        <Link
          href="/dashboard/chat/new"
          className="inline-flex items-center justify-center rounded-full bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors shadow-sm"
        >
          <Plus className="mr-2 h-4 w-4" />
          Start New Chat
        </Link>
      </div>
    </ChatLayout>
  );
}
