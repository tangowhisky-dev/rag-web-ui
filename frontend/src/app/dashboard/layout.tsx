"use client";

import { ChatProvider } from "@/contexts/chat-context";
import { PreflightCheck } from "@/components/preflight-check";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ChatProvider>
      <PreflightCheck />
      {children}
    </ChatProvider>
  );
}
