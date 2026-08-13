"use client";

import { ChatProvider } from "@/contexts/chat-context";
import { PreflightCheck } from "@/components/preflight-check";

export function DashboardClientLayout({
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
