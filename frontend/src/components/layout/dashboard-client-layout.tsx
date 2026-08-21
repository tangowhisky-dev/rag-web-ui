"use client";

import { ChatProvider } from "@/contexts/chat-context";
import { PreflightCheck } from "@/components/preflight-check";
import { useSessionHeartbeat } from "@/hooks/use-session-heartbeat";

export function DashboardClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  useSessionHeartbeat();

  return (
    <ChatProvider>
      <PreflightCheck />
      {children}
    </ChatProvider>
  );
}
