"use client";

import { ChatProvider } from "@/contexts/chat-context";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <ChatProvider>{children}</ChatProvider>;
}
