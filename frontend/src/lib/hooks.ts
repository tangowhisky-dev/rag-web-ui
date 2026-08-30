"use client";

import { useRouter } from "next/navigation";
import { useCallback, useState, useSyncExternalStore } from "react";
import { api } from "@/lib/api";

export function useHydrated() {
  return useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
}

export function useLogout() {
  const router = useRouter();
  return useCallback(async () => {
    await api.post("/api/auth/logout");
    router.push("/");
  }, [router]);
}

/** Sidebar collapse state synced to localStorage. */
export function useSidebarCollapse(storageKey: string) {
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem(storageKey) === "true";
  });

  const toggleCollapse = useCallback(() => {
    setCollapsed((prev) => {
      localStorage.setItem(storageKey, String(!prev));
      return !prev;
    });
  }, [storageKey]);

  return { collapsed, toggleCollapse };
}

/** Download a chat export as a markdown file. */
export async function exportChatToMarkdown(chatId: string | number): Promise<void> {
  const res = await api.getRaw(`/api/chat/${chatId}/export`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chat-${chatId}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
