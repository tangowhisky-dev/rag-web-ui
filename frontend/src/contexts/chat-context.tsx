"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";

export interface Chat {
  id: number;
  title: string;
  created_at: string;
  pinned: boolean;
}

interface ChatContextValue {
  chatList: Chat[];
  activeChat: number | null;
  setChatList: React.Dispatch<React.SetStateAction<Chat[]>>;
  setActiveChat: React.Dispatch<React.SetStateAction<number | null>>;
  renameChat: (id: number, title: string) => Promise<void>;
  deleteChat: (id: number) => Promise<void>;
  patchChat: (id: number, patch: Partial<Chat>) => Promise<void>;
  addChat: (chat: Chat) => void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [chatList, setChatList] = useState<Chat[]>([]);
  const [activeChat, setActiveChat] = useState<number | null>(null);

  useEffect(() => {
    api
      .get("/api/chat")
      .then((data: Chat[]) => setChatList([...data].sort((a, b) => b.id - a.id)))
      .catch(() => {
        // silently ignore; user may not be authenticated yet
      });
  }, []);

  const renameChat = useCallback(async (id: number, title: string) => {
    await api.patch(`/api/chat/${id}`, { title });
    setChatList((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c))
    );
  }, []);

  const deleteChat = useCallback(async (id: number) => {
    await api.delete(`/api/chat/${id}`);
    setChatList((prev) => prev.filter((c) => c.id !== id));
  }, []);

  const patchChat = useCallback(async (id: number, patch: Partial<Chat>) => {
    await api.patch(`/api/chat/${id}`, patch);
    setChatList((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...patch } : c))
    );
  }, []);

  // Prepend a newly created chat to the sorted list
  const addChat = useCallback((chat: Chat) => {
    setChatList((prev) => [chat, ...prev]);
  }, []);

  return (
    <ChatContext.Provider
      value={{ chatList, activeChat, setChatList, setActiveChat, renameChat, deleteChat, patchChat, addChat }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChatContext(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return ctx;
}
