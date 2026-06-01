"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";

export interface Folder {
  id: number;
  name: string;
  user_id: number;
  created_at: string;
}

export interface Chat {
  id: number;
  title: string;
  created_at: string;
  pinned: boolean;
  folder_id: number | null;
}

interface ChatContextValue {
  chatList: Chat[];
  activeChat: number | null;
  folderList: Folder[];
  graphRagActive: boolean;
  setChatList: React.Dispatch<React.SetStateAction<Chat[]>>;
  setActiveChat: React.Dispatch<React.SetStateAction<number | null>>;
  setGraphRagActive: React.Dispatch<React.SetStateAction<boolean>>;
  renameChat: (id: number, title: string) => Promise<void>;
  deleteChat: (id: number) => Promise<void>;
  patchChat: (id: number, patch: Partial<Chat>) => Promise<void>;
  addChat: (chat: Chat) => void;
  fetchFolders: () => Promise<void>;
  createFolder: (name: string) => Promise<Folder>;
  renameFolder: (id: number, name: string) => Promise<void>;
  deleteFolder: (id: number) => Promise<void>;
  assignChatToFolder: (chatId: number, folderId: number | null) => Promise<void>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

const CHAT_LIST_CACHE_KEY = "rag_chat_list_cache";

// Module-level cache: survives ChatProvider remounts during client-side navigation.
// Server-side: always [] (no persistence between requests).
// Client-side: populated after first fetch, reused on any subsequent mount.
let _chatListCache: Chat[] = [];

function saveChatListCache(list: Chat[]) {
  _chatListCache = list;
  try { sessionStorage.setItem(CHAT_LIST_CACHE_KEY, JSON.stringify(list)); } catch {}
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  // Initialize from module-level cache so remounts never flash empty.
  const [chatList, _setChatList] = useState<Chat[]>(_chatListCache);

  const setChatList: React.Dispatch<React.SetStateAction<Chat[]>> = useCallback(
    (action) => {
      _setChatList((prev) => {
        const next = typeof action === "function" ? action(prev) : action;
        saveChatListCache(next);
        return next;
      });
    },
    []
  );
  const [activeChat, setActiveChat] = useState<number | null>(null);
  const [folderList, setFolderList] = useState<Folder[]>([]);
  const [graphRagActive, setGraphRagActive] = useState(false);

  const fetchFolders = useCallback(async () => {
    try {
      const data: Folder[] = await api.get("/api/folders");
      setFolderList(data);
    } catch {
      // silently ignore; user may not be authenticated yet
    }
  }, []);

  useEffect(() => {
    api
      .get("/api/chat")
      .then((data: Chat[]) => setChatList([...data].sort((a, b) => b.id - a.id)))
      .catch(() => {
        // silently ignore; user may not be authenticated yet
      });
    fetchFolders();
  }, [fetchFolders]);

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

  const createFolder = useCallback(async (name: string): Promise<Folder> => {
    const folder: Folder = await api.post("/api/folders", { name });
    setFolderList((prev) => [...prev, folder]);
    return folder;
  }, []);

  const renameFolder = useCallback(async (id: number, name: string) => {
    await api.patch(`/api/folders/${id}`, { name });
    setFolderList((prev) =>
      prev.map((f) => (f.id === id ? { ...f, name } : f))
    );
  }, []);

  const deleteFolder = useCallback(async (id: number) => {
    await api.delete(`/api/folders/${id}`);
    setFolderList((prev) => prev.filter((f) => f.id !== id));
    // Unassign chats that were in this folder
    setChatList((prev) =>
      prev.map((c) => (c.folder_id === id ? { ...c, folder_id: null } : c))
    );
  }, []);

  const assignChatToFolder = useCallback(
    async (chatId: number, folderId: number | null) => {
      await api.patch(`/api/chat/${chatId}`, { folder_id: folderId });
      setChatList((prev) =>
        prev.map((c) => (c.id === chatId ? { ...c, folder_id: folderId } : c))
      );
    },
    []
  );

  return (
    <ChatContext.Provider
      value={{
        chatList,
        activeChat,
        folderList,
        graphRagActive,
        setChatList,
        setActiveChat,
        setGraphRagActive,
        renameChat,
        deleteChat,
        patchChat,
        addChat,
        fetchFolders,
        createFolder,
        renameFolder,
        deleteFolder,
        assignChatToFolder,
      }}
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
