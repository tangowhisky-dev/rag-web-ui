"use client";

import React, { createContext, useContext, useEffect, useLayoutEffect, useState, useCallback, useMemo } from "react";
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
  updated_at?: string;
  pinned: boolean;
  folder_id: number | null;
}

interface ChatContextValue {
  chatList: Chat[];
  activeChat: number | null;
  folderList: Folder[];
  graphRagActive: boolean;
  chatListLoaded: boolean;
  setChatList: React.Dispatch<React.SetStateAction<Chat[]>>;
  setActiveChat: React.Dispatch<React.SetStateAction<number | null>>;
  setGraphRagActive: React.Dispatch<React.SetStateAction<boolean>>;
  renameChat: (id: number, title: string) => Promise<void>;
  deleteChat: (id: number) => Promise<void>;
  patchChat: (id: number, patch: Partial<Chat>) => Promise<void>;
  addChat: (chat: Chat) => void;
  bumpChatToTop: (chatId: number) => void;
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

// Read cached chat list from module-level cache, falling back to sessionStorage.
// This ensures that even if the module is re-evaluated (e.g. after a full
// page reload), the sidebar doesn't flash empty while the API fetch is in-flight.
function loadCachedChatList(): Chat[] {
  if (_chatListCache.length > 0) return _chatListCache;
  try {
    const stored = sessionStorage.getItem(CHAT_LIST_CACHE_KEY);
    if (stored) {
      _chatListCache = JSON.parse(stored);
      return _chatListCache;
    }
  } catch {}
  return [];
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  // Initialize from module-level cache. On the client, also try sessionStorage
  // synchronously — but only after the first render to avoid hydration
  // mismatches. We use a lazy initializer that checks the module cache first,
  // then a useSyncExternalStore-like pattern to load sessionStorage ASAP.
  const [chatList, _setChatList] = useState<Chat[]>(_chatListCache);
  const [chatListLoaded, setChatListLoaded] = useState(_chatListCache.length > 0);
  const [hydrated, setHydrated] = useState(false);

  // Load from sessionStorage synchronously before paint — useLayoutEffect
  // fires before the browser paints, so the sidebar never flashes empty
  // if sessionStorage has cached data.
  useLayoutEffect(() => {
    if (hydrated) return;
    setHydrated(true);
    if (_chatListCache.length === 0) {
      const cached = loadCachedChatList();
      if (cached.length > 0) {
        _setChatList(cached);
        setChatListLoaded(true);
      }
    }
  }, [hydrated]);

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
    // Skip re-fetching if we already have cached data — prevents the
    // sidebar from flashing empty during chat-to-chat navigation.
    if (_chatListCache.length > 0) return;
    api
      .get("/api/chat")
      .then((data: Chat[]) => {
        const sorted = [...data].sort((a, b) => {
          // Sort by updated_at descending (most recently active first),
          // falling back to created_at, then id.
          const aTime = a.updated_at ?? a.created_at;
          const bTime = b.updated_at ?? b.created_at;
          if (aTime && bTime) return bTime.localeCompare(aTime);
          return b.id - a.id;
        });
        setChatList(sorted);
        setChatListLoaded(true);
      })
      .catch(() => {
        // silently ignore; user may not be authenticated yet
        setChatListLoaded(true);
      });
    fetchFolders();
  }, [fetchFolders, setChatList]);

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

  // Move a chat to the top of the unpinned list (called when a message is sent).
  // Pinned chats stay pinned — only unpinned chats are reordered.
  const bumpChatToTop = useCallback((chatId: number) => {
    setChatList((prev) => {
      const chat = prev.find((c) => c.id === chatId);
      if (!chat) return prev;
      const updated = { ...chat, updated_at: new Date().toISOString() };
      const pinned = prev.filter((c) => c.pinned);
      const unpinned = prev.filter((c) => !c.pinned && c.id !== chatId);
      // Bumped chat goes to top of unpinned
      return [...pinned, updated, ...unpinned];
    });
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

  const value = useMemo(() => ({
    chatList, activeChat, folderList, graphRagActive, chatListLoaded,
    setChatList, setActiveChat, setGraphRagActive,
    renameChat, deleteChat, patchChat, addChat, bumpChatToTop,
    fetchFolders, createFolder, renameFolder, deleteFolder, assignChatToFolder,
  }), [chatList, activeChat, folderList, graphRagActive, chatListLoaded,
       setChatList, setActiveChat, setGraphRagActive,
       renameChat, deleteChat, patchChat, addChat, bumpChatToTop,
       fetchFolders, createFolder, renameFolder, deleteFolder, assignChatToFolder]);

  return (
    <ChatContext.Provider value={value}>
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
