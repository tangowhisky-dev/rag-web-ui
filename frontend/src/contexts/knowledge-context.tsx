"use client";

import { createContext, useContext, useEffect, useState, useCallback, useRef, useMemo } from "react";
import { api } from "@/lib/api";

export interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
  documents: { id: number; file_name?: string; created_at?: string }[];
  data_sources?: { id: number; name: string; folder_path: string }[];
  data_source_count: number;
  created_at: string;
}

interface KnowledgeContextValue {
  kbList: KnowledgeBase[];
  setKbList: React.Dispatch<React.SetStateAction<KnowledgeBase[]>>;
  activeKbId: number | null;
  setActiveKbId: React.Dispatch<React.SetStateAction<number | null>>;
  refreshKbList: () => Promise<void>;
  renameKb: (id: number, name: string) => Promise<void>;
  deleteKb: (id: number) => Promise<void>;
}

const KnowledgeContext = createContext<KnowledgeContextValue | null>(null);

export function KnowledgeProvider({ children }: { children: React.ReactNode }) {
  const [kbList, setKbList] = useState<KnowledgeBase[]>([]);
  const [activeKbId, setActiveKbId] = useState<number | null>(null);

  const refreshKbList = useCallback(async () => {
    try {
      const data: KnowledgeBase[] = await api.get("/api/knowledge-base");
      setKbList([...data].sort((a, b) => b.id - a.id));
    } catch {
      // silently ignore on auth failure
    }
  }, []);

  useEffect(() => {
    refreshKbList();
  }, [refreshKbList]);

  // Keep a ref to kbList so renameKb always reads the freshest data
  // without being re-created on every kbList change (which would break
  // any closures that hold a reference to an older renameKb).
  const kbListRef = useRef(kbList);
  kbListRef.current = kbList;

  const renameKb = useCallback(async (id: number, name: string) => {
    const current = kbListRef.current.find((k) => k.id === id);
    if (!current) return;
    await api.put(`/api/knowledge-base/${id}`, { name, description: current.description });
    setKbList((prev) => prev.map((k) => (k.id === id ? { ...k, name } : k)));
  }, []);

  const deleteKb = useCallback(async (id: number) => {
    await api.delete(`/api/knowledge-base/${id}`);
    setKbList((prev) => prev.filter((k) => k.id !== id));
  }, []);

  const value = useMemo(() => ({
    kbList, setKbList, activeKbId, setActiveKbId, refreshKbList, renameKb, deleteKb,
  }), [kbList, activeKbId, refreshKbList, renameKb, deleteKb]);

  return (
    <KnowledgeContext.Provider value={value}>
      {children}
    </KnowledgeContext.Provider>
  );
}

export function useKnowledgeContext() {
  const ctx = useContext(KnowledgeContext);
  if (!ctx) throw new Error("useKnowledgeContext must be used inside KnowledgeProvider");
  return ctx;
}
