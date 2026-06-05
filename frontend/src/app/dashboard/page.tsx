"use client";

import { useEffect, useState } from "react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { Book, MessageSquare, ArrowRight, Plus, Upload, Brain, Sparkles, Shield } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { isAdmin } from "@/lib/auth";

// Track hydration to avoid SSR/client mismatch
const useHydrated = () => {
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);
  return hydrated;
};

interface Stats {
  knowledgeBases: number;
  chats: number;
}

export default function DashboardPage() {
  const hydrated = useHydrated();
  const [stats, setStats] = useState<Stats>({ knowledgeBases: 0, chats: 0 });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [kbData, chatData] = await Promise.all([
          api.get("/api/knowledge-base"),
          api.get("/api/chat"),
        ]);
        setStats({ knowledgeBases: kbData.length, chats: chatData.length });
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) return;
        console.error("Failed to fetch stats:", error);
      }
    };
    fetchStats();
  }, []);

  return (
    <DashboardLayout>
      <div className="p-6 max-w-7xl mx-auto">
        {/* Hero */}
        <div className="mb-12 rounded-2xl bg-muted border p-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
            <div className="space-y-1">
              <h1 className="text-4xl font-bold tracking-tight text-foreground">
                Enterprise Knowledge Assistant
              </h1>
              <p className="text-muted-foreground max-w-xl">
                Your personal AI-powered knowledge hub. Upload documents, create knowledge bases,
                and get instant answers through natural conversations.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <a
                href="/dashboard/knowledge/new"
                className="inline-flex items-center justify-center rounded-full bg-primary text-primary-foreground w-64 px-6 py-3 text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                <Plus className="mr-2 h-4 w-4" />
                New Knowledge Base
              </a>
              {hydrated && isAdmin() && (
                <a
                  href="/dashboard/admin"
                  className="inline-flex items-center justify-center rounded-full border border-input bg-background w-64 px-6 py-3 text-sm font-medium hover:bg-accent transition-colors"
                >
                  <Shield className="mr-2 h-4 w-4" />
                  Admin Panel
                </a>
              )}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid gap-6 md:grid-cols-2 mb-12">
          <div className="rounded-2xl border bg-card text-card-foreground p-8 hover:shadow-md transition-shadow">
            <div className="flex items-center gap-6">
              <div className="rounded-full bg-muted p-4">
                <Book className="h-8 w-8 text-foreground" />
              </div>
              <div>
                <h3 className="text-4xl font-bold">{stats.knowledgeBases}</h3>
                <p className="text-muted-foreground mt-1">Knowledge Bases</p>
              </div>
            </div>
            <a
              href="/dashboard/knowledge"
              className="mt-6 flex items-center text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              View all knowledge bases
              <ArrowRight className="ml-2 h-4 w-4" />
            </a>
          </div>

          <div className="rounded-2xl border bg-card text-card-foreground p-8 hover:shadow-md transition-shadow">
            <div className="flex items-center gap-6">
              <div className="rounded-full bg-muted p-4">
                <MessageSquare className="h-8 w-8 text-foreground" />
              </div>
              <div>
                <h3 className="text-4xl font-bold">{stats.chats}</h3>
                <p className="text-muted-foreground mt-1">Chat Sessions</p>
              </div>
            </div>
            <a
              href="/dashboard/chat"
              className="mt-6 flex items-center text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              View all chat sessions
              <ArrowRight className="ml-2 h-4 w-4" />
            </a>
          </div>
        </div>

        {/* How It Works */}
        <h2 className="text-2xl font-semibold mb-2">How It Works</h2>
        <div className="grid gap-4 md:grid-cols-3 mb-12">
          <a
            href="/dashboard/knowledge/new"
            className="relative flex flex-col items-center justify-center rounded-2xl border bg-card text-card-foreground p-5 hover:shadow-md hover:border-foreground/30 transition-all"
          >
            <span className="absolute top-2 left-2 flex h-7 w-7 items-center justify-center rounded-full bg-muted text-xs font-bold">
              1
            </span>
            <div className="rounded-full bg-muted p-3 mb-3">
              <Brain className="h-6 w-6 text-foreground" />
            </div>
            <h3 className="text-base font-medium mb-1">Create Knowledge Base</h3>
            <p className="text-sm text-muted-foreground text-center">
              Build a new AI-powered knowledge repository
            </p>
          </a>

          <a
            href="/dashboard/knowledge"
            className="relative flex flex-col items-center justify-center rounded-2xl border bg-card text-card-foreground p-5 hover:shadow-md hover:border-foreground/30 transition-all"
          >
            <span className="absolute top-2 left-2 flex h-7 w-7 items-center justify-center rounded-full bg-muted text-xs font-bold">
              2
            </span>
            <div className="rounded-full bg-muted p-3 mb-3">
              <Upload className="h-6 w-6 text-foreground" />
            </div>
            <h3 className="text-base font-medium mb-1">Upload Documents</h3>
            <p className="text-sm text-muted-foreground text-center">
              Add PDFs, documents or images to your knowledge bases
            </p>
          </a>

          <a
            href="/dashboard/chat/new"
            className="relative flex flex-col items-center justify-center rounded-2xl border bg-card text-card-foreground p-5 hover:shadow-md hover:border-foreground/30 transition-all"
          >
            <span className="absolute top-2 left-2 flex h-7 w-7 items-center justify-center rounded-full bg-muted text-xs font-bold">
              3
            </span>
            <div className="rounded-full bg-muted p-3 mb-3">
              <Sparkles className="h-6 w-6 text-foreground" />
            </div>
            <h3 className="text-base font-medium mb-1">Start Chatting</h3>
            <p className="text-sm text-muted-foreground text-center">
              Get instant answers from your knowledge with AI
            </p>
          </a>
        </div>
      </div>
    </DashboardLayout>
  );
}
