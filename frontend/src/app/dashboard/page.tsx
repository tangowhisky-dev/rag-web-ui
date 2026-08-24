"use client";

import { useEffect, useState } from "react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { Book, MessageSquare, ArrowRight, Plus, Brain, Sparkles, Shield, Upload, Link2, Search } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useHydrated } from "@/lib/hooks";

interface Stats {
  knowledgeBases: number;
  chats: number;
}

export default function DashboardPage() {
  const hydrated = useHydrated();
  const [stats, setStats] = useState<Stats>({ knowledgeBases: 0, chats: 0 });
  const [currentUser, setCurrentUser] = useState<{ role: string } | null>(null);

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

    api.get("/api/auth/test-token")
      .then((data) => setCurrentUser(data as { role: string }))
      .catch(() => setCurrentUser(null));
  }, []);

  return (
    <DashboardLayout>
      <div className="p-6 max-w-7xl mx-auto pt-4">
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
                className="inline-flex items-center justify-center rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                <Plus className="mr-2 h-4 w-4" />
                New Knowledge Base
              </a>
              {hydrated && (currentUser?.role === "admin" || currentUser?.role === "super_admin") && (
                <a
                  href="/dashboard/admin"
                  className="inline-flex items-center justify-center rounded-md border border-input bg-background px-5 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
                >
                  <Shield className="mr-2 h-4 w-4" />
                  Admin Panel
                </a>
              )}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid gap-6 md:grid-cols-3 mb-8">
          {/* Primary stat: accent-tinted */}
          <div className="rounded-lg border bg-gradient-to-b from-primary/5 to-transparent p-8">
            <div className="flex items-center gap-6">
              <div className="rounded-md bg-primary/10 p-3">
                <Book className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-4xl font-bold tabular-nums" data-stat>{stats.knowledgeBases}</h3>
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

          {/* Secondary stat: neutral */}
          <div className="rounded-md border bg-card p-8">
            <div className="flex items-center gap-6">
              <div className="rounded-md bg-muted p-3">
                <MessageSquare className="h-6 w-6 text-muted-foreground" />
              </div>
              <div>
                <h3 className="text-4xl font-bold tabular-nums" data-stat>{stats.chats}</h3>
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

          {/* Search card */}
          <div className="rounded-md border bg-card p-8">
            <div className="flex items-center gap-6">
              <div className="rounded-md bg-muted p-3">
                <Search className="h-6 w-6 text-muted-foreground" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">KB Search</h3>
                <p className="text-muted-foreground mt-1 text-sm">Search files directly</p>
              </div>
            </div>
            <a
              href="/dashboard/search"
              className="mt-6 flex items-center text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Search knowledge bases
              <ArrowRight className="ml-2 h-4 w-4" />
            </a>
          </div>
        </div>

        {/* How It Works — three horizontal steps */}
        <div className="mb-12">
          <h2 className="text-2xl font-semibold mb-6">How It Works</h2>
          <div className="grid gap-4 md:grid-cols-3">
            {/* Step 1 */}
            <a
              href="/dashboard/knowledge/new"
              className="group rounded-lg border bg-card p-5 hover:border-primary/30 transition-colors"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary text-xs font-semibold text-primary-foreground">1</span>
                <Brain className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
              </div>
              <h3 className="text-sm font-medium mb-1">Create Knowledge Base</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Build a new AI-powered knowledge repository
              </p>
            </a>

            {/* Step 2 */}
            <a
              href="/dashboard/knowledge"
              className="group rounded-lg border bg-card p-5 hover:border-primary/30 transition-colors"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary text-xs font-semibold text-primary-foreground">2</span>
                <div className="flex gap-2">
                  <Upload className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                  <Link2 className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                </div>
              </div>
              <h3 className="text-sm font-medium mb-1">Upload Documents or Link Data Stores</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Upload PDFs, docs, images, or link data stores
              </p>
            </a>

            {/* Step 3 */}
            <a
              href="/dashboard/chat/new"
              className="group rounded-lg border bg-card p-5 hover:border-primary/30 transition-colors"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary text-xs font-semibold text-primary-foreground">3</span>
                <Sparkles className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
              </div>
              <h3 className="text-sm font-medium mb-1">Start Chatting</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Get instant answers from your knowledge with AI
              </p>
            </a>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
