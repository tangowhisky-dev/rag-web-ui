"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { APP_NAME, APP_DESCRIPTION, APP_ICON_SRC } from "@/lib/app-config";
import {
  Brain,
  Upload,
  MessageSquare,
  Shield,
  Zap,
  Network,
  ChevronRight,
  Loader2,
} from "lucide-react";

export default function Home() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined" && localStorage.getItem("token")) {
      router.replace("/dashboard");
    }
  }, [router]);

  const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    const formData = new FormData(e.currentTarget);
    const formUrlEncoded = new URLSearchParams();
    formUrlEncoded.append("username", formData.get("username") as string);
    formUrlEncoded.append("password", formData.get("password") as string);
    try {
      const data = await api.post("/api/auth/token", formUrlEncoded, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      localStorage.setItem("token", data.access_token);
      document.cookie = `token=${data.access_token}; path=/; SameSite=Lax`;
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Invalid credentials");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* ── Top nav ─────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 w-full border-b bg-card/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex h-14 items-center gap-4">
          <Link href="/" className="flex items-center gap-2 font-semibold text-sm">
            <img src={APP_ICON_SRC} alt={APP_NAME} className="h-7 w-7 rounded-lg" />
            {APP_NAME}
          </Link>
          <nav className="hidden md:flex items-center gap-6 ml-8 text-sm text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-foreground transition-colors">How It Works</a>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <ThemeToggle />
            {/* <Link
              href="/register"
              className="text-sm px-4 py-1.5 rounded-full border hover:bg-muted transition-colors"
            >
              Create account
            </Link> */}
          </div>
        </div>
      </header>

      {/* ── Hero + Login ─────────────────────────────────────────────────── */}
      <section className="flex-1 flex items-center">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 w-full py-16 lg:py-24">
          <div className="grid lg:grid-cols-2 gap-16 items-center">

            {/* Left: marketing copy */}
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 rounded-full border bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
                <Zap className="h-3 w-3" />
                AI-Powered Enterprise Knowledge Platform
              </div>
              <h1 className="text-5xl sm:text-6xl font-bold tracking-tight leading-[1.1]">
                {APP_NAME}.<br />
                <span className="text-muted-foreground">Instantly searchable.</span>
              </h1>
              <p className="text-lg text-muted-foreground leading-relaxed max-w-lg">
                {APP_NAME} transforms your enterprise documents into an intelligent knowledge base.
                Ask questions in plain language and get precise, cited answers in seconds.
              </p>
              <div className="flex flex-col sm:flex-row gap-3">
                <Link
                  href="/register"
                  className="inline-flex items-center justify-center gap-2 rounded-full bg-primary text-primary-foreground px-6 py-3 text-sm font-semibold hover:bg-primary/90 transition-colors"
                >
                  Create Account
                  <ChevronRight className="h-4 w-4" />
                </Link>
              </div>
              {/* Trust signals */}
              <div className="flex flex-wrap items-center gap-6 pt-2">
                {[
                  { icon: Shield, label: "SOC 2 ready" },
                  { icon: Network, label: "GraphRAG engine" },
                  { icon: Zap, label: "Sub-second retrieval" },
                ].map(({ icon: Icon, label }) => (
                  <div key={label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Icon className="h-3.5 w-3.5" />
                    {label}
                  </div>
                ))}
              </div>
            </div>

            {/* Right: login form */}
            <div className="w-full max-w-md mx-auto lg:mx-0 lg:ml-auto">
              <div className="bg-card border rounded-2xl shadow-lg p-8 space-y-6">
                <div>
                  <h2 className="text-2xl font-semibold">Sign in</h2>
                  <p className="text-sm text-muted-foreground mt-1">Welcome back — continue where you left off.</p>
                </div>

                <form onSubmit={handleLogin} className="space-y-4">
                  <div>
                    <label htmlFor="username" className="block text-sm font-medium mb-1">Username</label>
                    <input
                      id="username" name="username" type="text" required disabled={loading}
                      placeholder="Enter your username"
                      className="w-full px-3 py-2 rounded-lg border border-input bg-background text-foreground text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground"
                    />
                  </div>
                  <div>
                    <label htmlFor="password" className="block text-sm font-medium mb-1">Password</label>
                    <input
                      id="password" name="password" type="password" required disabled={loading}
                      placeholder="Enter your password"
                      className="w-full px-3 py-2 rounded-lg border border-input bg-background text-foreground text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground"
                    />
                  </div>

                  {error && (
                    <div className="p-3 rounded-lg bg-destructive/10 text-destructive text-sm">{error}</div>
                  )}

                  <button
                    type="submit" disabled={loading}
                    className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Signing in…</> : "Sign in"}
                  </button>
                </form>

                <p className="text-center text-sm text-muted-foreground">
                  No account?{" "}
                  <Link href="/register" className="text-foreground font-medium hover:underline">
                    Create one now
                  </Link>
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────────── */}
      <section id="features" className="border-t bg-muted/40 py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold">Built for enterprise teams</h2>
            <p className="mt-3 text-muted-foreground max-w-xl mx-auto">
              From ingestion to insight — every layer is optimised for accuracy, speed, and trust.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: Brain,
                title: "Hybrid RAG Engine",
                body: "Dense, sparse, and exact retrieval legs — fused with a reranker — so every answer is grounded in your most relevant content.",
              },
              {
                icon: Network,
                title: "GraphRAG",
                body: "Entity extraction and knowledge-graph traversal surface cross-document relationships that keyword search simply misses.",
              },
              {
                icon: Upload,
                title: "Multi-format Ingestion",
                body: "Upload PDF, DOCX, Markdown, and TXT. Chunking and vector indexing run automatically in the background.",
              },
              {
                icon: MessageSquare,
                title: "Cited Answers",
                body: "Every response links back to the exact source passage with a relevance score, retrieval leg, and chunk rank.",
              },
              {
                icon: Zap,
                title: "Adaptive Retrieval",
                body: "Query classification routes each question to the optimal retrieval strategy — factual, entity-centric, multi-part, or ambiguous.",
              },
              {
                icon: Shield,
                title: "Private by Default",
                body: "Self-hosted. Your documents never leave your infrastructure. JWT auth with per-user isolation.",
              },
            ].map(({ icon: Icon, title, body }) => (
              <div key={title} className="bg-card border rounded-xl p-6 space-y-3 hover:shadow-md transition-shadow">
                <div className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center">
                  <Icon className="h-5 w-5 text-foreground" />
                </div>
                <h3 className="font-semibold">{title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────────────────── */}
      <section id="how-it-works" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold">Up and running in three steps</h2>
          </div>
          <div className="grid sm:grid-cols-3 gap-8">
            {[
              { step: "01", title: "Create a Knowledge Base", body: "Name it, describe it, hit create. Takes ten seconds." },
              { step: "02", title: "Upload Documents", body: "Drag in your PDFs and docs. Chunking and indexing run automatically." },
              { step: "03", title: "Ask Anything", body: "Open a chat, type your question, get a cited answer with full context." },
            ].map(({ step, title, body }) => (
              <div key={step} className="relative pl-12">
                <span className="absolute left-0 top-0 text-4xl font-black text-muted/60 leading-none select-none">{step}</span>
                <h3 className="font-semibold mb-2 mt-1">{title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <img src={APP_ICON_SRC} alt={APP_NAME} className="h-5 w-5 rounded" />
            <span>{APP_NAME} — {APP_DESCRIPTION}</span>
          </div>
          {/* <Link href="/register" className="hover:text-foreground transition-colors">Create account</Link> */}
        </div>
      </footer>
    </div>
  );
}
