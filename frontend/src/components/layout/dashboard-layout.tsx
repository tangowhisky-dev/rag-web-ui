"use client";

import { useEffect, useState } from "react";
import { CircuitBoard } from "lucide-react";
import Breadcrumb from "@/components/ui/breadcrumb";
import { NavActions } from "./nav-actions";

export default function DashboardLayout({
  children,
  pageTitle,
  graphRagActive,
}: {
  children: React.ReactNode;
  pageTitle?: string;
  graphRagActive?: boolean;
}) {
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setHydrated(true);
  }, []);

  // Wait for hydration before showing content
  if (!hydrated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="bg-background">
      {/* Single top bar: breadcrumb left, sign out right */}
      <header className="sticky top-0 z-40 w-full border-b bg-card/80 backdrop-blur-sm">
        <div className="flex h-12 items-center gap-4 px-4 sm:px-6">
          <Breadcrumb overrideLastLabel={pageTitle} />
          {graphRagActive && (
            <span className="inline-flex items-center gap-1 rounded-full border border-primary/20 bg-primary/5 px-2 py-0.5 text-xs font-medium text-muted-foreground">
              <CircuitBoard className="h-3 w-3" />
              GraphRAG
            </span>
          )}
          <div className="ml-auto flex items-center gap-1">
            <NavActions />
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="px-4 sm:px-6 lg:px-8 py-6">
        {children}
      </main>
    </div>
  );
}
