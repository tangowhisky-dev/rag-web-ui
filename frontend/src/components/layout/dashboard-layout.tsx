"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { LogOut } from "lucide-react";
import Breadcrumb from "@/components/ui/breadcrumb";

export default function DashboardLayout({
  children,
  pageTitle,
  graphRagActive,
}: {
  children: React.ReactNode;
  pageTitle?: string;
  graphRagActive?: boolean;
}) {
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
    }
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/login");
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Single top bar: breadcrumb left, sign out right */}
      <header className="sticky top-0 z-40 w-full border-b bg-card">
        <div className="flex h-12 items-center gap-4 px-4 sm:px-6">
          <Breadcrumb overrideLastLabel={pageTitle} />
          {graphRagActive && (
            <span className="inline-flex items-center rounded-full border border-violet-400/50 bg-violet-500/10 px-2 py-0.5 text-xs font-medium text-violet-400">
              ⬡ GraphRAG
            </span>
          )}
          <div className="ml-auto">
            <button
              onClick={handleLogout}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
            >
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Sign out</span>
            </button>
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
