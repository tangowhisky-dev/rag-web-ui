"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LogOut } from "lucide-react";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import Breadcrumb from "@/components/ui/breadcrumb";
import { ChangePasswordDialog } from "@/components/ui/change-password-dialog";
import { UserName } from "./user-name";

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
  const [hydrated, setHydrated] = useState(false);
  const [isAuthorized, setIsAuthorized] = useState(false);
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);

  useEffect(() => {
    setHydrated(true);
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/");
    } else {
      setIsAuthorized(true);
    }
  }, [router]);

  // Wait for hydration before showing content
  if (!hydrated || !isAuthorized) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  const handleLogout = () => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/");
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Single top bar: breadcrumb left, sign out right */}
      <header className="sticky top-0 z-40 w-full border-b bg-card/80 backdrop-blur-sm">
        <div className="flex h-12 items-center gap-4 px-4 sm:px-6">
          <Breadcrumb overrideLastLabel={pageTitle} />
          {graphRagActive && (
            <span className="inline-flex items-center rounded-full border border-violet-400/50 bg-violet-500/10 px-2 py-0.5 text-xs font-medium text-violet-400">
              ⬡ GraphRAG
            </span>
          )}
          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={() => setPasswordDialogOpen(true)}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
            >
              <span className="hidden sm:inline">Change Password</span>
              <span className="sm:hidden">Password</span>
            </button>
            <ThemeToggle />
            <UserName />
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

      {/* Change Password Dialog */}
      <ChangePasswordDialog
        open={passwordDialogOpen}
        onOpenChange={setPasswordDialogOpen}
      />

      {/* Content */}
      <main className="px-4 sm:px-6 lg:px-8 py-6">
        {children}
      </main>
    </div>
  );
}
