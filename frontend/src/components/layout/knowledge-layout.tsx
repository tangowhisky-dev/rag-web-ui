"use client";

import { useState } from "react";
import { Menu, LogOut } from "lucide-react";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { useRouter } from "next/navigation";
import KnowledgeSidebar from "@/components/knowledge/knowledge-sidebar";
import { KnowledgeProvider } from "@/contexts/knowledge-context";
import Breadcrumb from "@/components/ui/breadcrumb";

interface KnowledgeLayoutProps {
  children: React.ReactNode;
  pageTitle?: string;
}

export default function KnowledgeLayout({ children, pageTitle }: KnowledgeLayoutProps) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const router = useRouter();

  const handleLogout = () => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/login");
  };

  return (
    <KnowledgeProvider>
      <div className="relative h-screen bg-background overflow-hidden">
        {/* Breadcrumb bar */}
        <header className="absolute top-0 left-0 right-0 z-30 border-b bg-card/80 backdrop-blur-sm">
          <div className="flex h-12 items-center gap-2 px-4">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors shrink-0"
              aria-label="Open sidebar"
            >
              <Menu className="h-5 w-5" />
            </button>

            <Breadcrumb overrideLastLabel={pageTitle} />

            <div className="ml-auto flex items-center gap-1">
              <ThemeToggle />
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

        {/* Sidebar + content */}
        <div className="absolute inset-0 flex">
          <div className="pt-12 flex-shrink-0 h-full">
            <KnowledgeSidebar
              isOpen={isSidebarOpen}
              onClose={() => setIsSidebarOpen(false)}
            />
          </div>
          <main className="flex-1 min-w-0 overflow-hidden">
            {children}
          </main>
        </div>
      </div>
    </KnowledgeProvider>
  );
}
