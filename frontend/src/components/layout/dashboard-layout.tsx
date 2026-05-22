"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { Book, MessageSquare, LogOut, Home } from "lucide-react";
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
  const pathname = usePathname();

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

  const navigation = [
    { name: "Knowledge Base", href: "/dashboard/knowledge", icon: Book },
    { name: "Chat", href: "/dashboard/chat", icon: MessageSquare },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Top nav bar */}
      <header className="sticky top-0 z-40 w-full border-b bg-card">
        <div className="flex h-14 items-center gap-4 px-4 sm:px-6">
          {/* Logo + app name (home link) */}
          <Link
            href="/dashboard"
            className="flex items-center gap-2 font-semibold text-sm hover:text-primary transition-colors"
          >
            <img src="/logo.svg" alt="Logo" className="h-7 w-7 rounded-lg" />
            <span className="hidden sm:inline">InsightCore</span>
          </Link>

          {/* Nav links */}
          <nav className="flex items-center gap-1 ml-2">
            {navigation.map((item) => {
              const isActive = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent/60"
                  }`}
                >
                  <item.icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right side: sign out */}
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

      {/* Main content */}
      <main className="px-4 sm:px-6 lg:px-8 py-6">
        <Breadcrumb overrideLastLabel={pageTitle} />
        {graphRagActive && (
          <span className="ml-2 inline-flex items-center rounded-full border border-violet-400/50 bg-violet-500/10 px-2 py-0.5 text-xs font-medium text-violet-400">
            ⬡ GraphRAG
          </span>
        )}
        {children}
      </main>
    </div>
  );
}
