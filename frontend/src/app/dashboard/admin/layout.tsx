'use client';

import { useState, useEffect } from 'react';
import Breadcrumb from '@/components/ui/breadcrumb';
import AdminSidebar from '@/components/admin/admin-sidebar';
import { NavActions } from '@/components/layout/nav-actions';
import { api } from '@/lib/api';

// LLM config is managed via the Settings UI (Super Admin + per-org Settings).
// The standalone /dashboard/admin/llm-config page does not exist.

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [userRole, setUserRole] = useState<string | undefined>(undefined);

  useEffect(() => {
    setHydrated(true);
    api.get('/api/auth/test-token').then((data: { role?: string }) => {
      setUserRole(data?.role);
    }).catch(() => {});
  }, []);

  // Wait for hydration before rendering
  if (!hydrated) {
    return (
      <div className="relative h-screen bg-background overflow-hidden flex items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="relative h-screen bg-background overflow-hidden">
      {/* Header bar */}
      <header className="absolute top-0 left-0 right-0 z-30 border-b bg-card/80 backdrop-blur-sm">
        <div className="flex h-12 items-center gap-2 px-4">
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="lg:hidden p-1.5 rounded-lg hover:bg-muted transition-colors shrink-0"
            aria-label="Open sidebar"
          >
            <span className="sr-only">Open sidebar</span>
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <Breadcrumb overrideLastLabel="Admin" />
          <div className="ml-auto flex items-center gap-1">
            <NavActions showPasswordButton={false} />
          </div>
        </div>
      </header>

      {/* Sidebar + content */}
      <div className="absolute inset-0 flex">
        <div className="pt-12 flex-shrink-0 h-full">
          <AdminSidebar
            isOpen={isSidebarOpen}
            onClose={() => setIsSidebarOpen(false)}
            userRole={userRole}
          />
        </div>
        <main className="flex-1 min-w-0 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
