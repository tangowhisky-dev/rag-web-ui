'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import DashboardLayout from '@/components/layout/dashboard-layout';
import { isAdmin } from '@/lib/auth';

const NAV_ITEMS = [
  { label: 'Orgs', href: '/dashboard/admin/orgs' },
  { label: 'Users', href: '/dashboard/admin/users' },
  { label: 'LLM Config', href: '/dashboard/admin/llm-config' },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  useEffect(() => {
    if (!isAdmin()) {
      router.replace('/dashboard');
    }
  }, [router]);

  return (
    <DashboardLayout pageTitle="Admin">
      <div className="flex min-h-screen">
        <nav className="w-48 shrink-0 border-r bg-card px-3 py-6 space-y-1">
          {NAV_ITEMS.map(({ label, href }) => (
            <Link
              key={href}
              href={href}
              className="block rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              {label}
            </Link>
          ))}
        </nav>
        <main className="flex-1 px-6 py-6">{children}</main>
      </div>
    </DashboardLayout>
  );
}
