'use client';

import { Building2, Database, Users } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

interface AdminCounts {
  organizations: number;
  users: number;
  data_sources: number;
}

type AdminCountsKey = keyof AdminCounts;

// LLM config is managed via the Settings UI (Super Admin + per-org Settings).
// The standalone /dashboard/admin/llm-config page does not exist.

const ROUTE_MAP: Record<AdminCountsKey, string> = {
  organizations: '/dashboard/admin/orgs',
  users: '/dashboard/admin/users',
  data_sources: '/dashboard/admin/data-sources',
};

const STAT_CARDS = [
  { label: 'Organisations', key: 'organizations' as const, icon: Building2 },
  { label: 'Users', key: 'users' as const, icon: Users },
  { label: 'Data Stores', key: 'data_sources' as const, icon: Database },
];

export default function AdminPage() {
  const router = useRouter();
  const [counts, setCounts] = useState<AdminCounts | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/admin/counts')
      .then((data) => setCounts(data))
      .catch((e) => setError(e.message ?? 'Failed to load counts'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 pt-16">
      <div className="max-w-5xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">Admin Panel</h1>
          <p className="mt-2 text-muted-foreground">
            Manage organisations, users, and LLM configuration.
          </p>
        </div>

        {error && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            Unable to load counts: {error}
          </div>
        )}

        {loading && (
          <div className="flex justify-center py-12">
            <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin" />
          </div>
        )}

        {!loading && (
          <div className="grid gap-6 md:grid-cols-3">
            {STAT_CARDS.map(({ label, key, icon: Icon }) => (
              <button
                key={label}
                onClick={() => router.push(ROUTE_MAP[key])}
                className="rounded-2xl border bg-card text-card-foreground p-8 text-left hover:shadow-md transition-shadow cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <div className="flex items-center gap-4">
                  <div className="rounded-full bg-muted p-3">
                    <Icon className="h-6 w-6 text-foreground" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-bold">
                      {counts?.[key] ?? '—'}
                    </h3>
                    <p className="text-muted-foreground mt-1 text-sm">{label}</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
