'use client';

import { Building2, Users, Settings } from 'lucide-react';

const STAT_CARDS = [
  { label: 'Organisations', value: 0, icon: Building2 },
  { label: 'Users', value: 0, icon: Users },
  { label: 'LLM Configs', value: 0, icon: Settings },
];

export default function AdminPage() {
  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Admin Panel</h1>
        <p className="mt-2 text-muted-foreground">
          Manage organisations, users, and LLM configuration.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {STAT_CARDS.map(({ label, value, icon: Icon }) => (
          <div
            key={label}
            className="rounded-2xl border bg-card text-card-foreground p-8 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center gap-4">
              <div className="rounded-full bg-muted p-3">
                <Icon className="h-6 w-6 text-foreground" />
              </div>
              <div>
                <h3 className="text-3xl font-bold">{value}</h3>
                <p className="text-muted-foreground mt-1 text-sm">{label}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
