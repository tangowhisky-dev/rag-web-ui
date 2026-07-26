'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

interface CurrentUser {
  sub: string;
  role: string;
}

export function UserName() {
  const [user, setUser] = useState<CurrentUser | null>(null);

  useEffect(() => {
    api.get('/api/auth/test-token')
      .then((data) => setUser(data as CurrentUser))
      .catch(() => setUser(null));
  }, []);

  const username = user?.sub ?? null;

  if (!username || !user) return null;

  // Extract just the username part (strip domain if present, e.g. "user@domain" -> "user")
  const displayName = username.includes('@') ? username.split('@')[0] : username;
  const role = user.role ?? 'user';

  const tooltipText = `signed in as ${displayName} (${role})`;

  return (
    <span
      title={tooltipText}
      className="relative group cursor-default"
    >
      <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-xs font-medium text-primary-foreground">
        {displayName.charAt(0).toUpperCase()}
      </div>
    </span>
  );
}
