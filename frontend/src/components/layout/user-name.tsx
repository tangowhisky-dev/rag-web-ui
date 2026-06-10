'use client';

import { useState, useEffect } from 'react';
import { getTokenClaims, type TokenClaims } from '@/lib/auth';

export function UserName() {
  const [claims, setClaims] = useState<TokenClaims | null>(null);

  useEffect(() => {
    setClaims(getTokenClaims());
  }, []);

  const username = claims?.sub ?? null;

  if (!username || !claims) return null;

  // Extract just the username part (strip domain if present, e.g. "user@domain" → "user")
  const displayName = username.includes('@') ? username.split('@')[0] : username;
  const role = claims.role ?? 'user';

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
