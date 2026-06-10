'use client';

import { getTokenClaims } from '@/lib/auth';

export function UserName() {
  const claims = getTokenClaims();
  const username = claims?.sub ?? null;

  if (!username) return null;

  // Extract just the username part (strip domain if present, e.g. "user@domain" → "user")
  const displayName = username.includes('@') ? username.split('@')[0] : username;
  const role = claims.role ?? 'user';

  // Build tooltip text: "signed in as username" or "signed in as username (admin)"
  const tooltipText =
    role === 'user' || role === 'super_admin'
      ? `signed in as ${displayName} (${role})`
      : `signed in as ${displayName} (${role})`;

  return (
    <div className="flex items-center gap-2 shrink-0">
      <span
        title={tooltipText}
        className="relative group cursor-default"
      >
        <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-xs font-medium text-primary-foreground">
          {displayName.charAt(0).toUpperCase()}
        </div>
      </span>
    </div>
  );
}
