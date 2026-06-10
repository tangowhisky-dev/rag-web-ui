'use client';

import { getTokenClaims } from '@/lib/auth';

export function UserName() {
  const claims = getTokenClaims();
  const username = claims?.sub ?? null;

  if (!username) return null;

  // Extract just the username part (strip domain if present, e.g. "user@domain" → "user")
  const displayName = username.includes('@') ? username.split('@')[0] : username;

  return (
    <div className="flex items-center gap-2 shrink-0">
      <div className="h-8 w-8 rounded-full bg-muted/60 flex items-center justify-center text-xs font-medium text-muted-foreground">
        {displayName.charAt(0).toUpperCase()}
      </div>
      <span className="hidden sm:inline text-sm font-medium text-muted-foreground">
        {displayName}
      </span>
    </div>
  );
}
