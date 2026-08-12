export interface TokenClaims {
  sub: string
  role: string
  org_id?: number | null
}

// In-memory cache of user claims fetched from /api/auth/test-token.
let _cachedClaims: TokenClaims | null = null

/**
 * Fetch user claims from the backend test-token endpoint.
 * The JWT is stored in an HTTP-only cookie so we cannot parse it client-side.
 * Returns null if the request fails (e.g. not authenticated).
 */
export async function fetchTokenClaims(): Promise<TokenClaims | null> {
  if (_cachedClaims) return _cachedClaims
  try {
    const resp = await fetch('/api/auth/test-token')
    if (!resp.ok) return null
    const data = await resp.json()
    _cachedClaims = {
      sub: data.username ?? '',
      role: data.role ?? 'user',
      org_id: data.org_id ?? null,
    }
    return _cachedClaims
  } catch {
    return null
  }
}

/**
 * Synchronous accessor for cached claims. Call fetchTokenClaims() first
 * (e.g. in a useEffect) to populate the cache.
 */
export function getTokenClaims(): TokenClaims | null {
  return _cachedClaims
}

export function isAdmin(): boolean {
  const claims = getTokenClaims()
  return claims?.role === 'admin' || claims?.role === 'super_admin'
}

/** Clear cached claims (call on logout). */
export function clearTokenClaims(): void {
  _cachedClaims = null
}

export function validatePasswordStrength(password: string): string | null {
  // Validate password strength.
  // Returns null if valid, or a human-readable error message.
  // Rules: minimum 8 characters, must contain at least one letter and one digit.
  if (password.length < 8) return 'Password must be at least 8 characters';
  if (!/[a-zA-Z]/.test(password)) return 'Password must contain at least one letter';
  if (!/[0-9]/.test(password)) return 'Password must contain at least one number';
  return null;
}
