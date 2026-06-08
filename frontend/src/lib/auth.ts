export interface TokenClaims {
  sub: string
  role: string
  org_id?: string
}

export function getTokenClaims(): TokenClaims | null {
  if (typeof window === 'undefined') return null
  try {
    const token = localStorage.getItem('token')
    if (!token) return null
    const payload = token.split('.')[1]
    if (!payload) return null
    const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'))
    return JSON.parse(decoded) as TokenClaims
  } catch {
    console.error('[auth] Failed to decode JWT token claims')
    return null
  }
}

export function isAdmin(): boolean {
  const c = getTokenClaims()
  return c?.role === 'admin' || c?.role === 'super_admin'
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
