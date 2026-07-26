export interface TokenClaims {
  sub: string
  role: string
  org_id?: string
}

export function getTokenClaims(): TokenClaims | null {
  return null
}

export function isAdmin(): boolean {
  return false
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
