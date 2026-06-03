import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  // ── API proxy: inject real client IP for backend rate limiting ──
  if (request.nextUrl.pathname.startsWith('/api/')) {
    const forwarded = request.headers.get('x-forwarded-for');
    const realIP = forwarded
      ? forwarded.split(',')[0].trim()
      : request.headers.get('x-real-ip') || request.ip || 'unknown';

    const response = NextResponse.next();
    response.headers.set('X-Real-IP', realIP);
    return response;
  }

  // ── Dashboard auth: require valid token ──
  const token = request.cookies.get('token')?.value;

  if (request.nextUrl.pathname.startsWith('/dashboard/admin')) {
    if (!token) {
      return NextResponse.redirect(new URL('/login', request.url));
    }
    try {
      const rawPayload = Buffer.from(token.split('.')[1], 'base64url').toString();
      const claims = JSON.parse(rawPayload) as { role?: string };
      if (claims.role !== 'admin' && claims.role !== 'super_admin') {
        return NextResponse.redirect(new URL('/dashboard', request.url));
      }
    } catch {
      return NextResponse.redirect(new URL('/login', request.url));
    }
  }

  if (!token) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/api/:path*'],
};
