import { NextRequest, NextResponse } from 'next/server';

export function proxy(request: NextRequest) {
  // ── API proxy: inject real client IP for backend rate limiting ──
  if (request.nextUrl.pathname.startsWith('/api/')) {
    const forwarded = request.headers.get('x-forwarded-for');
    const realIP = forwarded
      ? forwarded.split(',')[0].trim()
      : request.headers.get('x-real-ip') || 'unknown';

    const response = NextResponse.next();
    response.headers.set('X-Real-IP', realIP);
    return response;
  }

  // ── Dashboard auth: require token cookie ──
  const token = request.cookies.get('token')?.value;

  if (request.nextUrl.pathname.startsWith('/dashboard')) {
    if (!token) {
      return NextResponse.redirect(new URL('/', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/api/:path*'],
};
