import { NextRequest, NextResponse } from "next/server";

/**
 * Streaming proxy for chat messages.
 *
 * Next.js `rewrites` buffer the full response body before forwarding it to the
 * browser, which breaks SSE streaming.  By handling the route here we can pipe
 * the ReadableStream from FastAPI straight to the client without any
 * intermediate buffering.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const backendUrl =
    process.env.BACKEND_URL || "http://backend:8000";

  const body = await req.text();
  const auth = req.headers.get("Authorization") ?? "";

  const upstream = await fetch(
    `${backendUrl}/api/chat/${params.id}/messages`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: auth,
      },
      body,
    }
  );

  // Pipe the ReadableStream straight through — no buffering
  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      // Tell any intermediate proxies/nginx not to buffer
      "X-Accel-Buffering": "no",
      // Required by the Vercel AI SDK useChat hook
      "x-vercel-ai-data-stream": "v1",
    },
  });
}
