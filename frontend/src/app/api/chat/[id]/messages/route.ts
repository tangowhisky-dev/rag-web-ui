import { NextRequest } from "next/server";
import * as http from "http";

export const dynamic = "force-dynamic";

/**
 * Streaming proxy for chat messages.
 *
 * Node.js http.request() pipes raw TCP chunks directly into a ReadableStream
 * without any intermediate buffering — unlike undici (Next.js built-in fetch)
 * which accumulates the full response body before delivering it.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const backendHost = process.env.BACKEND_HOST || "backend";
  const backendPort = parseInt(process.env.BACKEND_PORT || "8000", 10);
  const body = await req.text();
  const auth = req.headers.get("Authorization") ?? "";

  const stream = new ReadableStream({
    start(controller) {
      const options: http.RequestOptions = {
        hostname: backendHost,
        port: backendPort,
        path: `/api/chat/${params.id}/messages`,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: auth,
          Connection: "keep-alive",
        },
      };

      const proxyReq = http.request(options, (res) => {
        res.on("data", (chunk: Buffer) => controller.enqueue(chunk));
        res.on("end", () => controller.close());
        res.on("error", (err) => controller.error(err));
      });

      proxyReq.on("error", (err) => controller.error(err));
      proxyReq.write(body);
      proxyReq.end();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
