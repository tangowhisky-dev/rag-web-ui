import { NextRequest } from "next/server";
import * as http from "http";

export const dynamic = "force-dynamic";

/**
 * Streaming proxy for multipart file-upload messages.
 * Mirrors the /messages proxy — pipes raw TCP chunks so streaming works.
 * Next.js built-in fetch buffers the full response; node http.request does not.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const backendHost = process.env.BACKEND_HOST || "backend";
  const backendPort = parseInt(process.env.BACKEND_PORT || "8000", 10);
  const auth = req.headers.get("Authorization") ?? "";
  const contentType = req.headers.get("Content-Type") ?? "";

  // Read the raw multipart body as an ArrayBuffer and forward it verbatim
  const bodyBuffer = Buffer.from(await req.arrayBuffer());

  const stream = new ReadableStream({
    start(controller) {
      const options: http.RequestOptions = {
        hostname: backendHost,
        port: backendPort,
        path: `/api/chat/${params.id}/messages/with-file`,
        method: "POST",
        headers: {
          "Content-Type": contentType,
          "Content-Length": bodyBuffer.length,
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
      proxyReq.write(bodyBuffer);
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
