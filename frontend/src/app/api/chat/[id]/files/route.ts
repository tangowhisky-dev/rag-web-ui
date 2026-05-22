import { NextRequest } from "next/server";
import * as http from "http";

export const dynamic = "force-dynamic";

/** Proxy: POST /api/chat/[id]/files — multipart file upload (no streaming needed) */
export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const backendHost = process.env.BACKEND_HOST || "backend";
  const backendPort = parseInt(process.env.BACKEND_PORT || "8000", 10);
  const auth = req.headers.get("Authorization") ?? "";
  const contentType = req.headers.get("Content-Type") ?? "";
  const bodyBuffer = Buffer.from(await req.arrayBuffer());

  return new Promise<Response>((resolve) => {
    const options: http.RequestOptions = {
      hostname: backendHost,
      port: backendPort,
      path: `/api/chat/${params.id}/files`,
      method: "POST",
      headers: { "Content-Type": contentType, "Content-Length": bodyBuffer.length, Authorization: auth },
    };
    const proxyReq = http.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => resolve(new Response(data, { status: res.statusCode ?? 200, headers: { "Content-Type": "application/json" } })));
    });
    proxyReq.on("error", (err) => resolve(new Response(JSON.stringify({ error: err.message }), { status: 500 })));
    proxyReq.write(bodyBuffer);
    proxyReq.end();
  });
}
