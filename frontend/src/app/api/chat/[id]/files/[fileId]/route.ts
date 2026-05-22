import { NextRequest } from "next/server";
import * as http from "http";

export const dynamic = "force-dynamic";

function proxy(req: NextRequest, method: string, chatId: string, fileId: string): Promise<Response> {
  const backendHost = process.env.BACKEND_HOST || "backend";
  const backendPort = parseInt(process.env.BACKEND_PORT || "8000", 10);
  const auth = req.headers.get("Authorization") ?? "";
  return new Promise((resolve) => {
    const options: http.RequestOptions = {
      hostname: backendHost, port: backendPort,
      path: `/api/chat/${chatId}/files/${fileId}`,
      method, headers: { Authorization: auth },
    };
    const proxyReq = http.request(options, (res) => {
      let data = "";
      res.on("data", (c) => { data += c; });
      res.on("end", () => resolve(new Response(data || null, { status: res.statusCode ?? 200, headers: { "Content-Type": "application/json" } })));
    });
    proxyReq.on("error", (err) => resolve(new Response(JSON.stringify({ error: err.message }), { status: 500 })));
    proxyReq.end();
  });
}

export async function GET(req: NextRequest, { params }: { params: { id: string; fileId: string } }) {
  return proxy(req, "GET", params.id, params.fileId);
}
export async function DELETE(req: NextRequest, { params }: { params: { id: string; fileId: string } }) {
  return proxy(req, "DELETE", params.id, params.fileId);
}
