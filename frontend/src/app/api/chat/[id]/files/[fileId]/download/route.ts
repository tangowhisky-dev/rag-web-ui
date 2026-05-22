import { NextRequest } from "next/server";
import * as http from "http";

export const dynamic = "force-dynamic";

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string; fileId: string } }
) {
  const backendHost = process.env.BACKEND_HOST || "backend";
  const backendPort = parseInt(process.env.BACKEND_PORT || "8000", 10);
  const auth = req.headers.get("Authorization") ?? "";

  return new Promise<Response>((resolve) => {
    const options: http.RequestOptions = {
      hostname: backendHost,
      port: backendPort,
      path: `/api/chat/${params.id}/files/${params.fileId}/download`,
      method: "GET",
      headers: { Authorization: auth },
    };

    const proxyReq = http.request(options, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        const body = Buffer.concat(chunks);
        const headers: Record<string, string> = {};
        if (res.headers["content-type"]) headers["Content-Type"] = res.headers["content-type"] as string;
        if (res.headers["content-disposition"]) headers["Content-Disposition"] = res.headers["content-disposition"] as string;
        if (res.headers["content-length"]) headers["Content-Length"] = res.headers["content-length"] as string;
        resolve(new Response(body, { status: res.statusCode ?? 200, headers }));
      });
    });
    proxyReq.on("error", (err) =>
      resolve(new Response(JSON.stringify({ error: err.message }), { status: 500 }))
    );
    proxyReq.end();
  });
}
