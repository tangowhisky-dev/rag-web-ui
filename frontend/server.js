/**
 * Custom Next.js dev server.
 *
 * Intercepts POST /api/chat/<id>/messages and pipes it directly to the
 * backend using raw Node.js http.request(). This bypasses the Next.js
 * request pipeline (which uses undici and buffers streaming responses)
 * so SSE chunks reach the browser immediately as they arrive.
 *
 * All other routes are handled normally by Next.js (HMR still works).
 */

const { createServer } = require("http");
const { parse } = require("url");
const next = require("next");
const http = require("http");

const dev = process.env.NODE_ENV !== "production";
const app = next({ dev });
const handle = app.getRequestHandler();

const BACKEND_HOST = process.env.BACKEND_HOST || "backend";
const BACKEND_PORT = parseInt(process.env.BACKEND_PORT || "8000", 10);

// Matches /api/chat/<id>/messages  (id = any non-slash segment)
const STREAMING_ROUTE = /^\/api\/chat\/[^/]+\/messages$/;

app.prepare().then(() => {
  createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    const { pathname } = parsedUrl;

    if (req.method === "POST" && STREAMING_ROUTE.test(pathname)) {
      const chunks = [];
      req.on("data", (chunk) => chunks.push(chunk));
      req.on("end", () => {
        const body = Buffer.concat(chunks);

        const proxyReq = http.request(
          {
            hostname: BACKEND_HOST,
            port: BACKEND_PORT,
            path: pathname,
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: req.headers["authorization"] || "",
              "Content-Length": body.length,
            },
          },
          (proxyRes) => {
            res.writeHead(proxyRes.statusCode, {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache, no-transform",
              "X-Accel-Buffering": "no",
            });

            // Disable Nagle algorithm so each write is flushed immediately
            if (res.socket) res.socket.setNoDelay(true);

            proxyRes.on("data", (chunk) => res.write(chunk));
            proxyRes.on("end", () => res.end());
            proxyRes.on("error", () => res.end());
          }
        );

        proxyReq.on("error", (err) => {
          console.error("[stream-proxy] backend error:", err.message);
          if (!res.headersSent) res.writeHead(502);
          res.end();
        });

        proxyReq.write(body);
        proxyReq.end();
      });

      return;
    }

    // All other routes → Next.js
    handle(req, res, parsedUrl);
  }).listen(3000, (err) => {
    if (err) throw err;
    console.log("> Ready on http://localhost:3000");
  });
});
