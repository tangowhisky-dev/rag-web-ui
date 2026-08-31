/** @type {import('next').NextConfig} */

// Allow HMR/dev resource access from LAN IPs. Set ALLOWED_DEV_ORIGINS
// env var to the server's LAN address, e.g. http://192.168.1.21:3000
const devOrigins = (process.env.ALLOWED_DEV_ORIGINS || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

module.exports = {
  output: "standalone",
  skipProxyUrlNormalize: true,
  skipTrailingSlashRedirect: true,
  allowedDevOrigins: [
    "0.0.0.0",
    "127.0.0.1",
    "localhost",
    ...devOrigins,
  ],
  experimental: {
    proxyTimeout: 120000,
  },
  turbopack: {
    resolveAlias: {
      "@huggingface/transformers":
        "./node_modules/@huggingface/transformers/dist/transformers.web.js",
    },
  },
  webpack: (config, { isServer }) => {
    // @huggingface/transformers resolves to the Node.js bundle under the
    // "node" exports condition, which pulls in onnxruntime-node (native
    // .node binary) and WASM files webpack can't bundle. The whisper worker
    // runs in the browser, so alias to the browser bundle directly.
    const fs = require("fs");
    const path = require("path");
    const realDir = fs.realpathSync(
      path.resolve("node_modules/@huggingface/transformers")
    );
    config.resolve.alias = {
      ...config.resolve.alias,
      "@huggingface/transformers": path.join(realDir, "dist", "transformers.web.js"),
    };
    config.externals = config.externals || [];
    config.externals.push({
      "onnxruntime-node": "commonjs onnxruntime-node",
    });

    // onnxruntime-web ships pre-bundled .mjs files that use `import.meta.url`
    // to locate their WASM assets at runtime. Terser (used by Next.js
    // production builds) fails to parse `import.meta` unless the file is
    // explicitly treated as an ES module. Without this rule, `next build`
    // fails with: "'import.meta' cannot be used outside of module code".
    config.module = config.module || {};
    config.module.rules = config.module.rules || [];
    config.module.rules.push({
      test: /\.mjs$/,
      include: /onnxruntime-web/,
      type: "javascript/esm",
    });

    return config;
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL || "http://backend:8000"}/api/:path*`,
      },
      {
        source: "/assets/:path*",
        destination: `${process.env.BACKEND_URL || "http://backend:8000"}/assets/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "SAMEORIGIN" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-XSS-Protection", value: "1; mode=block" },
        ],
      },
    ];
  },
};
