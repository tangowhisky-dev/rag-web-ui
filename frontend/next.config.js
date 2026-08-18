/** @type {import('next').NextConfig} */
module.exports = {
  output: "standalone",
  skipMiddlewareUrlNormalize: true,
  skipTrailingSlashRedirect: true,
  experimental: {
    outputFileTracingRoot: undefined,
    proxyTimeout: 120000,
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
};
