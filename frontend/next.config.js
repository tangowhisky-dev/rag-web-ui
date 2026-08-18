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
    // @huggingface/transformers ships a Node.js bundle (transformers.node.mjs)
    // that imports onnxruntime-node and native WASM files. The whisper worker
    // runs in the browser and only needs the WASM (web) bundle.
    const browserBundle = require.resolve(
      "@huggingface/transformers/dist/transformers.web.js"
    );
    config.resolve.alias = {
      ...config.resolve.alias,
      "@huggingface/transformers": browserBundle,
    };
    // Externalize native bindings so webpack never tries to parse .node files.
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
