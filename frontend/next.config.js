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
    // @huggingface/transformers pulls in onnxruntime-node which ships
    // native .node binaries. Webpack can't parse these — exclude them.
    // The worker only uses onnxruntime-web (WASM), not the native binding.
    config.externals = config.externals || [];
    config.externals.push({
      "onnxruntime-node": "commonjs onnxruntime-node",
    });
    if (isServer) {
      config.externals.push({
        "onnxruntime-web": "commonjs onnxruntime-web",
      });
    }
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
