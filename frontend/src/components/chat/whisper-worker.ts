/// <reference lib="webworker" />

import { pipeline, env } from "@huggingface/transformers";

// Serve model from our own server, never download from HuggingFace.
env.allowRemoteModels = false;
env.localModelPath = "/assets/";
env.allowLocalModels = true;

// Use a separate ONNX runtime thread pool inside the worker.
if (env.backends.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

let transcriber: any = null;
let loading: Promise<any> | null = null;

async function getTranscriber() {
  if (transcriber) return transcriber;
  if (loading) return loading;
  loading = pipeline("automatic-speech-recognition", "whisper", {
    dtype: "q8",
  }).then((t) => {
    transcriber = t;
    loading = null;
    return t;
  });
  return loading;
}

self.addEventListener("message", async (e: MessageEvent) => {
  const { type, audio } = e.data;
  if (type === "load") {
    try {
      await getTranscriber();
      (self as any).postMessage({ type: "ready" });
    } catch (err) {
      (self as any).postMessage({ type: "error", error: String(err) });
    }
    return;
  }
  if (type === "transcribe") {
    try {
      const t = await getTranscriber();
      const output = await t(audio, {
        return_timestamps: false,
        chunk_length_s: 30,
        stride_chunk_length_s: 5,
      });
      (self as any).postMessage({ type: "result", text: output.text.trim() });
    } catch (err) {
      (self as any).postMessage({ type: "error", error: String(err) });
    }
    return;
  }
});
