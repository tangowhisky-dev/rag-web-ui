/// <reference lib="webworker" />

import { pipeline, env, TextStreamer } from "@huggingface/transformers";

const DEBUG = process.env.NEXT_PUBLIC_DEBUG === "true";
const log = (...args: any[]) => { if (DEBUG) console.log("[whisper-worker]", ...args); };
const logError = (...args: any[]) => console.error("[whisper-worker]", ...args);

// Serve model from our own server, never download from HuggingFace.
env.allowRemoteModels = false;
env.localModelPath = "/assets/";
env.allowLocalModels = true;

// Force WASM backend — the quantized ONNX model is incompatible with WebGPU
// (missing dequantization scales for QDQ nodes).
if (env.backends.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
}

let transcriber: any = null;
let loading: Promise<any> | null = null;

async function getTranscriber() {
  if (transcriber) return transcriber;
  if (loading) return loading;
  loading = pipeline("automatic-speech-recognition", "whisper", {
    dtype: "q8",
    device: "wasm",
    session_options: {
      graphOptimizationLevel: "basic",
    },
  }).then((t) => {
    transcriber = t;
    loading = null;
    return t;
  }).catch((err) => {
    loading = null;
    throw err;
  });
  return loading;
}

self.addEventListener("message", async (e: MessageEvent) => {
  const { type, audio } = e.data;
  if (type === "load") {
    try {
      log("loading model…");
      await getTranscriber();
      log("model loaded, posting ready");
      (self as any).postMessage({ type: "ready" });
    } catch (err) {
      logError("load failed:", err);
      (self as any).postMessage({ type: "error", error: String(err) });
    }
    return;
  }
  if (type === "transcribe") {
    try {
      // Gate: skip silent audio (all zeros or near-zero energy).
      // Whisper hallucinates "you" on silent input.
      let maxVal = 0;
      for (let i = 0; i < audio.length; i++) {
        const abs = Math.abs(audio[i]);
        if (abs > maxVal) maxVal = abs;
      }
      if (maxVal < 0.001) {
        log("skipping silent chunk");
        (self as any).postMessage({ type: "result", text: "" });
        return;
      }

      const t = await getTranscriber();
      log(`transcribing ${audio.length} samples (${(audio.length / 16000).toFixed(1)}s) max=${maxVal.toFixed(4)}`);

      // Stream partial text to the main thread as tokens are generated.
      const streamer = new TextStreamer(t.tokenizer, {
        skip_prompt: true,
        callback_function: (text: string) => {
          (self as any).postMessage({ type: "partial", text });
        },
      });

      const output = await t(audio, {
        return_timestamps: false,
        chunk_length_s: 30,
        stride_chunk_length_s: 5,
        streamer,
      });
      log("result:", output.text);
      (self as any).postMessage({ type: "result", text: output.text.trim() });
    } catch (err) {
      logError("transcribe failed:", err);
      (self as any).postMessage({ type: "error", error: String(err) });
    }
    return;
  }
});
