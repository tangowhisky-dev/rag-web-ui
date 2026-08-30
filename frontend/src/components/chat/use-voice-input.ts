"use client";

import { useState, useRef, useCallback, useEffect, useSyncExternalStore } from "react";

const DEBUG = process.env.NEXT_PUBLIC_DEBUG === "true";
const log = (...args: any[]) => { if (DEBUG) console.log("[voice]", ...args); };
const logError = (...args: any[]) => console.error("[voice]", ...args);

/**
 * Real-time voice-to-text using Whisper (Transformers.js) running fully
 * in-browser via Web Workers. Model is served from /assets/models/.
 *
 * Smart chunking: uses energy-based Voice Activity Detection to send
 * audio at natural pause boundaries instead of fixed time slices.
 */

const SAMPLE_RATE = 16000;
const SILENCE_THRESHOLD = 0.01;   // RMS energy below this = silence (VAD)
const SPEECH_THRESHOLD = 0.05;    // max amplitude below this = no speech (chunk gate)
const SILENCE_DURATION_MS = 700;  // pause needed to trigger a chunk
const MIN_CHUNK_MS = 3000;        // Whisper needs >=3s to avoid hallucinations
const MAX_CHUNK_MS = 30000;       // Whisper's window limit (force send)

/**
 * Linear-interpolation resampler: convert Float32Array from srcRate to dstRate.
 */
function resample(input: Float32Array, srcRate: number, dstRate: number): Float32Array {
  if (srcRate === dstRate) return input;
  const ratio = srcRate / dstRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const low = Math.floor(srcIdx);
    const high = Math.min(low + 1, input.length - 1);
    const frac = srcIdx - low;
    out[i] = input[low] * (1 - frac) + input[high] * frac;
  }
  return out;
}

export function useVoiceInput(onFinal: (text: string) => void) {
  const [isListening, setIsListening] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [interim, setInterim] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);

  // Defer browser-capability check to after hydration so SSR and client
  // render the same initial UI (no hydration mismatch).
  const isSupported = useSyncExternalStore(
    () => () => {},
    () => !!window.AudioContext && typeof Worker !== "undefined",
    () => false,
  );

  const onFinalRef = useRef(onFinal);
  useEffect(() => {
    onFinalRef.current = onFinal;
  }, [onFinal]);

  const workerRef = useRef<Worker | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // PCM accumulation buffer
  const pcmBuffer = useRef<Float32Array>(new Float32Array(0));
  const silenceStartRef = useRef<number | null>(null);
  const chunkStartRef = useRef<number>(0);
  const vadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const ensureWorker = useCallback(async () => {
    if (workerRef.current) return workerRef.current;
    const worker = new Worker(new URL("./whisper-worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;
    worker.onerror = (e) => {
      logError("worker error:", e.message, e.filename, e.lineno);
    };
    return new Promise<Worker>((resolve, reject) => {
      const onReady = (e: MessageEvent) => {
        if (e.data.type === "ready") {
          worker.removeEventListener("message", onReady);
          resolve(worker);
        } else if (e.data.type === "error") {
          worker.removeEventListener("message", onReady);
          reject(new Error(e.data.error));
        }
      };
      worker.addEventListener("message", onReady);
      worker.postMessage({ type: "load" });
    });
  }, []);

  const sendChunk = useCallback(() => {
    const worker = workerRef.current;
    if (!worker) return;
    const chunk = pcmBuffer.current;
    if (chunk.length < (SAMPLE_RATE * MIN_CHUNK_MS) / 1000) {
      pcmBuffer.current = new Float32Array(0);
      return;
    }

    // Simple energy gate: skip chunks with no speech.
    let maxAmp = 0;
    for (let i = 0; i < chunk.length; i++) {
      const a = Math.abs(chunk[i]);
      if (a > maxAmp) maxAmp = a;
    }
    if (maxAmp < SPEECH_THRESHOLD) {
      pcmBuffer.current = new Float32Array(0);
      chunkStartRef.current = performance.now();
      silenceStartRef.current = null;
      return;
    }

    const copy = new Float32Array(chunk);
    worker.postMessage({ type: "transcribe", audio: copy }, [copy.buffer]);
    pcmBuffer.current = new Float32Array(0);
    chunkStartRef.current = performance.now();
    silenceStartRef.current = null;
  }, []);

  const stop = useCallback(() => {
    // Flush any remaining audio
    sendChunk();

    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setIsListening(false);
    setInterim("");
    setAudioLevel(0);
  }, [sendChunk]);

  const start = useCallback(async () => {
    if (!isSupported) return;
    try {
      setIsModelLoading(true);
      const worker = await ensureWorker();
      setIsModelLoading(false);

      // Set up worker message handler for transcription results
      worker.onmessage = (e: MessageEvent) => {
        if (e.data.type === "partial") {
          setInterim(e.data.text);
        } else if (e.data.type === "result") {
          const text = (e.data.text as string).trim();
          if (text) onFinalRef.current(text);
          setInterim("");
        } else if (e.data.type === "error") {
          logError("worker error:", e.data.error);
          setInterim("");
        }
      };

      // Set up audio capture — let the AudioContext use the hardware sample
      // rate. Forcing 16kHz causes silent input on some browsers (Chrome).
      // We resample to 16kHz before sending to Whisper.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const ctxSampleRate = audioCtx.sampleRate;
      log(`AudioContext sampleRate=${ctxSampleRate}, will resample to ${SAMPLE_RATE}`);

      const source = audioCtx.createMediaStreamSource(stream);

      // AnalyserNode for VAD (energy detection)
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      source.connect(analyser);

      // ScriptProcessorNode to capture PCM samples
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        const input = e.inputBuffer.getChannelData(0);
        // Log first capture to verify data is flowing
        if (pcmBuffer.current.length === 0) {
          let max = 0;
          for (let i = 0; i < input.length; i++) max = Math.max(max, Math.abs(input[i]));
          log(`first PCM frame: ${input.length} samples @ ${ctxSampleRate}Hz, max amplitude=${max.toFixed(4)}`);
        }
        // Resample from hardware rate to 16kHz, then append to buffer
        const resampled = resample(input, ctxSampleRate, SAMPLE_RATE);
        const newBuf = new Float32Array(pcmBuffer.current.length + resampled.length);
        newBuf.set(pcmBuffer.current);
        newBuf.set(resampled, pcmBuffer.current.length);
        pcmBuffer.current = newBuf;
      };
      source.connect(processor);
      // ScriptProcessorNode must connect to destination to fire onaudioprocess,
      // but we route through a zero-gain node to avoid echo.
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(audioCtx.destination);

      chunkStartRef.current = performance.now();
      silenceStartRef.current = null;
      setIsListening(true);

      // VAD loop — check energy every 100ms
      const timeData = new Float32Array(analyser.fftSize);
      vadIntervalRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getFloatTimeDomainData(timeData);

        // Compute RMS energy
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) sum += timeData[i] * timeData[i];
        const rms = Math.sqrt(sum / timeData.length);
        setAudioLevel(Math.min(rms * 5, 1)); // normalize for display

        const now = performance.now();
        const chunkDuration = now - chunkStartRef.current;

        if (rms < SILENCE_THRESHOLD) {
          // Silence detected
          if (silenceStartRef.current === null) {
            silenceStartRef.current = now;
          }
          const silenceDuration = now - silenceStartRef.current;
          if (
            silenceDuration > SILENCE_DURATION_MS &&
            chunkDuration > MIN_CHUNK_MS
          ) {
            // Natural pause boundary — send the chunk
            sendChunk();
          }
        } else {
          // Speaking — reset silence timer
          silenceStartRef.current = null;
        }

        // Force send if buffer reaches max window
        if (chunkDuration > MAX_CHUNK_MS) {
          sendChunk();
        }
      }, 100);
    } catch (err) {
      logError("failed to start:", err);
      setIsModelLoading(false);
      setIsListening(false);
    }
  }, [isSupported, ensureWorker, sendChunk]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [stop]);

  return {
    isListening,
    isModelLoading,
    isSupported,
    interim,
    audioLevel,
    start,
    stop,
  };
}
