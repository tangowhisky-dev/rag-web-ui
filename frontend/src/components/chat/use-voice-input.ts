"use client";

import { useState, useRef, useCallback, useEffect } from "react";

/**
 * Real-time voice-to-text using Whisper (Transformers.js) running fully
 * in-browser via Web Workers. Model is served from /assets/models/.
 *
 * Smart chunking: uses energy-based Voice Activity Detection to send
 * audio at natural pause boundaries instead of fixed time slices.
 */

const SAMPLE_RATE = 16000;
const SILENCE_THRESHOLD = 0.01;   // RMS energy below this = silence
const SILENCE_DURATION_MS = 350;  // pause needed to trigger a chunk
const MIN_CHUNK_MS = 1000;        // don't send chunks shorter than this
const MAX_CHUNK_MS = 30000;       // Whisper's window limit (force send)

export function useVoiceInput(onFinal: (text: string) => void) {
  const [isListening, setIsListening] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [interim, setInterim] = useState("");
  const [isSupported, setIsSupported] = useState(false);

  // Defer browser-capability check to after hydration so SSR and client
  // render the same initial UI (no hydration mismatch).
  useEffect(() => {
    setIsSupported(
      !!window.AudioContext && typeof Worker !== "undefined"
    );
  }, []);

  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

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
      console.error("[voice] worker error:", e.message, e.filename, e.lineno);
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
    if (chunk.length < (SAMPLE_RATE * MIN_CHUNK_MS) / 1000) return;

    // Send a copy and reset the buffer
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
  }, [sendChunk]);

  const start = useCallback(async () => {
    if (!isSupported) return;
    try {
      setIsModelLoading(true);
      const worker = await ensureWorker();
      setIsModelLoading(false);

      // Set up worker message handler for transcription results
      worker.onmessage = (e: MessageEvent) => {
        if (e.data.type === "result") {
          const text = (e.data.text as string).trim();
          if (text) onFinalRef.current(text);
          setInterim("");
        } else if (e.data.type === "error") {
          console.error("[voice] worker error:", e.data.error);
          setInterim("");
        }
      };

      // Set up audio capture
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = audioCtx;

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
        // Append to buffer
        const newBuf = new Float32Array(pcmBuffer.current.length + input.length);
        newBuf.set(pcmBuffer.current);
        newBuf.set(input, pcmBuffer.current.length);
        pcmBuffer.current = newBuf;
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);

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
          setInterim("listening…");
        }

        // Force send if buffer reaches max window
        if (chunkDuration > MAX_CHUNK_MS) {
          sendChunk();
        }
      }, 100);
    } catch (err) {
      console.error("[voice] failed to start:", err);
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
    start,
    stop,
  };
}
