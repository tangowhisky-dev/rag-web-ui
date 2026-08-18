#!/usr/bin/env bash
# Download Whisper tiny.en model files (ONNX quantized) for local browser-based STT.
# Files are served from frontend/public/assets/models/ — no HuggingFace CDN at runtime.
set -euo pipefail

MODEL_DIR="frontend/public/assets/models/Xenova/whisper-tiny.en"
ONNX_DIR="$MODEL_DIR/onnx"
BASE_URL="https://huggingface.co/Xenova/whisper-tiny.en/resolve/main"

mkdir -p "$ONNX_DIR"

echo "Downloading Whisper tiny.en model files to $MODEL_DIR..."

# Config / tokenizer files
for f in config.json tokenizer.json tokenizer_config.json normalizer.json \
         preprocessor_config.json special_tokens_map.json vocab.json merges.txt; do
  if [ ! -f "$MODEL_DIR/$f" ]; then
    curl -sL -o "$MODEL_DIR/$f" "$BASE_URL/$f"
  fi
done

# ONNX model files (quantized = smallest, ~70MB total)
for f in encoder_model_quantized.onnx decoder_model_merged_quantized.onnx \
         decoder_with_past_model_quantized.onnx; do
  if [ ! -f "$ONNX_DIR/$f" ]; then
    echo "  Downloading $f..."
    curl -sL -o "$ONNX_DIR/$f" "$BASE_URL/onnx/$f"
  fi
done

echo "Done. Model files in $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
