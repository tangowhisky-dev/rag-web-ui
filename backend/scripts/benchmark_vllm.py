#!/usr/bin/env python3
"""
Benchmark vLLM server at 192.168.1.4:8888
Measures prefill (TTFT) and decode throughput for various context lengths.
"""
import time
import json
import urllib.request
import urllib.error
import io

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
BASE_URL = "http://192.168.1.4:8888/v1/chat/completions"
MAX_GEN = 16384  # 16K generation

# Use Qwen2 tokenizer (compatible vocabulary)
from transformers import AutoTokenizer
enc = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)


def make_prompt_tokens(n_tokens):
    """Generate a prompt with approximately n_tokens tokens."""
    words = "The quick brown fox jumps over the lazy dog. " * 1000
    tokens = enc.encode(words, add_special_tokens=False)
    tokens_per_block = len(tokens)
    blocks_needed = (n_tokens + tokens_per_block - 1) // tokens_per_block
    full_text = words * blocks_needed
    truncated = enc.decode(enc.encode(full_text, add_special_tokens=False)[:n_tokens])
    actual_count = len(enc.encode(truncated, add_special_tokens=False))
    return truncated, actual_count


def stream_completion(messages, max_tokens, temperature=0.7):
    """Send streaming request and measure TTFT + decode throughput using line-by-line SSE."""
    data = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    ttft = None
    token_times = []
    total_tokens = 0
    full_text_parts = []
    stop_reason = None

    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            # Read line by line - standard SSE parsing
            for line in io.TextIOWrapper(response, encoding="utf-8", errors="replace", line_buffering=True):
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    stop_reason = "done"
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "choices" not in event or len(event["choices"]) == 0:
                    continue
                delta = event["choices"][0].get("delta", {})
                # Handle reasoning models: content may be in "reasoning" field
                token_text = delta.get("content") or delta.get("reasoning") or ""
                if token_text:
                    if ttft is None:
                        ttft = time.monotonic() - start_mono
                    token_times.append(time.monotonic())
                    total_tokens += 1
                    if total_tokens <= 3:
                        full_text_parts.append(token_text)
                finish = event["choices"][0].get("finish_reason")
                if finish:
                    stop_reason = finish
    except urllib.error.URLError as e:
        return {
            "error": str(e),
            "total_tokens": total_tokens,
            "stop_reason": stop_reason,
        }
    except Exception as e:
        # If TextIOWrapper fails, fall back to reading all data and splitting
        return {
            "error": f"Stream error: {e}",
            "total_tokens": total_tokens,
            "stop_reason": stop_reason,
        }

    total_time = time.monotonic() - start_mono
    stop_reason = stop_reason or "max_tokens"

    if len(token_times) >= 2:
        decode_start = start_mono + ttft
        decode_end = token_times[-1]
        decode_duration = decode_end - decode_start
        tps = total_tokens / decode_duration if decode_duration > 0 else 0
    else:
        decode_duration = 0
        tps = 0

    return {
        "ttft_seconds": round(ttft, 2) if ttft else None,
        "total_tokens": total_tokens,
        "total_time_seconds": round(total_time, 2),
        "decode_duration_seconds": round(decode_duration, 2) if decode_duration else None,
        "tokens_per_second": round(tps, 2),
        "full_text_preview": "".join(full_text_parts)[:200] if total_tokens > 0 else "",
        "stop_reason": stop_reason,
    }


def benchmark(context_len, label):
    print(f"\n{'='*70}")
    print(f"  Benchmark: {label} ({context_len:,} tokens input, {MAX_GEN:,} tokens gen)")
    print(f"{'='*70}")

    prompt_text, actual_tokens = make_prompt_tokens(context_len)
    print(f"  Target: {context_len:,} tokens | Actual: {actual_tokens:,} tokens")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    global start_mono
    start_mono = time.monotonic()

    result = stream_completion(messages, MAX_GEN)

    print(f"  Prefill (TTFT):    {result['ttft_seconds']}s" if result.get('ttft_seconds') else f"  Prefill (TTFT):    TIMEOUT or error")
    print(f"  Tokens generated:  {result['total_tokens']}")
    print(f"  Decode duration:   {result['decode_duration_seconds']}s" if result.get('decode_duration_seconds') else f"  Decode duration:   N/A")
    print(f"  Decode throughput: {result['tokens_per_second']} tok/s" if result.get('tokens_per_second') else f"  Decode throughput: N/A")
    print(f"  Total time:        {result['total_time_seconds']}s")
    print(f"  Stop reason:       {result.get('stop_reason', 'unknown')}")

    if result.get("error"):
        print(f"  ERROR: {result['error']}")

    return result


if __name__ == "__main__":
    print("vLLM Server Benchmark")
    print(f"  Server: {BASE_URL}")
    print(f"  Model:  {MODEL}")
    print(f"  Gen:    {MAX_GEN} tokens")

    results = {}
    for ctx_len, label in [
        (64000, "64K"),
        (128000, "128K"),
        (200000, "200K"),
    ]:
        results[label] = benchmark(ctx_len, label)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Context':<15} {'TTFT (s)':<12} {'Tokens':<10} {'Decode (s)':<12} {'TPS':<10}")
    print(f"  {'-'*60}")
    for label, r in results.items():
        ttft = str(r['ttft_seconds']) if r.get('ttft_seconds') else "N/A"
        tokens = str(r['total_tokens'])
        decode = str(r['decode_duration_seconds']) if r.get('decode_duration_seconds') else "N/A"
        tps = str(r['tokens_per_second']) if r.get('tokens_per_second') else "N/A"
        print(f"  {label+'K':<15} {ttft:<12} {tokens:<10} {decode:<12} {tps:<10}")
    print(f"{'='*70}")
