import asyncio
import random
import time
import uuid
from typing import Dict, Any
import httpx
from transformers import AutoTokenizer

# Configuration
API_URL = "http://192.168.1.4:8888/v1/chat/completions" # Change to your endpoint
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"       # Change to your model
API_KEY = "your-api-key-here"                       # Optional

# Load tokenizer once for accurate input token counting.
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Tokenizer loaded.")

def count_input_tokens(prompt: str) -> int:
    """Count input tokens using the model tokenizer."""
    return len(tokenizer.encode(prompt, add_special_tokens=True))


def generate_unique_prompt(target_tokens: int = 64000) -> str:
    """
    Build a prompt of approximately target_tokens that is unique on every run.
    A unique run ID plus random sentence content prevents the inference server
    from reusing a KV cache keyed on identical input token sequences.
    """
    run_id = uuid.uuid4().hex
    rng = random.Random(run_id)

    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
        "xray", "yankee", "zulu", "planet", "nebula", "quasar", "orbit", "galaxy",
        "rocket", "satellite", "meteor", "comet", "asteroid", "gravity", "fusion",
        "quantum", "prism", "spectrum", "voltage", "matrix", "tensor", "vector",
    ]

    # Start with a unique instruction so the prefix is never cached.
    instruction = (
        f"Run ID {run_id}. Summarize the following unique random text in exhaustive "
        "detail, preserving every single mention of the subject. "
        "The text consists of randomly generated sentences and begins now:\n\n"
    )

    # Estimate tokens remaining for the body after the instruction prefix.
    instruction_tokens = count_input_tokens(instruction)
    body_token_budget = max(target_tokens - instruction_tokens, 0)

    # Build body text in batches to minimize tokenizer calls.
    batch_sentences = []
    batch_token_estimate = 0
    body_text_parts = []

    # Avg sentence is ~15 tokens; generate enough to overshoot, then trim.
    tokens_per_sentence = 15
    sentences_needed = (body_token_budget // tokens_per_sentence) + 1000

    for _ in range(sentences_needed):
        sentence = (
            "The " + rng.choice(words) + " " + rng.choice(words)
            + " observed that the " + rng.choice(words) + " "
            + str(rng.randint(1000, 9999)) + " " + rng.choice(words)
            + " moved across the " + rng.choice(words) + " in a "
            + rng.choice(words) + " manner. "
        )
        batch_sentences.append(sentence)
        batch_token_estimate += tokens_per_sentence

        # Encode a batch every ~2k estimated tokens to stay near the budget.
        if batch_token_estimate >= 2000:
            body_text_parts.append("".join(batch_sentences))
            current_body = "".join(body_text_parts)
            current_total = count_input_tokens(instruction + current_body)
            if current_total >= target_tokens:
                break
            batch_sentences = []
            batch_token_estimate = 0

    if batch_sentences:
        body_text_parts.append("".join(batch_sentences))

    candidate = instruction + "".join(body_text_parts)

    # Trim exactly to target_tokens.
    encoded = tokenizer.encode(candidate, add_special_tokens=True)
    if len(encoded) > target_tokens:
        encoded = encoded[:target_tokens]
        candidate = tokenizer.decode(encoded, skip_special_tokens=True)

    actual_tokens = count_input_tokens(candidate)
    print(f"Generated unique prompt: {actual_tokens} tokens (target: {target_tokens})")
    return candidate

async def measure_llm_phases(prompt: str, max_tokens: int = 128) -> Dict[str, Any]:
    """
    Sends a streaming request to an OpenAI-compatible API to compute 
    Prefill Speed (TTFT), Prefill Throughput, and Decode Speed (TPOT / Tokens per Second).
    """
    input_tokens = count_input_tokens(prompt)
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }

    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", API_URL, json=payload, headers=headers, timeout=300.0) as response:
                if response.status_code != 200:
                    print(f"Error: Server returned status {response.status_code}")
                    return {}

                async for line in response.aiter_lines():
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue
                    
                    # Track timestamps on incoming data chunks
                    current_time = time.perf_counter()
                    token_count += 1

                    if first_token_time is None:
                        # Prefill stage ends when the first chunk/token arrives
                        first_token_time = current_time
                    
                    last_token_time = current_time

        except Exception as e:
            print(f"Connection or stream parsing error: {e}")
            return {}

    # Calculate Metrics
    if not first_token_time or token_count <= 1:
        return {"error": "Insufficient tokens received to measure performance."}

    # Prefill metrics
    prefill_latency_s = first_token_time - start_time
    prefill_latency_ms = prefill_latency_s * 1000
    prefill_throughput_tok_per_sec = input_tokens / prefill_latency_s if prefill_latency_s > 0 else 0

    # Decode metrics
    total_decode_time = last_token_time - first_token_time
    decode_tokens = token_count - 1 # Exclude the first token
    
    tpot_ms = (total_decode_time / decode_tokens) * 1000 if decode_tokens > 0 else 0
    tokens_per_sec = decode_tokens / total_decode_time if total_decode_time > 0 else 0

    return {
        "input_tokens": input_tokens,
        "prefill_latency_ttft_ms": round(prefill_latency_ms, 2),
        "prefill_throughput_tok_per_sec": round(prefill_throughput_tok_per_sec, 2),
        "decode_latency_tpot_ms": round(tpot_ms, 2),
        "decode_throughput_tok_per_sec": round(tokens_per_sec, 2),
        "total_generated_tokens": token_count
    }

async def main():
    # Build a unique ~64k-token prompt so the server cannot reuse a KV cache.
    test_prompt = generate_unique_prompt(target_tokens=250000)
    print("Sending request to measure prefill and decode speeds...")

    metrics = await measure_llm_phases(test_prompt, max_tokens=100)
    
    if "error" not in metrics and metrics:
        print("\n=== Benchmark Results ===")
        print(f"Input Tokens               : {metrics['input_tokens']}")
        print(f"Prefill Speed (TTFT)       : {metrics['prefill_latency_ttft_ms']} ms")
        print(f"Prefill Throughput         : {metrics['prefill_throughput_tok_per_sec']} tokens/sec")
        print(f"Decode Cost (TPOT)         : {metrics['decode_latency_tpot_ms']} ms/token")
        print(f"Decode Generation Speed    : {metrics['decode_throughput_tok_per_sec']} tokens/sec")
        print(f"Total Chunks Streamed      : {metrics['total_generated_tokens']}")
    else:
        print(f"Benchmark failed: {metrics.get('error', 'Unknown issue')}")

if __name__ == "__main__":
    asyncio.run(main())
