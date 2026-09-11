#!/usr/bin/env python3
"""
Presenton REST API test script.

Tests the full presentation generation workflow against a self-hosted
Presenton instance running on 192.168.1.3:5001, backed by LM Studio
(qwen/qwen3.5-9b) on port 2244.

Usage:
    python scripts/presenton_test.py

API reference: https://docs.presenton.ai/api-reference/

Key endpoints:
    POST /api/v1/auth/login           — session-based auth (cookie)
    POST /api/v1/ppt/presentation/generate/async  — start async generation
    GET  /api/v1/async-tasks/status/{id}          — poll task status
    GET  /api/v1/ppt/presentation/all             — list presentations
    GET  /api/v1/ppt/presentation/{id}            — get presentation details
    POST /api/v1/ppt/presentation/{id}/export      — export PPTX/PDF
    DELETE /api/v1/ppt/presentation/{id}          — delete presentation

Auth:
    - Session: POST /api/v1/auth/login, use cookie jar
    - API key: Authorization: Bearer sk-presenton-...
    - BasicAuth also documented but session/API-key are more reliable

GeneratePresentationRequest fields:
    content (required)       — prompt text
    n_slides (optional)       — number of slides (auto-detected if omitted)
    language (optional)       — presentation language
    export_as (default pptx)  — "pptx" or "pdf"
    template (default general)— template name
    tone (default default)    — default|casual|professional|funny|educational|sales_pitch
    verbosity (default standard) — concise|standard|text-heavy
    instructions (optional)   — additional instructions
    slides_markdown (optional)— pre-written slide markdown
    files (optional)         — uploaded file IDs
    web_search (default false)— enable web search
    include_table_of_contents (default false)
    include_title_slide (default true)
"""

import requests
import time
import json
import sys

HOST = "http://192.168.1.3:5001"
API_KEY = "sk-presenton-68c0d48df55278c8.PTtH2At5FzOrnJ7snk7X7uMXx2elTwZHHrXpbFUVK0CSxSYX5DPTNg"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def generate_presentation(content, n_slides=None, language="English", export_as="pptx",
                          template="general", tone="default", verbosity="standard",
                          instructions=None, web_search=False):
    """Start async presentation generation. Returns task dict."""
    payload = {
        "content": content,
        "language": language,
        "export_as": export_as,
        "template": template,
        "tone": tone,
        "verbosity": verbosity,
        "web_search": web_search,
    }
    if n_slides is not None:
        payload["n_slides"] = n_slides
    if instructions:
        payload["instructions"] = instructions

    resp = requests.post(f"{HOST}/api/v1/ppt/presentation/generate/async",
                         headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def poll_task(task_id, interval=10, max_wait=300):
    """Poll async task status until completed or error."""
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(f"{HOST}/api/v1/async-tasks/status/{task_id}",
                            headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"  Status poll error: {resp.status_code} {resp.text[:200]}")
            time.sleep(interval)
            continue
        data = resp.json()
        status = data.get("status", "unknown")
        msg = data.get("message", "")
        slides = data.get("data", {}).get("created_slides", 0)
        remaining = data.get("data", {}).get("remaining_slides", "?")
        print(f"  status={status} slides={slides} remaining={remaining} msg={msg}")
        if status in ("completed", "error"):
            return data
        time.sleep(interval)
    return {"status": "timeout", "error": "max_wait exceeded"}


def get_presentation(presentation_id):
    """Get full presentation with slides."""
    resp = requests.get(f"{HOST}/api/v1/ppt/presentation/{presentation_id}",
                        headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def list_presentations():
    """List all presentations."""
    resp = requests.get(f"{HOST}/api/v1/ppt/presentation/all",
                        headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def export_presentation(presentation_id, format="pptx"):
    """Export presentation as PPTX or PDF."""
    resp = requests.post(f"{HOST}/api/v1/ppt/presentation/{presentation_id}/export",
                         headers=HEADERS, json={"format": format}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def delete_presentation(presentation_id):
    """Delete a presentation."""
    resp = requests.delete(f"{HOST}/api/v1/ppt/presentation/{presentation_id}",
                           headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    print("=== Presenton REST API Test ===\n")

    # 1. List existing presentations
    print("1. Listing presentations...")
    presentations = list_presentations()
    print(f"   Found {len(presentations)} presentation(s)")
    for p in presentations:
        print(f"   - {p.get('id')}: {p.get('title', 'N/A')[:60]}")

    # 2. Generate a new presentation
    print("\n2. Generating test presentation...")
    task = generate_presentation(
        content="A 5-slide overview of how AI is transforming healthcare: diagnostics, drug discovery, personalized medicine, robotic surgery, and ethical considerations.",
        n_slides=5,
        language="English",
        export_as="pptx",
        tone="professional",
        verbosity="concise",
    )
    task_id = task["id"]
    presentation_id = task.get("data", {}).get("presentation_id")
    print(f"   Task ID: {task_id}")
    print(f"   Presentation ID: {presentation_id}")
    print(f"   Status: {task['status']}")

    # 3. Poll for completion
    print("\n3. Polling for completion...")
    result = poll_task(task_id, interval=10, max_wait=300)
    print(f"   Final status: {result['status']}")

    if result["status"] != "completed":
        print(f"   Error: {result.get('error', 'unknown')}")
        sys.exit(1)

    ppt_path = result.get("data", {}).get("path", "")
    print(f"   PPTX path: {ppt_path}")

    # 4. Get presentation details
    print("\n4. Getting presentation details...")
    if presentation_id:
        details = get_presentation(presentation_id)
        print(f"   Title: {details.get('title', 'N/A')[:80]}")
        slides = details.get("slides", [])
        print(f"   Slides: {len(slides)}")
        for i, slide in enumerate(slides, 1):
            layout = slide.get("layout", "unknown")
            content = slide.get("content", {})
            # Extract title from content blocks
            title = ""
            for key in ("centered_title_block", "title_block", "heading_block"):
                if key in content:
                    title = content[key].get("main_heading", content[key].get("title", ""))
                    break
            print(f"   Slide {i}: layout={layout} title={title[:60]}")

    # 5. List all presentations again
    print("\n5. Listing presentations after generation...")
    presentations = list_presentations()
    print(f"   Found {len(presentations)} presentation(s)")

    print("\n=== Test complete ===")


if __name__ == "__main__":
    main()
