"""
Reasoning tag detection and stripping.

Two formats supported (hardcoded):
  1. OpenAI/DeepSeek/Qwen HTML-style: <think>...</think>  <reasoning>...</reasoning>
  2. Gemma channel-style:              <|channel>thought ... <channel|>
"""

import re

# Format definitions: (format_name, [tag_names], closing_tag)
# closing_tag=None means channel-style (no closing tag).
_FORMATS = [
    # OpenAI o-series / DeepSeek / Qwen thinking style: <think>...</think>
    ("html", ["think", "reasoning"], "/"),
    # Gemma channel style: <|channel>thought ... <channel|>
    ("channel", ["thought"], None),
]

# Channel-style patterns — precompiled for performance
# Opening tag: <|channel>thought
_CHANNEL_OPEN = re.compile(r"<\s*\|channel\s*>thought")
# Closing tag: <channel|>
_CHANNEL_CLOSE = re.compile(r"<\s*channel\s*\|>")
# Full channel block: opening + content (non-greedy) + closing
_CHANNEL_PATTERN = re.compile(
    r"<\s*\|channel\s*>thought([\s\S]*?)<\s*channel\s*\|>"
)
# Channel prefix for streaming: opening tag + content (no closing yet)
_CHANNEL_PREFIX = re.compile(
    r"<\s*\|channel\s*>thought([\s\S]*)$"
)


def build_strip_patterns() -> tuple[list[re.Pattern], list[re.Pattern]]:
    """
    Build regex patterns for stripping reasoning tags from text.

    Returns:
        (full_patterns, prefix_patterns)
        full_patterns — match complete tags + content + closing tag
        prefix_patterns — match unclosed tag prefix (streaming/truncation)
    """
    full: list[re.Pattern] = []
    prefix: list[re.Pattern] = []

    for fmt_name, tags, closing in _FORMATS:
        for tag in tags:
            if closing is None:
                # Channel-style: <|channel>thought ... <channel|>
                full.append(_CHANNEL_PATTERN)
                prefix.append(_CHANNEL_PREFIX)
            else:
                # HTML-style: <think>content</think>
                esc_tag = re.escape(tag)
                full.append(
                    re.compile(
                        rf"<{esc_tag}>\s*([\s\S]*?)\s*<\s*{closing}\s*{esc_tag}\s*>",
                        re.DOTALL,
                    )
                )
                # Prefix: unclosed tag at end of text (streaming/truncation)
                prefix.append(
                    re.compile(
                        rf"<{esc_tag}>\s*([\s\S]*)$",
                        re.DOTALL,
                    )
                )

    return full, prefix


def strip_reasoning_tags(text: str) -> str:
    """
    Remove reasoning tag blocks from text.

    Handles both complete blocks and partial (unclosed) blocks from
    streaming/truncation.
    """
    full_patterns, prefix_patterns = build_strip_patterns()

    cleaned = text
    # Strip complete tags + content + closing tag
    for pat in full_patterns:
        cleaned = pat.sub("", cleaned)

    # Strip unclosed tag prefix (truncated streaming)
    for pat in prefix_patterns:
        cleaned = pat.sub("", cleaned).strip()

    return cleaned
