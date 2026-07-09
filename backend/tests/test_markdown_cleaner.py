"""Unit tests for markdown_cleaner.py."""
import pytest
from app.services.ingestion import clean_markdown


def test_strips_page_numbers():
    """Lines that are just page numbers should be removed."""
    text = "Introduction\nPage 3\nSome content\n5 of 20\nMore content\n- 12 -\n7\nEnd"
    result = clean_markdown(text)
    assert "Page 3" not in result
    assert "5 of 20" not in result
    assert "- 12 -" not in result
    # bare "7" on its own line removed
    lines = result.splitlines()
    assert "7" not in [l.strip() for l in lines]
    # real content preserved
    assert "Introduction" in result
    assert "Some content" in result
    assert "More content" in result
    assert "End" in result


def test_strips_repeated_headers():
    """A short line appearing 3+ times should be deduplicated to one occurrence."""
    header = "ACME Corp Confidential"
    text = "\n".join([
        "Chapter 1",
        header,
        "Some body text here.",
        header,
        "More body text.",
        header,
        header,
        "Final line.",
    ])
    result = clean_markdown(text)
    assert result.count(header) == 1
    assert "Chapter 1" in result
    assert "Final line." in result


def test_strips_broken_table_separators():
    """Lines that are pure pipe/dash separators with no real content are removed."""
    text = "| Name | Age |\n|---|---|\n| Alice | 30 |\n| --- | --- |\n| Bob | 25 |"
    result = clean_markdown(text)
    assert "|---|---|" not in result
    assert "| --- | --- |" not in result
    # Real table rows preserved
    assert "| Name | Age |" in result
    assert "| Alice | 30 |" in result
    assert "| Bob | 25 |" in result


def test_strips_ocr_garbage():
    """Lines that are mostly non-alphanumeric characters should be removed."""
    text = "Normal paragraph.\n@#$%^&*()!@#$%\nAnother normal line."
    result = clean_markdown(text)
    assert "@#$%^&*()!@#$%" not in result
    assert "Normal paragraph." in result
    assert "Another normal line." in result


def test_collapses_excessive_blank_lines():
    """More than 2 consecutive blank lines should be collapsed to at most 2."""
    text = "First paragraph.\n\n\n\n\nSecond paragraph."
    result = clean_markdown(text)
    # At most 2 consecutive blank lines means at most 2 empty lines between content,
    # i.e. no run of 3+ empty lines ("\n\n\n\n" in joined output)
    assert "\n\n\n\n" not in result
    assert "First paragraph." in result
    assert "Second paragraph." in result


def test_preserves_real_content():
    """Normal prose and markdown structure must pass through unchanged."""
    text = (
        "# Title\n\n"
        "This is a normal paragraph with **bold** and *italic* text.\n\n"
        "- Bullet one\n"
        "- Bullet two\n\n"
        "```python\nprint('hello')\n```\n"
    )
    result = clean_markdown(text)
    assert "# Title" in result
    assert "**bold**" in result
    assert "- Bullet one" in result
    assert "print('hello')" in result


def test_empty_string():
    """Empty input returns empty string without error."""
    assert clean_markdown("") == ""
