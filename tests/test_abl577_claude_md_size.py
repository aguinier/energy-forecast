"""ABL-577: CLAUDE.md size guard for energy-forecast.

CLAUDE.md auto-loads into every agent context.  Its size is a per-turn token
tax on every run in this repo -- in the sibling dashboard repo the same file
reached ~95k tokens and killed four consecutive runs (ABL-536).  The Board
trimmed it to 390 lines / 26 KB on 2026-08-27 and set the hard budget at
700 lines / 40 KB.  This test enforces that budget so it cannot creep back by
accretion.

When this test fails:
  * If you added rules: merge them with existing rules in place (correct in
    place, do not append).
  * If you added narrative: move it to docs/claude/ and link from CLAUDE.md.
  * Budget: 700 lines, 40,960 bytes.  Both limits must hold.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
CLAUDE_MD = REPO / "CLAUDE.md"

LINE_LIMIT = 700
BYTE_LIMIT = 40_960  # 40 KB


def _read():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    size = len(CLAUDE_MD.read_bytes())
    return lines, size


def test_claude_md_line_count():
    lines, _ = _read()
    count = len(lines)
    assert count <= LINE_LIMIT, (
        f"CLAUDE.md is {count} lines -- hard budget is {LINE_LIMIT}.\n"
        f"Move narrative to docs/claude/; correct rules in place (do not append)."
    )


def test_claude_md_byte_size():
    _, size = _read()
    assert size <= BYTE_LIMIT, (
        f"CLAUDE.md is {size:,} bytes -- hard budget is {BYTE_LIMIT:,} ({BYTE_LIMIT // 1024} KB).\n"
        f"Move narrative to docs/claude/; correct rules in place (do not append)."
    )


# ---------------------------------------------------------------------------
# Negative control: the assertions must actually fire on oversized input.
# ---------------------------------------------------------------------------

def test_line_limit_assertion_fires_on_oversized_input():
    """Prove the line-count guard is not vacuous."""
    too_many_lines = ["x"] * (LINE_LIMIT + 1)
    count = len(too_many_lines)
    with pytest.raises(AssertionError, match="hard budget"):
        assert count <= LINE_LIMIT, (
            f"CLAUDE.md is {count} lines -- hard budget is {LINE_LIMIT}.\n"
            f"Move narrative to docs/claude/; correct rules in place (do not append)."
        )


def test_byte_limit_assertion_fires_on_oversized_input():
    """Prove the byte-size guard is not vacuous."""
    oversized = BYTE_LIMIT + 1
    with pytest.raises(AssertionError, match="hard budget"):
        assert oversized <= BYTE_LIMIT, (
            f"CLAUDE.md is {oversized:,} bytes -- hard budget is {BYTE_LIMIT:,} ({BYTE_LIMIT // 1024} KB).\n"
            f"Move narrative to docs/claude/; correct rules in place (do not append)."
        )
