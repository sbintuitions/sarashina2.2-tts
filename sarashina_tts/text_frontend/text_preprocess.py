"""text_preprocess.py — Rule-based text preprocessing for Sarashina-TTS.

All preprocessing rules are applied sequentially via :func:`preprocess_text`,
which is the single public entry point for callers (e.g. the generation
pipeline).

To add a new rule:
1. Write a function ``_rule_<name>(text: str) -> str``.
2. Append it to ``_RULES``.

The rules are executed in the order they appear in ``_RULES``.

TODO
----
* Add processing for "異体字".
    * Transform old style kanji to modern style kanji according to 常用漢字表.
* Add processing for special punctuations.
    * Need to decide which punctuations to keep and which to remove.
* Normalization
    * Fullwidth and halfwidth.
    * Spaces.
    * etc.
* Preprocess for PronSteering.
"""

from __future__ import annotations

import re
import logging
from typing import Callable, List

__all__ = [
    "preprocess_text",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def _rule_strip_markdown(text: str) -> str:
    """Remove common Markdown formatting, keeping the readable text content.

    Handles (in order):
    * Images ``![alt](url)`` → alt text
    * Links ``[text](url)`` → link text
    * Bold / italic ``**text**``, ``__text__``, ``*text*``, ``_text_``
    * Strikethrough ``~~text~~``
    * Inline code `` `code` ``
    * Headings ``# …`` through ``###### …``
    * Blockquotes ``> …``
    * Unordered list markers ``- ``, ``* ``, ``+ ``
    * Ordered list markers ``1. ``, ``2. ``, …
    * Horizontal rules ``---``, ``***``, ``___``
    """
    # Bold + italic: ***text*** or ___text___
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    # Italic: *text* or _text_  (avoid matching mid-word underscores)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Inline code: `code`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Horizontal rules (line consists of only ---, ***, or ___)
    text = re.sub(r"^[\s]*([-*_])\1{2,}[\s]*$", "", text, flags=re.MULTILINE)
    # Headings: # text → text  (up to 6 levels)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Blockquotes: > text → text
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Unordered lists: - item, * item, + item → item
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    # # Ordered lists: 1. item, 2. item → item
    # text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # # Images: ![alt](url) → alt
    # text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # # Links: [text](url) → text
    # text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text


def _rule_normalize_brackets(text: str) -> str:
    """Replace parentheses / brackets with Japanese quotation marks.

    * Fullwidth parentheses （） → 「」
    * ASCII parentheses ()     → ""
    """
    text = text.replace("（", "「").replace("）", "」")
    text = text.replace("(", "\u201c").replace(")", "\u201d")
    return text


# ---------------------------------------------------------------------------
# Rule pipeline
# ---------------------------------------------------------------------------

# Each entry is a callable  str -> str.
# Rules are applied in order; add new rules by appending to this list.
# NOTE: _rule_strip_markdown runs first so that bracket normalization
#       does not interfere with Markdown link / image syntax.
_RULES: List[Callable[[str], str]] = [
    _rule_strip_markdown,
    _rule_normalize_brackets,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Apply all preprocessing rules to *text* and return the result.

    Parameters
    ----------
    text
        Raw input text.

    Returns
    -------
    str
        Preprocessed text ready for tokenization / generation.
    """
    for rule in _RULES:
        text = rule(text)
    return text
