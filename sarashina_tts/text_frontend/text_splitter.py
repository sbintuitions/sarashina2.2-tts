"""text_splitter.py — Split long text into chunks for Sarashina-TTS inference.

The model is trained on a maximum of ~30 seconds of audio (prompt + generated).
When the input text is too long to fit within the remaining time budget after
the prompt audio, it must be split into smaller segments that are generated
independently and concatenated.

This module provides three splitting strategies selectable by the user:

* ``no_split``  — pass the text through unchanged (single segment).
* ``auto``      — automatically split only when the text exceeds the
                  estimated time budget, using natural sentence boundaries.
* ``sentence``  — always split at every sentence boundary.

Typical usage
=============
>>> from sarashina_tts.text_frontend.text_splitter import split_text
>>> segments = split_text(
...     "こんにちは。今日はいい天気ですね。散歩に行きましょう。",
...     strategy="auto",
...     prompt_duration_s=5.0,
... )
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

__all__ = [
    "split_text",
    "STRATEGY_NO_SPLIT",
    "STRATEGY_AUTO",
    "STRATEGY_SENTENCE",
    "SPLIT_STRATEGIES",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strategy names (used in Gradio dropdown / API parameter)
STRATEGY_NO_SPLIT = "no_split"
STRATEGY_AUTO = "auto"
STRATEGY_SENTENCE = "sentence"

SPLIT_STRATEGIES = [STRATEGY_NO_SPLIT, STRATEGY_AUTO, STRATEGY_SENTENCE]

# Maximum total audio duration the model can handle (seconds).
_MAX_TOTAL_DURATION_S = 30.0

# Safety margin subtracted from the available duration to avoid edge-case
# truncation (seconds).
_SAFETY_MARGIN_S = 2.0

# Conservative characters-per-second estimate.
# Japanese: ~5-7 chars/s, English: ~12-15 chars/s.
# We use a conservative (low) value so that we split sooner rather than risk
# truncation.  This can be refined with empirical data later.
_CHARS_PER_SECOND = 5.0

# ---------------------------------------------------------------------------
# Splitting regex (by priority)
# ---------------------------------------------------------------------------

# Priority 0 (highest): newlines — explicit user-intended breaks
_NEWLINE_RE = re.compile(r"\n+")

# Priority 1: sentence-ending punctuation (JP & EN)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？.!?])")

# Priority 2: clause-level punctuation
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[、，,;；:：])")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _estimate_max_chars(prompt_duration_s: float) -> int:
    """Estimate the maximum number of characters that fit the time budget.

    Parameters
    ----------
    prompt_duration_s
        Duration of the prompt audio in seconds.

    Returns
    -------
    int
        Conservative upper bound on character count.
    """
    available_s = _MAX_TOTAL_DURATION_S - _SAFETY_MARGIN_S - prompt_duration_s
    if available_s <= 0:
        logger.warning(
            "Prompt audio (%.1fs) leaves no room for generation. "
            "Returning minimum character budget of 1.",
            prompt_duration_s,
        )
        return 1
    return max(1, int(available_s * _CHARS_PER_SECOND))


def _split_at_boundaries(text: str, pattern: re.Pattern) -> List[str]:
    """Split *text* at positions matched by *pattern*, keeping delimiters.

    Returned segments are stripped of leading/trailing whitespace;
    empty segments are discarded.
    """
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_newlines(text: str) -> List[str]:
    """Split text at newline boundaries, discarding empty lines."""
    parts = _NEWLINE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at sentence-ending punctuation."""
    return _split_at_boundaries(text, _SENTENCE_BOUNDARY_RE)


def _split_clauses(text: str) -> List[str]:
    """Split text into clauses at clause-level punctuation."""
    return _split_at_boundaries(text, _CLAUSE_BOUNDARY_RE)


def _greedy_merge(segments: List[str], max_chars: int) -> List[str]:
    """Greedily merge consecutive *segments* as long as the result ≤ *max_chars*.

    Parameters
    ----------
    segments
        Ordered list of text segments (e.g. sentences).
    max_chars
        Maximum character count per merged chunk.

    Returns
    -------
    List[str]
        Merged chunks, each ≤ *max_chars* (unless a single input segment
        already exceeds the limit — in that case it is kept as-is and will
        be further split by the caller).
    """
    if not segments:
        return []

    merged: List[str] = []
    current = segments[0]

    for seg in segments[1:]:
        candidate = current + seg
        if len(candidate) <= max_chars:
            current = candidate
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    return merged


def _split_oversized(segment: str, max_chars: int) -> List[str]:
    """Split a single oversized segment into pieces ≤ *max_chars*.

    Tries clause boundaries first; falls back to hard character-count split.
    """
    if len(segment) <= max_chars:
        return [segment]

    # Try clause-level splitting first
    clauses = _split_clauses(segment)
    if len(clauses) > 1:
        merged = _greedy_merge(clauses, max_chars)
        # Recursively handle any still-oversized pieces
        result: List[str] = []
        for m in merged:
            result.extend(_split_oversized(m, max_chars))
        return result

    # Hard split as last resort — break at max_chars boundaries
    pieces: List[str] = []
    while len(segment) > max_chars:
        pieces.append(segment[:max_chars])
        segment = segment[max_chars:]
    if segment:
        pieces.append(segment)
    return pieces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_text(
    text: str,
    strategy: str = STRATEGY_AUTO,
    prompt_duration_s: float = 5.0,
) -> List[str]:
    """Split *text* into segments suitable for sequential TTS generation.

    Parameters
    ----------
    text
        The full input text to synthesize.
    strategy
        One of ``"no_split"``, ``"auto"``, or ``"sentence"``.
    prompt_duration_s
        Duration of the prompt audio in seconds.  Used by ``"auto"`` to
        estimate how many characters can be generated in the remaining
        time budget.

    Returns
    -------
    List[str]
        Ordered list of text segments.  For ``"no_split"`` this is always
        a single-element list.
    """
    text = text.strip()
    if not text:
        return []

    if strategy not in SPLIT_STRATEGIES:
        logger.warning(
            "Unknown split strategy %r, falling back to %r.",
            strategy, STRATEGY_AUTO,
        )
        strategy = STRATEGY_AUTO

    # ------------------------------------------------------------------
    # Priority 0: always split at newlines first (user-intended breaks).
    # Then apply the chosen strategy to each paragraph.
    # ------------------------------------------------------------------
    paragraphs = _split_by_newlines(text)
    if not paragraphs:
        return []

    # ------------------------------------------------------------------
    # no_split — split at newlines only, no further splitting
    # ------------------------------------------------------------------
    if strategy == STRATEGY_NO_SPLIT:
        return paragraphs

    # ------------------------------------------------------------------
    # sentence — split each paragraph at sentence boundaries
    # ------------------------------------------------------------------
    if strategy == STRATEGY_SENTENCE:
        result: List[str] = []
        for para in paragraphs:
            sentences = _split_sentences(para)
            result.extend(sentences if sentences else [para])
        return result

    # ------------------------------------------------------------------
    # auto — split only paragraphs that exceed the estimated budget
    # ------------------------------------------------------------------
    max_chars = _estimate_max_chars(prompt_duration_s)
    logger.info(
        "Auto split: prompt=%.1fs, budget=%d chars (%.1f chars/s × %.1fs)",
        prompt_duration_s,
        max_chars,
        _CHARS_PER_SECOND,
        _MAX_TOTAL_DURATION_S - _SAFETY_MARGIN_S - prompt_duration_s,
    )

    result: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            result.append(para)
            continue

        # Step 1: split at sentence boundaries
        sentences = _split_sentences(para)
        if not sentences:
            sentences = [para]

        # Step 2: greedily merge short sentences
        merged = _greedy_merge(sentences, max_chars)

        # Step 3: handle any segments still over the limit
        for seg in merged:
            result.extend(_split_oversized(seg, max_chars))

    logger.info(
        "Auto split: %d paragraph(s) → %d segment(s).",
        len(paragraphs), len(result),
    )
    return result
