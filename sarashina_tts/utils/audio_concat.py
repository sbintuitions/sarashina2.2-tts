"""audio_concat.py — Concatenate waveform segments with natural silence gaps.

When long text is split into multiple segments for sequential TTS generation,
the resulting waveforms need to be concatenated.  A naive ``torch.cat`` may
produce jarring transitions if adjacent segments lack sufficient silence at
their boundaries.

This module measures the existing silence at the tail of the preceding segment
and the head of the following segment, and inserts just enough zero-padding to
reach a target gap duration.

Typical usage
=============
>>> from sarashina_tts.utils.audio_concat import concat_wavs
>>> combined = concat_wavs(wav_list, sample_rate=24000)
"""

from __future__ import annotations

import logging
from typing import List

import torch

__all__ = [
    "concat_wavs",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target silence duration between segments (seconds).
_TARGET_GAP_S = 0.2

# RMS threshold below which a frame is considered "silent".
# This is relative to the full-scale range of float32 audio (±1.0).
_SILENCE_RMS_THRESHOLD = 0.01

# Window size used to scan for silence at segment boundaries (seconds).
# We look at up to this much audio at the tail / head of each segment.
_SCAN_WINDOW_S = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms(signal: torch.Tensor) -> float:
    """Compute the root-mean-square of a 1-D signal."""
    if signal.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(signal.float() ** 2)))


def _measure_tail_silence(wav: torch.Tensor, sample_rate: int) -> float:
    """Measure how many seconds of silence exist at the end of *wav*.

    Scans backwards from the end in small frames, stopping when the RMS
    of a frame exceeds ``_SILENCE_RMS_THRESHOLD``.

    Parameters
    ----------
    wav
        1-D or 2-D (1, T) waveform tensor.
    sample_rate
        Sample rate in Hz.

    Returns
    -------
    float
        Duration of trailing silence in seconds (capped at
        ``_SCAN_WINDOW_S``).
    """
    wav = wav.squeeze()
    scan_samples = min(wav.numel(), int(_SCAN_WINDOW_S * sample_rate))
    if scan_samples == 0:
        return 0.0

    tail = wav[-scan_samples:]
    frame_size = max(1, int(0.01 * sample_rate))  # 10 ms frames

    silent_samples = 0
    for start in range(len(tail) - frame_size, -1, -frame_size):
        frame = tail[start:start + frame_size]
        if _rms(frame) > _SILENCE_RMS_THRESHOLD:
            break
        silent_samples += frame_size

    return min(silent_samples / sample_rate, _SCAN_WINDOW_S)


def _measure_head_silence(wav: torch.Tensor, sample_rate: int) -> float:
    """Measure how many seconds of silence exist at the start of *wav*.

    Parameters
    ----------
    wav
        1-D or 2-D (1, T) waveform tensor.
    sample_rate
        Sample rate in Hz.

    Returns
    -------
    float
        Duration of leading silence in seconds (capped at
        ``_SCAN_WINDOW_S``).
    """
    wav = wav.squeeze()
    scan_samples = min(wav.numel(), int(_SCAN_WINDOW_S * sample_rate))
    if scan_samples == 0:
        return 0.0

    head = wav[:scan_samples]
    frame_size = max(1, int(0.01 * sample_rate))  # 10 ms frames

    silent_samples = 0
    for start in range(0, len(head) - frame_size + 1, frame_size):
        frame = head[start:start + frame_size]
        if _rms(frame) > _SILENCE_RMS_THRESHOLD:
            break
        silent_samples += frame_size

    return min(silent_samples / sample_rate, _SCAN_WINDOW_S)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def concat_wavs(
    wavs: List[torch.Tensor],
    sample_rate: int,
    target_gap_s: float = _TARGET_GAP_S,
) -> torch.Tensor:
    """Concatenate waveform segments, inserting silence gaps as needed.

    For each pair of adjacent segments, the existing trailing silence of the
    preceding segment and leading silence of the following segment are
    measured.  If their sum is less than *target_gap_s*, a zero-padding is
    inserted to make up the difference.

    Parameters
    ----------
    wavs
        List of waveform tensors.  Each may be 1-D ``(T,)`` or
        2-D ``(1, T)``.
    sample_rate
        Sample rate in Hz (e.g. 24000).
    target_gap_s
        Desired silence duration between segments in seconds.

    Returns
    -------
    torch.Tensor
        Concatenated waveform with the same shape convention as the input
        tensors (preserves 2-D if all inputs are 2-D).
    """
    if not wavs:
        return torch.zeros(1, 0)

    if len(wavs) == 1:
        return wavs[0]

    # Determine whether inputs are 2-D (1, T) — preserve shape on output
    is_2d = all(w.dim() == 2 for w in wavs)

    parts: List[torch.Tensor] = []
    for i, wav in enumerate(wavs):
        squeezed = wav.squeeze()
        if squeezed.numel() == 0:
            continue

        parts.append(squeezed)

        # Insert gap before the next segment (if any)
        if i < len(wavs) - 1:
            next_wav = wavs[i + 1].squeeze()
            if next_wav.numel() == 0:
                continue

            tail_s = _measure_tail_silence(squeezed, sample_rate)
            head_s = _measure_head_silence(next_wav, sample_rate)
            existing_s = tail_s + head_s

            if existing_s < target_gap_s:
                pad_s = target_gap_s - existing_s
                pad_samples = int(pad_s * sample_rate)
                if pad_samples > 0:
                    logger.debug(
                        "Segment %d→%d: tail=%.3fs, head=%.3fs, "
                        "inserting %.3fs (%d samples) of silence.",
                        i, i + 1, tail_s, head_s, pad_s, pad_samples,
                    )
                    parts.append(torch.zeros(pad_samples, device=squeezed.device))
            else:
                logger.debug(
                    "Segment %d→%d: tail=%.3fs + head=%.3fs = %.3fs ≥ target %.3fs, "
                    "no padding needed.",
                    i, i + 1, tail_s, head_s, existing_s, target_gap_s,
                )

    result = torch.cat(parts, dim=0)

    if is_2d:
        result = result.unsqueeze(0)

    return result
