# FlowDecoder: unified wrapper for Flow Matching + HiFi-GAN inference.

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from .flow import CausalMaskedDiffWithXvec
from .hifigan import HiFTGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HiFT configuration constants
# ---------------------------------------------------------------------------
# Mel hop size: 1 mel frame = _HIFT_HOP_LENGTH audio samples.
# Derived from HiFTGenerator: upsample_rates=[8,5,3] × istft_hop_len=4 = 480.
_HIFT_HOP_LENGTH = 480

# ---------------------------------------------------------------------------
# HiFT mel-length bucketing (cuDNN plan cache warm-up)
# ---------------------------------------------------------------------------
# cuDNN creates an execution plan for each unique conv input shape.
# Plan creation costs ~90ms (CPU-side) while the actual GPU compute is ~7ms.
# By padding mel to a fixed set of bucket lengths, we limit the shape space
# and can pre-warm all plans at startup.
#
# 30s audio @ 24kHz with hop_size=480 → 1500 mel frames.
# Bucket interval = 64 → 24 buckets → ~2s warmup.
_HIFT_BUCKET_INTERVAL = 64
_HIFT_MAX_MEL_T = 1536  # ceil(1500 / 64) * 64 = 1536 → ~30.72s
_HIFT_BUCKETS = list(range(_HIFT_BUCKET_INTERVAL, _HIFT_MAX_MEL_T + 1, _HIFT_BUCKET_INTERVAL))

# ---------------------------------------------------------------------------
# Timing switch — flip to True to enable cuda.synchronize() around
# flow / hift and print wall-clock breakdown.  Keep False for production.
# ---------------------------------------------------------------------------
_DEBUG_TIMING = False


def _pad_mel_to_bucket(mel: torch.Tensor) -> torch.Tensor:
    """Pad mel spectrogram (1, 80, T) to the nearest bucket length."""
    T = mel.shape[-1]
    target = ((T + _HIFT_BUCKET_INTERVAL - 1) // _HIFT_BUCKET_INTERVAL) * _HIFT_BUCKET_INTERVAL
    if target > T:
        mel = F.pad(mel, (0, target - T), value=0.0)
    return mel


# ---------------------------------------------------------------------------
# Mel spectrogram extraction
# ---------------------------------------------------------------------------
_mel_basis_cache = {}
_hann_window_cache = {}


def extract_mel_spectrogram(
    y: torch.Tensor,
    *,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000,
    center: bool = False,
) -> torch.Tensor:
    """Compute log-mel spectrogram matching CosyVoice2 frontend.

    Parameters
    ----------
    y : (B, wave_len) or (wave_len,) raw waveform at 24 kHz

    Returns
    -------
    (B, num_mels, T) log-mel spectrogram
    """
    global _mel_basis_cache, _hann_window_cache

    if y.dim() == 1:
        y = y.unsqueeze(0)

    key = f"{fmax}_{y.device}"
    if key not in _mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)

    mel_basis = _mel_basis_cache[key]
    hann_window = _hann_window_cache[str(y.device)]

    y = F.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y, n_fft,
            hop_length=hop_size, win_length=win_size,
            window=hann_window, center=center,
            pad_mode="reflect", normalized=False,
            onesided=True, return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec


class FlowDecoder:
    """Drop-in replacement for CosyVoice2, providing token → waveform decoding.

    Usage
    -----
    ```python
    decoder = FlowDecoder("/path/to/CosyVoice2-0.5B", fp16=True)

    wav = decoder.token2wav(
        token=tokens,
        prompt_token=ptok,
        prompt_feat=pfeat,
        embedding=emb,
    )
    ```
    """

    # sample_rate exposed for backward compat (e.g. save_audios)
    sample_rate: int = 24000

    def __init__(
        self,
        model_dir: str,
        *,
        fp16: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_dir : str
            Directory containing flow.pt and hift.pt.
        fp16 : bool
            Whether to run Flow model in FP16.
        device : str or torch.device, optional
            Target device.
        """
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp16 = fp16

        # ---- Flow (token → mel) ------------------------------------
        self.flow = CausalMaskedDiffWithXvec()
        logger.info("Using Flow model")
        print("Using Flow model")
        flow_state = torch.load(
            f"{model_dir}/flow.pt", map_location="cpu", weights_only=True
        )
        self.flow.load_state_dict(flow_state, strict=True)
        if self.fp16:
            self.flow.half()
        self.flow.to(self.device).eval()
        logger.info("Flow model loaded from %s/flow.pt", model_dir)

        # ---- HiFi-GAN / HiFT (mel → waveform) ---------------------
        self.hift = HiFTGenerator()
        hift_state = {
            k.replace("generator.", ""): v
            for k, v in torch.load(
                f"{model_dir}/hift.pt", map_location="cpu", weights_only=True
            ).items()
        }
        self.hift.load_state_dict(hift_state, strict=True)
        # Remove weight_norm to eliminate per-call CPU dispatch overhead.
        # In eager mode, weight_norm recomputes weights on every forward
        # call via Python dispatcher — this costs ~80ms CPU per HiFT call
        # while the actual GPU compute is only ~5ms.
        self.hift.remove_weight_norm()
        self.hift.to(self.device).eval()
        logger.info("HiFT model loaded from %s/hift.pt (weight_norm removed)", model_dir)

        # ---- Warm up HiFT for all bucket shapes ----
        # This pre-creates cuDNN execution plans so that inference never
        # hits the ~90ms cold-start penalty per unseen shape.
        # Only meaningful on CUDA; skip on CPU to avoid wasted startup time.
        if self.device.type == "cuda":
            self._warmup_hift()

        # Expose for compat with code that accesses codec_decoder.model.*
        self.model = self

    def _warmup_hift(self) -> None:
        """Run dummy HiFT forwards for all bucket shapes to pre-create cuDNN plans."""
        import time as _time
        logger.info(
            "Warming up HiFT for %d bucket shapes (interval=%d, max=%d) ...",
            len(_HIFT_BUCKETS), _HIFT_BUCKET_INTERVAL, _HIFT_MAX_MEL_T,
        )
        _t0 = _time.time()
        with torch.no_grad():
            for bucket_T in _HIFT_BUCKETS:
                dummy_mel = torch.zeros(1, 80, bucket_T, device=self.device)
                self.hift(speech_feat=dummy_mel)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        _elapsed = _time.time() - _t0
        logger.info("HiFT warmup done in %.2fs (%d buckets)", _elapsed, len(_HIFT_BUCKETS))
        print(f"HiFT warmup done in {_elapsed:.2f}s ({len(_HIFT_BUCKETS)} buckets)")

    def token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """Convert speech tokens → waveform for a single utterance.

        Parameters
        ----------
        token : (T,) tensor — generated speech token ids
        prompt_token : (T_prompt,) tensor — prompt token ids
        prompt_feat : (T_prompt_mel, 80) tensor — prompt mel feats
        embedding : (192,) tensor — speaker embedding
        speed : float — playback speed

        Returns
        -------
        (1, wave_len) waveform tensor (on CPU)
        """
        # --- Prepare inputs (single utterance, batch dim = 1) ---
        combined = torch.cat([prompt_token.flatten(), token.flatten()])
        flow_input = combined.unsqueeze(0)  # (1, T)
        flow_input_len = torch.tensor([combined.shape[0]], dtype=torch.int32)

        prompt_mel_len = prompt_feat.shape[0]
        prompt_feat_2d = prompt_feat.unsqueeze(0)  # (1, T_prompt_mel, 80)
        prompt_feat_len_t = torch.tensor([prompt_mel_len], dtype=torch.int32)

        emb = embedding.flatten()[:192].unsqueeze(0)  # (1, 192)

        # --- Flow: tokens → mel ---
        if _DEBUG_TIMING:
            import time as _time
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_flow_start = _time.time()

        with torch.amp.autocast(self.device.type, enabled=self.fp16):
            mel_out, mel_len = self.flow(
                flow_input.to(self.device),
                flow_input_len.to(self.device),
                prompt_feat_2d.to(self.device),
                prompt_feat_len_t.to(self.device),
                emb.to(self.device),
                streaming=False,
                finalize=True,
            )

        if _DEBUG_TIMING:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_flow_end = _time.time()

        # --- HiFi-GAN: mel → wav (bucket-padded) ---
        ml = int(mel_len[0].item())
        mel = mel_out[0, :, prompt_mel_len:ml].unsqueeze(0)  # (1, 80, T_gen)

        if speed != 1.0:
            mel = F.interpolate(
                mel, size=int(mel.shape[2] / speed), mode="linear"
            )

        orig_T = mel.shape[-1]
        mel_padded = _pad_mel_to_bucket(mel)

        wav, _ = self.hift(speech_feat=mel_padded)
        wav = wav[:, :orig_T * _HIFT_HOP_LENGTH]  # truncate padding

        result = wav.cpu()

        if _DEBUG_TIMING:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_hift_end = _time.time()
            print(
                f"token2wav breakdown: flow={_t_flow_end - _t_flow_start:.4f}s | "
                f"hift={_t_hift_end - _t_flow_end:.4f}s | "
                f"total={_t_hift_end - _t_flow_start:.4f}s"
            )

        return result
