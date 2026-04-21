"""Audio watermarking utilities using SilentCipher.

This module provides a thin wrapper around
`SilentCipher <https://github.com/sony/silentcipher>`_ for embedding
imperceptible watermarks into generated audio waveforms.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default watermark payload
# ---------------------------------------------------------------------------
# 5 × 8-bit values (0–255) = 40-bit payload.
# Encodes "SRSNA" in ASCII as a Sarashina-TTS identifier.
WATERMARK_DEFAULT_MESSAGE: List[int] = [83, 82, 83, 78, 65]


class AudioWatermarker:
    """Manages a SilentCipher model and applies watermarks to audio tensors.

    Parameters
    ----------
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    model_type : str
        SilentCipher model variant.  ``"44.1k"`` (default) or ``"16k"``.
    """

    def __init__(self, device: str = "cuda", model_type: str = "44.1k") -> None:
        self._model = None
        try:
            import silentcipher

            logger.info("Loading SilentCipher watermark model (%s)...", model_type)
            self._model = silentcipher.get_model(
                model_type=model_type,
                device=device,
            )
            logger.info("✓ SilentCipher watermark model loaded successfully")
        except ImportError:
            logger.warning(
                "silentcipher is not installed. Watermarking will be unavailable. "
                "Install it with: pip install silentcipher"
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize SilentCipher watermark model: %s. "
                "Watermarking will be unavailable.",
                e,
            )

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        """Return ``True`` if the watermark model was loaded successfully."""
        return self._model is not None

    # ------------------------------------------------------------------
    def apply(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        message: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Embed a SilentCipher watermark into a waveform tensor.

        Parameters
        ----------
        wav : torch.Tensor
            Waveform tensor, shape ``(1, T)`` or ``(T,)``.
        sample_rate : int
            Sample rate of *wav*.
        message : list[int], optional
            5 × 8-bit values (each 0–255).  Defaults to
            :data:`WATERMARK_DEFAULT_MESSAGE`.

        Returns
        -------
        torch.Tensor
            Watermarked waveform at the original *sample_rate*, same shape
            as input.  If the model is not available the input is returned
            unchanged.
        """
        if self._model is None:
            return wav

        message = message or WATERMARK_DEFAULT_MESSAGE

        # Squeeze to 1-D and move to the same device as the watermark model
        # so that SilentCipher's internal CUDA tensors don't clash.
        wav_1d = wav.detach().squeeze()
        if wav_1d.dim() == 0:
            return wav

        model_device = self._model.device
        wav_input = wav_1d.to(model_device).float()

        # encode_wav accepts a torch.Tensor and returns a torch.Tensor.
        # calc_sdr=False avoids a numpy/torch incompatibility inside SilentCipher.
        encoded, _sdr = self._model.encode_wav(
            wav_input, sample_rate, message, calc_sdr=False,
        )

        # Move back to CPU and restore the original shape
        encoded_tensor = encoded.detach().cpu().float()
        if wav.dim() == 2:
            encoded_tensor = encoded_tensor.unsqueeze(0)

        return encoded_tensor

    # ------------------------------------------------------------------
    def apply_to_list(
        self,
        audios: List[torch.Tensor],
        sample_rate: int,
        message: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Apply watermark to a list of waveform tensors.

        Convenience wrapper around :meth:`apply` that skips empty tensors.
        """
        if not self.is_available:
            return audios

        logger.info("Applying SilentCipher watermark...")
        result: List[torch.Tensor] = []
        for wav in audios:
            if isinstance(wav, torch.Tensor) and wav.numel() > 0:
                wav = self.apply(wav, sample_rate, message)
            result.append(wav)
        logger.info("✓ Watermark applied")
        return result
