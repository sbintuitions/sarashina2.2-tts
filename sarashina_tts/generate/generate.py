from __future__ import annotations

"""generate.py — Sarashina-TTS inference pipeline.

Supports both HuggingFace and vLLM backends for LLM text generation.
Set ``use_vllm=True`` to use the vLLM backend for faster inference.

Typical usage
=============

1. Simplest (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~
```python
from sarashina_tts.generate.generate import SarashinaTTSGenerator

# Models are automatically downloaded from HuggingFace on first run
gen = SarashinaTTSGenerator()

# Extract prompt-related features from a reference audio
prompt_path = "examples/prompts/VOICEACTRESS100_001.wav"
prompt_text = "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"

flow_embedding = gen._extract_zero_shot_embedding(prompt_path)
audio_prompt_tokens = gen._extract_audio_prompt_tokens(prompt_path)
audio_prompt_feat = gen._extract_audio_prompt_feat(prompt_path)

# Generate speech
wavs = gen.generate(
    texts=["今日の天気いいですね！"],
    flow_embedding=flow_embedding,
    audio_prompt_text=prompt_text,
    audio_prompt_tokens=audio_prompt_tokens,
    audio_prompt_feat=audio_prompt_feat,
    audio_prompt_path=prompt_path,
)

gen.save_audios(wavs, output_dir="outputs")
```

2. With vLLM (faster inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
gen = SarashinaTTSGenerator(use_vllm=True)
```

3. From JSON config (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```python
gen = SarashinaTTSGenerator.from_config("path/to/config.json")
```
"""

import os
import sys

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

from typing import List, Optional, Union, Dict, Any
import json
import logging
import traceback
import time

import s3tokenizer
import soundfile as sf
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

# Sarashina‑TTS specific
from sarashina_tts.additional_tokens import SPEECH_START_TOKEN, SEMANTIC_TOKENS
from sarashina_tts.speech_encoder.speech_encoder import SpeechEncoder
from sarashina_tts.text_frontend.text_preprocess import preprocess_text

# Flow decoder & helpers
from sarashina_tts.utils.codec_utils import (
    audio_tokens_to_tensor_list_batch,
    audio_decode_flash,
)
from sarashina_tts.flow_matching.decoder import FlowDecoder
from sarashina_tts.utils.watermark import AudioWatermarker

__all__ = [
    "SarashinaTTSGenerator",
]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _semantic_ids_to_str(token_ids: List[int]) -> str:
    """Convert semantic token IDs → concatenated string for the LLM prompt."""
    return "".join(str(SEMANTIC_TOKENS[t]) for t in token_ids)


class BaseTextGenerator:
    """Abstract base class for text generation backends."""
    
    def generate(self, prompts: List[str], gen_kwargs: Dict[str, Any]) -> List[str]:
        """Generate text from prompts."""
        raise NotImplementedError


class HuggingFaceGenerator(BaseTextGenerator):
    """HuggingFace transformers backend for text generation."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompts: List[str], gen_kwargs: Dict[str, Any]) -> List[str]:
        """Generate text using HuggingFace transformers."""
        logger.info("Using HuggingFace transformers for generation")
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            add_special_tokens=True,
            padding_side="left"
        ).to(self.device)
        
        # Convert vLLM-style parameters to HF parameters
        hf_kwargs = {
            'max_length': inputs.input_ids.shape[1] + gen_kwargs.get('max_tokens', 2048),
            'do_sample': True,
            'temperature': gen_kwargs.get('temperature', 0.9),
            'top_p': gen_kwargs.get('top_p', 0.95),
            'repetition_penalty': 1.0,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **hf_kwargs,
            )
        
        # Extract only the newly generated tokens
        new_tokens = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        logger.debug(f"Generated tokens: {new_tokens}")
        return new_tokens


class VLLMGenerator(BaseTextGenerator):
    """vLLM backend for text generation."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._vllm_model = None
        self._initialize_vllm()
    
    def _initialize_vllm(self):
        """Initialize vLLM model with proper error handling."""
        try:
            from vllm import LLM, SamplingParams
            
            logger.info("Initializing vLLM model...")
            logger.info("Note: If you encounter CUDA multiprocessing errors, "
                       "set VLLM_WORKER_MULTIPROC_METHOD=spawn before running the script")
            
            # Use gpu_memory_utilization OR kv_cache_memory_bytes. 
            # Inferencing will need 3~4 GB (or more, need check) to load the models and ~100 MB for the KV cache for each generation thread. 
            # If your GPU is only for deploying this TTS model and hope for better speed and throughput, use gpu_memory_utilization to let 
            # vLLM utilize most of the GPU memory. 
            config = {
                "model": self.model_path,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "max_model_len": 2048,
                # "gpu_memory_utilization": 0.8, 
                "kv_cache_memory_bytes": 128 * 1024 * 1024, # 128MB for KV cache
            }
            
            logger.info(f"vLLM config: {json.dumps(config, indent=2, default=str)}")
            self._vllm_model = LLM(**config)
            logger.info("✓ vLLM initialization successful!")
            
        except Exception as e:
            logger.error(f"vLLM initialization failed: {str(e)}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize vLLM: {str(e)}")
    
    def generate(self, prompts: List[str], gen_kwargs: Dict[str, Any]) -> List[str]:
        """Generate text using vLLM."""
        from vllm import SamplingParams
        
        logger.info("Using vLLM for generation")
        
        if self._vllm_model is None:
            raise RuntimeError("vLLM model not initialized")
        
        # Convert to vLLM parameters
        vllm_kwargs = {
            'max_tokens': gen_kwargs.get('max_tokens', 2048),
            'temperature': gen_kwargs.get('temperature', 0.9),
            'top_p': gen_kwargs.get('top_p', 0.95),
            'top_k': gen_kwargs.get('top_k', 50),
            'repetition_penalty': gen_kwargs.get('repetition_penalty', 1.0),
        }
        
        sampling_params = SamplingParams(**vllm_kwargs)
        outputs = self._vllm_model.generate(prompts, sampling_params)
        
        # Extract generated text
        new_tokens = []
        for output in outputs:
            sequence = output.outputs[0]  # we only generate one sequence (n=1)
            new_tokens.append(sequence.text)
        
        logger.debug(f"Generated tokens: {new_tokens}")
        return new_tokens


class SarashinaTTSGenerator:
    """High‑level wrapper for Sarashina‑TTS × FlowDecoder."""

    # Default HuggingFace model repository
    DEFAULT_MODEL_ID = "sbintuitions/Sarashina-TTS"
    DEFAULT_MODEL_DIR = "pretrained_models"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        *,
        use_vllm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        decoder_fp16: bool = True,
        watermark: bool = True,
    ) -> None:
        """Initialize SarashinaTTSGenerator.

        Parameters
        ----------
        model_dir : str, optional
            Local directory containing all model files.
            Defaults to ``pretrained_models``.
            If the directory does not contain the expected files, they will
            be automatically downloaded from HuggingFace.
        model_id : str, optional
            HuggingFace repository ID for automatic download.
            Defaults to ``sbintuitions/Sarashina-TTS``.
        use_vllm : bool
            Whether to use vLLM for faster inference.
        device : Optional[Union[str, torch.device]]
            Device to use for computation. Auto-detected if not specified.
        decoder_fp16 : bool
            Whether to use FP16 for the flow decoder.
        watermark : bool
            Whether to embed an audio watermark into generated speech using
            `SilentCipher <https://github.com/sony/silentcipher>`_.
            Defaults to ``False``.
        """
        logger.info("Initializing SarashinaTTSGenerator...")

        model_dir = model_dir or self.DEFAULT_MODEL_DIR
        model_id = model_id or self.DEFAULT_MODEL_ID

        # Auto-download models from HuggingFace if needed
        self._ensure_models(model_dir, model_id)

        # Derive all paths from model_dir
        base_model_path = model_dir
        codec_decoder_path = model_dir
        speech_encoder_ckpt = os.path.join(model_dir, "campplus_cn_common.bin")

        self.base_model_path = base_model_path
        self.use_vllm = use_vllm

        # Initialize device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|pad|>"
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        # Set padding side to left for VLLM compatibility
        self.tokenizer.padding_side = "left"

        # Initialize text generation backend
        self._init_text_generator(use_vllm)

        # Initialize codec decoder (FlowDecoder)
        logger.info("Loading FlowDecoder...")
        self.codec_decoder = FlowDecoder(
            codec_decoder_path,
            fp16=decoder_fp16,
            device=self.device,
        )

        # Initialize speech encoder
        logger.info("Loading speech encoder...")
        self.speech_encoder = SpeechEncoder(speech_encoder_ckpt, device=self.device)

        # Initialize S3 tokenizer
        logger.info("Loading S3 tokenizer...")
        self.s3_tokenzier = s3tokenizer.load_model("speech_tokenizer_v2_25hz").to(self.device)

        # Initialize watermark
        self.watermark_enabled = watermark
        self._watermarker: Optional[AudioWatermarker] = None
        if watermark:
            logger.info("Loading AudioWatermarker...")
            device_str = str(self.device) if not isinstance(self.device, str) else self.device
            self._watermarker = AudioWatermarker(device=device_str)
            if self._watermarker.is_available:
                logger.info("✓ AudioWatermarker initialized successfully")
            else:
                logger.warning("AudioWatermarker is not available, watermarking will be disabled")
                self.watermark_enabled = False

        logger.info("✓ SarashinaTTSGenerator initialization complete!")

    # Files that must be present for a complete model installation
    REQUIRED_MODEL_FILES = [
        "config.json",              # LLM config
        "flow.pt",                  # Flow decoder
        "hift.pt",                  # HiFT vocoder
        "campplus_cn_common.bin",   # Speech encoder
    ]

    @classmethod
    def _ensure_models(cls, model_dir: str, model_id: str) -> None:
        """Download models from HuggingFace if not already present.

        Checks all required files. If any are missing, calls
        ``snapshot_download`` which will skip already-downloaded files
        and only fetch the missing ones.
        """
        missing = [
            f for f in cls.REQUIRED_MODEL_FILES
            if not os.path.isfile(os.path.join(model_dir, f))
        ]

        if not missing:
            logger.info("All model files found in %s, skipping download.", model_dir)
            return

        logger.info(
            "Missing model files in %s: %s. Downloading from %s ...",
            model_dir, missing, model_id,
        )
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                model_id,
                local_dir=model_dir,
                ignore_patterns=["samples/*"],
            )
            logger.info("✓ Models downloaded to %s", model_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download models from {model_id}: {e}\n"
                f"You can manually download the models and place them in '{model_dir}'.\n"
                f"  huggingface-cli download {model_id} --local-dir {model_dir}"
            ) from e
    
    def _init_text_generator(self, use_vllm: bool):
        """Initialize text generation backend with fallback."""
        self.text_generator = None
        
        if use_vllm:
            try:
                logger.info("Attempting to initialize vLLM backend...")
                self.text_generator = VLLMGenerator(self.base_model_path)
                logger.info("✓ vLLM backend initialized successfully")
                return
            except Exception as e:
                logger.warning(f"vLLM initialization failed: {str(e)}")
                logger.info("Falling back to HuggingFace backend...")
        
        # Fallback to HuggingFace
        logger.info("Loading HuggingFace model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
        ).eval()
        model = model.to(self.device)
        self.text_generator = HuggingFaceGenerator(model, self.tokenizer, self.device)
        logger.info("✓ HuggingFace backend initialized successfully")

    def generate(
        self,
        texts: List[str],
        *,
        flow_embedding: Optional[torch.Tensor],
        audio_prompt_text: Optional[str] = "",
        audio_prompt_tokens: Optional[Union[str, List[int]]] = None,
        audio_prompt_feat: Optional[torch.Tensor] = None,
        audio_prompt_path: Optional[str] = None,
        flow_embedding_only: Optional[bool] = False,
        watermark: Optional[bool] = None,
        gen_kwargs: Optional[Dict] = None,
        decode_kwargs: Optional[Dict] = None,
    ) -> List[torch.Tensor]:
        """Return list of waveform tensors (no saving).

        Parameters
        ----------
        texts
            Utterances. Each should include ``<|speech_start|>`` if required.
        flow_embedding
            1×192 tensor for CosyVoice2 flow‑matching.
        audio_prompt_text / audio_prompt_tokens
            Extra conditioning snippets to prepend / append.
        flow_embedding_only
            If ``True``, only flow embedding is used for generation.
            In this case, flow model will not use audio prompt tokens and features, 
            so the generated audio will only catch the timbre and prosody, 
            not catching the acoustic details of the prompt audio.
        watermark
            Whether to embed an audio watermark. If ``None`` (default),
            uses the instance-level ``watermark_enabled`` setting.
        gen_kwargs / decode_kwargs
            Override defaults for LLM generate & codec decode.
        """
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        start_time = time.time()
        gen_kwargs = gen_kwargs or {}
        decode_kwargs = decode_kwargs or {}

        # Consistency check for zero‑shot prompt parts
        has_prompt_text = bool(audio_prompt_text)
        has_prompt_tokens = audio_prompt_tokens is not None
        has_prompt_path = audio_prompt_path is not None
        if not (has_prompt_text == has_prompt_tokens == has_prompt_path):
            raise ValueError(
                "`audio_prompt_text`, `audio_prompt_tokens`, and `audio_prompt_path` "
                "must be provided together for zero‑shot prompting."
            )

        # Preprocess input texts
        texts = [preprocess_text(t) for t in texts]

        # Log input texts
        for i, t in enumerate(texts):
            logger.info("Generating text [%d/%d]: %s", i + 1, len(texts), t)

        # Build LLM prompts
        if has_prompt_text and has_prompt_tokens:
            token_suffix = (
                audio_prompt_tokens
                if isinstance(audio_prompt_tokens, str)
                else _semantic_ids_to_str(audio_prompt_tokens)
            )
            prompts = [f"{audio_prompt_text}{t}{SPEECH_START_TOKEN}{token_suffix}" for t in texts]
        else:
            prompts = [f"{t}{SPEECH_START_TOKEN}" for t in texts]  # plain mode

        # Generation parameters
        default_gen = {
            'max_tokens': 2048,
            'temperature': 0.9,
            'top_p': 0.95,
            'presence_penalty': 0.0,
            'frequency_penalty': 1.0,
        }
        default_gen.update(gen_kwargs)
        
        # Generate text tokens
        logger.debug("Starting text generation...")
        new_tokens = self.text_generator.generate(prompts, default_gen)
        
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        llm_end_time = time.time()
        logger.info(f"LLM generation took: {llm_end_time - start_time:.3f}s")
        
        # Convert text tokens to audio tokens
        audio_token_tensors = audio_tokens_to_tensor_list_batch(new_tokens)

        # Decode audio
        if flow_embedding is not None:
            flow_embedding = flow_embedding.to(self.device)
        if audio_prompt_feat is not None:
            audio_prompt_feat = audio_prompt_feat.to(self.device)

        logger.debug("Starting audio decoding...")
        audios: List[torch.Tensor] = []
        for tok_list in audio_token_tensors:
            if len(tok_list) == 0:
                audios.append(torch.zeros(1, 0))
                continue
            wav = audio_decode_flash(
                tok_list,
                self.codec_decoder,
                flow_embedding_only=flow_embedding_only,
                code_layer=1,
                num_latency_tokens=1,
                speed=decode_kwargs.get("speed", 1.0),
                flow_embedding=flow_embedding,
                prompt_tokens=audio_prompt_tokens if not isinstance(audio_prompt_tokens, str) else None,
                prompt_feat=audio_prompt_feat,
            )
            audios.append(wav)
        
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        decode_end_time = time.time()
        logger.info(f"Total audio decoding (Flow + HiFT) took: {decode_end_time - llm_end_time:.3f}s")

        # Apply watermark if enabled
        do_watermark = watermark if watermark is not None else self.watermark_enabled
        if do_watermark:
            if self._watermarker is None:
                logger.info("Lazy-loading AudioWatermarker...")
                device_str = str(self.device) if not isinstance(self.device, str) else self.device
                self._watermarker = AudioWatermarker(device=device_str)
            if self._watermarker.is_available:
                watermark_start_time = time.time()
                audios = self._watermarker.apply_to_list(audios, FlowDecoder.sample_rate)
                watermark_end_time = time.time()
                logger.info(f"Watermarking took: {watermark_end_time - watermark_start_time:.3f}s")

        total_end_time = time.time()
        logger.info(f"Total generation time: {total_end_time - start_time:.3f}s")
        logger.info(f"RTF: {((total_end_time - start_time) / max(1e-9, sum((wav.numel() / float(FlowDecoder.sample_rate)) for wav in audios if isinstance(wav, torch.Tensor) and wav.numel() > 0))):.3f}")
        
        return audios

    def generate_zero_shot_from_file(
        self,
        *,
        texts: List[str],
        flow_embedding: Optional[torch.Tensor],
        audio_prompt_path: str,
        audio_prompt_text: str = "",
        audio_prompt_tokens: Optional[Union[str, List[int]]] = None,
        gen_kwargs: Optional[Dict] = None,
        decode_kwargs: Optional[Dict] = None,
    ) -> List[torch.Tensor]:
        """High‑level convenience wrapper for zero‑shot voice cloning."""
        # Extract flow embedding if not provided
        if flow_embedding is None:
            flow_embedding = self._extract_zero_shot_embedding(audio_prompt_path)

        # Extract audio prompt tokens if not provided
        if audio_prompt_tokens is None:
            audio_prompt_tokens = self._extract_audio_prompt_tokens(audio_prompt_path)

        return self.generate(
            texts=texts,
            flow_embedding=flow_embedding,
            audio_prompt_text=audio_prompt_text,
            audio_prompt_tokens=audio_prompt_tokens,
            gen_kwargs=gen_kwargs,
            decode_kwargs=decode_kwargs,
        )
    
    def save_audios(
        self,
        audios: List[torch.Tensor],
        output_dir: str,
        *,
        prefix: str = "output_",
    ) -> List[str]:
        """Persist waveform tensors with incremental names; returns paths."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, wav in enumerate(audios):
            path = os.path.join(output_dir, f"{prefix}{i}.wav")
            # torchaudio.save(path, wav, FlowDecoder.sample_rate)
            wav_np = wav.detach().cpu().numpy().T
            sf.write(path, wav_np, FlowDecoder.sample_rate)
            paths.append(path)
        return paths

    @classmethod
    def from_config(
        cls,
        cfg_path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "SarashinaTTSGenerator":
        """Instantiate SarashinaTTSGenerator from a JSON config (advanced).

        Parameters
        ----------
        cfg_path
            Path to a JSON file. Supported keys:
            ``model_dir``, ``model_id``, ``use_vllm``, ``device``, ``decoder_fp16``.

            For backward compatibility, ``base_model_path`` / ``codec_decoder_path``
            / ``speech_encoder_ckpt`` are also accepted and take precedence over
            ``model_dir``.
        device
            Override CUDA / CPU selection if needed.
        """
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Backward-compatible: if old-style keys exist, use model_dir from base_model_path
        if "base_model_path" in cfg:
            model_dir = cfg["base_model_path"]
        else:
            model_dir = cfg.get("model_dir")

        gen = cls(
            model_dir=model_dir,
            model_id=cfg.get("model_id"),
            use_vllm=cfg.get("use_vllm", False),
            device=device or cfg.get("device"),
            decoder_fp16=cfg.get("decoder_fp16", True),
            watermark=cfg.get("watermark", False),
        )

        return gen

    # Private utilities
    @staticmethod
    def _load_audio_sf(file: str, sr: int = 16000) -> torch.Tensor:
        """Load audio via soundfile and resample to *sr* Hz.

        Returns a 1-D float32 :class:`torch.Tensor`, same contract as
        ``s3tokenizer.load_audio`` but without the torchaudio dependency.
        """
        data, sample_rate = sf.read(file, dtype="float32")          # (N,) or (N, C)
        audio = torch.from_numpy(data)
        if audio.dim() == 2:
            audio = audio[:, 0]                                     # mono: first channel
        if sample_rate != sr:
            audio = torchaudio.transforms.Resample(sample_rate, sr)(audio)
        return audio

    def _extract_audio_prompt_tokens(self, audio_prompt_path: str) -> List[int]:
        """Extract semantic tokens from audio prompt."""
        mels = []
        wav_paths = [audio_prompt_path]
        for wav_path in wav_paths:
            audio = self._load_audio_sf(wav_path)
            mels.append(s3tokenizer.log_mel_spectrogram(audio))
        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = self.s3_tokenzier.quantize(mels.to(self.device), mels_lens.to(self.device))
        return codes[0].tolist()
        
    def _extract_zero_shot_embedding(self, audio_prompt_path: str) -> torch.Tensor:
        """Extract zero-shot embedding from audio prompt."""
        return self.speech_encoder.encode_from_path(audio_prompt_path)

    def _extract_audio_prompt_feat(self, audio_prompt_path: str) -> torch.Tensor:
        """Extract mel features from audio prompt for flow conditioning.
        
        Returns (1, T, 80) mel spectrogram tensor.
        Uses the same mel extraction as CosyVoice2:
        n_fft=1920, hop_size=480, win_size=1920, fmin=0, fmax=8000.
        """
        from sarashina_tts.flow_matching.decoder import extract_mel_spectrogram
        # speech, sr = torchaudio.load(audio_prompt_path)
        speech_np, sr = sf.read(audio_prompt_path, always_2d=True)
        speech_np = speech_np[:, 0:1]  # force mono (keep 2D shape)
        speech = torch.from_numpy(speech_np.T).float()
        prompt_speech_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(speech)
        # (1, 80, T) log-mel
        mel = extract_mel_spectrogram(prompt_speech_24k.squeeze(0))
        # Transpose to (1, T, 80)
        mel = mel.transpose(1, 2)  # (1, T, 80)
        return mel


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    import argparse
    parser = argparse.ArgumentParser(description="Sarashina‑TTS generation")
    parser.add_argument("--model-dir", default=None, help="Path to model directory")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend")
    parser.add_argument("--watermark", action="store_true", help="Embed an inaudible watermark into generated audio")
    parser.add_argument("--config", default=None, help="Path to JSON config (advanced)")
    args = parser.parse_args()

    if args.config:
        generator = SarashinaTTSGenerator.from_config(args.config)
    else:
        generator = SarashinaTTSGenerator(
            model_dir=args.model_dir,
            use_vllm=args.use_vllm,
            watermark=args.watermark,
        )

    audio_prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "prompts", "example.wav"
    )
    audio_prompt_text = "ここにpromptのテキストを入れてください。"

    # Cache prompt related variables to avoid recomputing them
    audio_prompt_tokens = generator._extract_audio_prompt_tokens(
        audio_prompt_path=audio_prompt_path
    )
    flow_embedding = generator._extract_zero_shot_embedding(
        audio_prompt_path=audio_prompt_path
    )
    audio_prompt_feat = generator._extract_audio_prompt_feat(
        audio_prompt_path=audio_prompt_path
    )

    texts = [
        "今日の天気いいですね！",
        "東京から金沢までは新幹線を利用するのが便利で、所要時間は約２時間半です。",
    ]

    for text in texts:
        wavs = generator.generate(
            [text],
            flow_embedding=flow_embedding,
            audio_prompt_text=audio_prompt_text,
            audio_prompt_tokens=audio_prompt_tokens,
            audio_prompt_feat=audio_prompt_feat,
            audio_prompt_path=audio_prompt_path,
            flow_embedding_only=False,
        )
    generator.save_audios(wavs, output_dir="./output")
