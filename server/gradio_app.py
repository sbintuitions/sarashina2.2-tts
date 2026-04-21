import argparse
import os
import hashlib

import torch
import torchaudio
import soundfile as sf
import gradio as gr

from typing import List

from sarashina_tts.generate.generate import SarashinaTTSGenerator
from sarashina_tts.flow_matching.decoder import FlowDecoder
from sarashina_tts.text_frontend.text_splitter import (
    split_text,
    STRATEGY_NO_SPLIT,
    STRATEGY_AUTO,
    STRATEGY_SENTENCE,
    SPLIT_STRATEGIES,
)
from sarashina_tts.utils.audio_concat import concat_wavs


def _get_audio_duration(filepath: str) -> float:
    """Return the duration of an audio file in seconds."""
    info = sf.info(filepath)
    return info.duration


def synthesize(
    prompt_file, prompt_text, text_to_synth, split_strategy,
    max_length, repetition_penalty, do_sample, temperature, top_p,
):
    # Compute cache key from audio bytes and prompt text
    with open(prompt_file, "rb") as f:
        data = f.read()
    key = hashlib.md5(data + prompt_text.encode("utf-8")).hexdigest()

    if key not in cache:
        flow_emb = gen._extract_zero_shot_embedding(prompt_file)
        tokens = gen._extract_audio_prompt_tokens(prompt_file)
        feat = gen._extract_audio_prompt_feat(prompt_file)
        cache[key] = (flow_emb, tokens, feat)
    else:
        flow_emb, tokens, feat = cache[key]

    gen_kwargs = {
        "max_length": int(max_length),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    # Split text into segments based on strategy
    prompt_duration = _get_audio_duration(prompt_file)
    segments = split_text(
        text_to_synth,
        strategy=split_strategy,
        prompt_duration_s=prompt_duration,
    )

    # Generate each segment and concatenate
    all_wavs: List[torch.Tensor] = []
    for segment in segments:
        wavs = gen.generate(
            texts=[segment],
            flow_embedding=flow_emb,
            audio_prompt_path=prompt_file,
            audio_prompt_text=prompt_text,
            audio_prompt_tokens=tokens,
            audio_prompt_feat=feat,
            watermark=True,
            gen_kwargs=gen_kwargs,
        )
        all_wavs.extend(wavs)

    # Concatenate all segments, inserting silence gaps where needed
    final_wav = concat_wavs(all_wavs, sample_rate=FlowDecoder.sample_rate)

    output_dir = "gradio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    paths = gen.save_audios([final_wav], output_dir=output_dir)
    return paths[0]


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks UI (pure layout, no heavy init)."""
    demo = gr.Blocks()
    with demo:
        gr.Markdown("## Sarashina-TTS Demo")
        gr.Markdown(
            "This is a demo of Sarashina-TTS zero-shot text-to-speech synthesis. "
            "This model can generate speech in the voice style and timbre of a prompt audio.\n"
            "1. Upload or record a prompt audio file (around 5 seconds).\n"
            "2. Input the transcription of the prompt audio in the \"Prompt Text\" box.\n"
            "3. Enter your text to synthesize.\n"
            "4. Adjust the generation parameters as needed.\n"
            "5. Click \"Synthesize\" to generate audio."
        )

        # --- 1. Prompt selection ---
        gr.Markdown("### 1. Add Speech Prompt")
        with gr.Row():
            prompt_audio = gr.Audio(label="Prompt Audio", type="filepath")
            prompt_text = gr.Textbox(label="Prompt Text", placeholder="Enter the transcription of the prompt audio")

        # --- 2. Text input ---
        gr.Markdown("### 2. Input Text to Synthesize")
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text_to_synth = gr.Textbox(label="Text to Synthesize (single line)")
                    split_strategy = gr.Radio(
                        choices=[
                            (STRATEGY_AUTO, STRATEGY_AUTO),
                            (STRATEGY_SENTENCE, STRATEGY_SENTENCE),
                            (STRATEGY_NO_SPLIT, STRATEGY_NO_SPLIT),
                        ],
                        label="Text Splitting Strategy",
                        value=STRATEGY_AUTO,
                        info=(
                            "auto: split long text automatically to fit the ~30s generation limit. "
                            "sentence: always split at every sentence boundary. "
                            "no_split: send the entire text as-is (may truncate if too long)."
                        ),
                    )

        # --- 3. Generation parameters ---
        with gr.Row():
            max_length = gr.Slider(label="Max Length", minimum=1, maximum=4000, step=1, value=2000)
            repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
        with gr.Row():
            do_sample = gr.Checkbox(label="Do Sample", value=True)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.9)
            top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.01, value=0.95)

        # --- Synthesize ---
        synth_button = gr.Button("Synthesize")
        out_generated = gr.Audio(label="Generated Audio", type="filepath")

        synth_button.click(
            fn=synthesize,
            inputs=[
                prompt_audio, prompt_text, text_to_synth, split_strategy,
                max_length, repetition_penalty, do_sample, temperature, top_p,
            ],
            outputs=[out_generated],
        )


    return demo


# ---------------------------------------------------------------------------
# Entry point — protected by __main__ guard so that vLLM spawn workers
# importing this module don't re-trigger model initialization.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarashina‑TTS Gradio Demo")
    parser.add_argument("--model-dir", default=None, help="Path to model directory (default: pretrained_models)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend for faster inference")
    parser.add_argument("--config", default=None, help="Path to JSON config (advanced)")
    args = parser.parse_args()

    # Initialize generator (watermark=True to pre-load the watermark model)
    if args.config:
        gen = SarashinaTTSGenerator.from_config(args.config)
    else:
        gen = SarashinaTTSGenerator(
            model_dir=args.model_dir,
            use_vllm=args.use_vllm,
            watermark=True,
        )

    cache = {}

    # Build & launch
    demo = build_ui()

    server_name = os.getenv("GRADIO_SERVER_HOST", "0.0.0.0")
    server_port_env = os.getenv("GRADIO_SERVER_PORT", 7860)
    try:
        server_port = int(server_port_env)
    except (TypeError, ValueError):
        server_port = 7860
    root_path = os.getenv("GRADIO_ROOT_PATH", "")
    demo.launch(server_name=server_name, server_port=server_port, root_path=root_path, share=False)
