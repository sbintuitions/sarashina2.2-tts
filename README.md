# sarashina2.2-tts

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-sarashina2.2--tts-yellow)](https://huggingface.co/sbintuitions/sarashina2.2-tts) [![Demo](https://img.shields.io/badge/DemoPage-Listen-blue)](https://huggingface.co/sbintuitions/sarashina2.2-tts#audio-samples) [![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)](#)

**English** | [日本語](README_ja.md)

**sarashina2.2-tts** is a Japanese-centric text-to-speech system built on a large language model, developed by [SB Intuitions](https://www.sbintuitions.co.jp/). It supports Japanese and English, delivering high pronunciation accuracy, naturalness, and stability across diverse speaking styles, with zero-shot voice generation support.

> 🎧 Listen to our audio samples [here](https://huggingface.co/sbintuitions/sarashina2.2-tts#audio-samples).

## Highlights

- 🇯🇵 **Japanese-Centric**: Designed and optimized specifically for Japanese, with broad coverage of real-world use cases.
- 🎯 **High Accuracy**: Delivers strong pronunciation accuracy on Japanese text through large-scale end-to-end training.
- 🔒 **Responsibly Sourced Training Data**: Trained exclusively on legitimately acquired and properly licensed speech data.
- 🎙️ **Zero-shot Voice Generation**: Reproduces a speaker's voice, speaking style, and acoustic characteristics from a short reference clip.
- 🔊 **Natural & Expressive**: Produces highly natural speech with consistent quality, supporting a wide range of speaking styles including narration, broadcast, conversation, and customer service.
- 🌐 **Bilingual**: Supports both Japanese and English text-to-speech synthesis.

## Training Data

This model was trained on audio data collected from legitimately purchased audio sources, public speech archives, and data gathered in compliance with applicable domestic laws. During collection, we adhered to robots.txt directives and terms of service to ensure proper data acquisition.

## Quick Start

### Local Installation

1. Clone this repo
    ```bash
    git clone https://github.com/sbintuitions/sarashina2.2-tts.git
    cd sarashina2.2-tts
    ```

2. Install dependencies
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -e .
    ```

    If you want to use vLLM for faster inference:
    ```bash
    pip install -e ".[vllm]"
    ```

3. Start the Gradio web UI. Models will be automatically downloaded from HuggingFace on first run.

    ```bash
    python server/gradio_app.py
    ```

    To use vLLM backend:
    ```bash
    python server/gradio_app.py --use-vllm
    ```

    Open `http://localhost:7860` in your browser.

### Docker
By default, the Docker image uses the HuggingFace Transformers backend (no vLLM). This keeps the image small and works on GPUs with limited VRAM (~6 GB).

```bash
docker build -t sarashina2.2-tts .
docker run --gpus all -p 7860:7860 sarashina2.2-tts
```

Models are downloaded inside the container on first run. To avoid re-downloading every time, download the models to a local directory and mount it:
```bash
# Download models
huggingface-cli download sbintuitions/sarashina2.2-tts --local-dir /path/to/local/pretrained_models

# Run with mounted models
docker run --gpus all -p 7860:7860 \
  -v /path/to/local/pretrained_models:/app/pretrained_models \
  sarashina2.2-tts
```

To use vLLM for faster inference and higher throughput (requires more VRAM), build and run with:
```bash
# Build with vLLM support
docker build --build-arg INSTALL_VLLM=1 -t sarashina2.2-tts-vllm .

# Run with vLLM enabled
docker run --gpus all -e USE_VLLM=1 -p 7860:7860 sarashina2.2-tts-vllm
```

### Use the model in your code
```python
from sarashina_tts.generate.generate import SarashinaTTSGenerator

# Models are automatically downloaded on first run.
# An inaudible watermark is embedded by default. Please do not remove it.
generator = SarashinaTTSGenerator()

# Or use vLLM for faster inference
# generator = SarashinaTTSGenerator(use_vllm=True)

audio_prompt_path = "path/to/your/audio_prompt.wav"
audio_prompt_text = "ここに音声プロンプトに対応するテキストを入力してください。"

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
    "東京から金沢までは新幹線を利用するのが便利で、所要時間は約２時間半です。",
]

wavs = generator.generate(
    texts, 
    flow_embedding=flow_embedding, 
    audio_prompt_text=audio_prompt_text, 
    audio_prompt_tokens=audio_prompt_tokens, 
    audio_prompt_feat=audio_prompt_feat,
    audio_prompt_path=audio_prompt_path,
    flow_embedding_only=False,
)
generator.save_audios(wavs, output_dir="./output")
```
The generated audio will be saved in `./output` directory.

The generated audio contains an inaudible watermark powered by [SilentCipher](https://github.com/sony/silentcipher), which allows the audio to be identified as AI-generated. Please do not remove or disable the watermark when using this model.

## Acknowledgments
This model is built upon or incorporates code and models from the following open-source projects:
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [FlashCosyVoice](https://github.com/xingchensong/FlashCosyVoice)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [SilentCipher](https://github.com/sony/silentcipher)

## License
This model is licensed under [Sarashina Model NonCommercial License Agreement](./LICENSE).

If you are interested in using this model for commercial purposes, please feel free to contact us through our [contact page](https://www.sbintuitions.co.jp/contact/).
