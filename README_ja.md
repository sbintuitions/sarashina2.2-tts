# sarashina2.2-tts

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-sarashina2.2--tts-yellow)](https://huggingface.co/sbintuitions/sarashina2.2-tts) [![Demo](https://img.shields.io/badge/DemoPage-Listen-blue)](https://huggingface.co/sbintuitions/sarashina2.2-tts#audio-samples) [![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)](#)

[English](README.md) | **日本語**

**sarashina2.2-tts** は、[SB Intuitions](https://www.sbintuitions.co.jp/) が開発した、大規模言語モデルをベースにした日本語中心の音声合成（TTS）システムです。日本語と英語に対応し、高い発音精度・自然性・安定性を備え、多様な話し方スタイルに対応したゼロショット音声クローニングをサポートしています。

> 🎧 デモ音声は[こちら](https://huggingface.co/sbintuitions/sarashina2.2-tts#audio-samples)から試聴できます。

## 特長

- 🇯🇵 **日本語中心**: 日本語に特化して設計・最適化されており、幅広い実用シーンに対応します。
- 🎯 **高い発音精度**: 大規模な end-to-end 学習により、日本語テキストに対して高い発音正確性を実現します。
- 🔒 **適正なデータで学習**: 正規購入音源や公的な音声アーカイブ、現行の国内法規に従って収集された音声データを用いて学習しています。
- 🎙️ **ゼロショット音声再現**: 短い参照音声から、話者の声質、話し方、音響的特徴を再現できます。
- 🔊 **自然で多彩な表現**: 高い自然性と安定した品質で音声を生成し、ナレーション、放送、日常会話、接客など多様なスタイルに対応します。
- 🌐 **バイリンガル**: 日本語と英語の両方の音声合成に対応します。

## 学習データ

本モデルは、正規購入音源や公的な音声アーカイブ、現行の国内法規に従って収集された音声データを用いて学習されています。収集に際しては、robots.txtや利用規約を遵守し、適正な情報取得を行っています。

## クイックスタート

### ローカルインストール

1. リポジトリをクローン
    ```bash
    git clone https://github.com/sbintuitions/sarashina2.2-tts.git
    cd sarashina2.2-tts
    ```

2. 依存関係をインストール
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -e .
    ```

    vLLMを使用してより高速な推論を行う場合:
    ```bash
    pip install -e ".[vllm]"
    ```

3. Gradio Web UIを起動します。初回実行時にHuggingFaceからモデルが自動的にダウンロードされます。

    ```bash
    python server/gradio_app.py
    ```

    vLLMバックエンドを使用する場合:
    ```bash
    python server/gradio_app.py --use-vllm
    ```

    ブラウザで `http://localhost:7860` を開いてください。

### Docker
デフォルトでは、DockerイメージはHuggingFace Transformersバックエンド（vLLMなし）を使用します。これにより、イメージサイズが小さく保たれ、限られたVRAM（約6 GB）のGPUでも動作します。

```bash
docker build -t sarashina2.2-tts .
docker run --gpus all -p 7860:7860 sarashina2.2-tts
```

モデルは初回実行時にコンテナ内にダウンロードされます。毎回の再ダウンロードを避けるには、モデルをローカルディレクトリにダウンロードしてマウントしてください:
```bash
# モデルのダウンロード
huggingface-cli download sbintuitions/sarashina2.2-tts --local-dir /path/to/local/pretrained_models

# マウントして実行
docker run --gpus all -p 7860:7860 \
  -v /path/to/local/pretrained_models:/app/pretrained_models \
  sarashina2.2-tts
```

vLLMを使用してより高速な推論と高いスループットを得る場合（より多くのVRAMが必要）:
```bash
# vLLMサポート付きでビルド
docker build --build-arg INSTALL_VLLM=1 -t sarashina2.2-tts-vllm .

# vLLMを有効にして実行
docker run --gpus all -e USE_VLLM=1 -p 7860:7860 sarashina2.2-tts-vllm
```

### コードでモデルを使用する
```python
from sarashina_tts.generate.generate import SarashinaTTSGenerator

# 初回実行時にモデルが自動的にダウンロードされます。
# デフォルトで不可聴の透かしが埋め込まれます。この透かしは除去しないでください。
generator = SarashinaTTSGenerator()

# vLLMを使用してより高速な推論を行う場合
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
生成された音声は `./output` ディレクトリに保存されます。

生成された音声には、[SilentCipher](https://github.com/sony/silentcipher) を利用した不可聴の透かし（ウォーターマーク）がデフォルトで埋め込まれており、AI生成音声であることを識別できます。本モデルを使用する際は、透かしを除去または無効化しないでください。

## 謝辞
本モデルの開発にあたり、以下のオープンソースプロジェクトのコードやモデルを参考・利用しています。
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [FlashCosyVoice](https://github.com/xingchensong/FlashCosyVoice)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [SilentCipher](https://github.com/sony/silentcipher)

## ライセンス
このモデルは[Sarashina Model NonCommercial License Agreement](./LICENSE)に基づいて公開されています。

もしこのモデルの商用利用にご興味がある場合は、お気軽に[コンタクトページ](https://www.sbintuitions.co.jp/contact/)へご連絡ください。
