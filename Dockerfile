FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# System dependencies:
#   libsndfile1  — required by librosa / soundfile
#   gcc          — required by Triton (vLLM) to compile runtime kernels
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 gcc git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# By default, install without vLLM (uses HuggingFace Transformers backend).
# To include vLLM, build with:  docker build --build-arg INSTALL_VLLM=1 -t sarashina2.2-tts .
ARG INSTALL_VLLM=0
RUN if [ "$INSTALL_VLLM" = "1" ]; then \
        pip install --no-cache-dir -e ".[vllm]"; \
    else \
        pip install --no-cache-dir -e .; \
    fi

EXPOSE 7860

# Pass --use-vllm via the USE_VLLM environment variable:
#   docker run --gpus all -e USE_VLLM=1 -p 7860:7860 sarashina2.2-tts
ENTRYPOINT ["sh", "-c", "if [ \"$USE_VLLM\" = \"1\" ]; then exec python server/gradio_app.py --use-vllm; else exec python server/gradio_app.py; fi"]
