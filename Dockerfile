# Qwen-Image-Edit-2511 (Q4_K_M GGUF) + Lightning LoRA + Real-ESRGAN x4
# RunPod serverless worker - runs on a 24 GB GPU (RTX 4090 / L4 / A5000).
#
# Build (from this folder; --platform matters on Apple Silicon Macs):
#   docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/qwen-edit-upscale:v2 .
#   docker push YOUR_DOCKERHUB_USER/qwen-edit-upscale:v2
#
# Baked in: Q4_K_M transformer (12.8 GB), Lightning LoRA (~0.8 GB), upscaler (64 MB).
# NOT baked in: the text encoder and VAE. Set the endpoint's "Model" field to
# Qwen/Qwen-Image-Edit-2511 so RunPod caches them on the host (fast cold starts,
# no download billing).

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Quantized transformer. Swap GGUF_FILE for Q5_K_M (15.4 GB) or Q6_K (17.9 GB) on a
# 32-48 GB GPU; Q4_K_M is the largest that leaves headroom on 24 GB.
ARG GGUF_REPO=unsloth/Qwen-Image-Edit-2511-GGUF
ARG GGUF_FILE=qwen-image-edit-2511-Q4_K_M.gguf
RUN mkdir -p /models/transformer \
    && python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('${GGUF_REPO}', '${GGUF_FILE}', local_dir='/models/transformer')" \
    && rm -rf /models/transformer/.cache
ENV GGUF_PATH=/models/transformer/${GGUF_FILE}

# transformer/config.json next to the GGUF: from_single_file needs the architecture
# config, and without this it is fetched from the Hub in the middle of every cold start.
ARG BASE_MODEL=Qwen/Qwen-Image-Edit-2511
RUN python -c "import shutil; from huggingface_hub import hf_hub_download; \
cfg = hf_hub_download('${BASE_MODEL}', 'transformer/config.json'); \
shutil.copy(cfg, '/models/transformer/config.json')" \
    && rm -rf /root/.cache/huggingface
ENV BASE_MODEL_BAKED=${BASE_MODEL}

# Optional: bake the text encoder, VAE and configs into the image so a cold worker
# downloads nothing. Adds ~17 GB to the image but removes a ~16 GB per-worker
# download when the endpoint has no cached model. The 40 GB transformer is NOT
# included: the GGUF above replaces it.
ARG BAKE_BASE=0
RUN if [ "$BAKE_BASE" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${BASE_MODEL}', local_dir='/models/base', \
allow_patterns=['model_index.json','*.json','*.txt','text_encoder/*','vae/*',\
'tokenizer/*','processor/*','scheduler/*'],\
ignore_patterns=['transformer/*'])" \
      && rm -rf /models/base/.cache; \
    fi
ENV LOCAL_BASE=/models/base
# Lightning 4-step LoRA (works at 4 or 8 steps) + Real-ESRGAN x4plus upscaler
RUN mkdir -p /models/lora /models/upscale \
    && python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('lightx2v/Qwen-Image-Edit-2511-Lightning', \
'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors', local_dir='/models/lora')" \
    && wget -q -O /models/upscale/RealESRGAN_x4plus.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    && rm -rf /models/lora/.cache

COPY handler.py .

CMD ["python", "handler.py"]
