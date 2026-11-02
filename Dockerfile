# Qwen-Image-Edit-2511 + Lightning LoRA + Real-ESRGAN x4 - RunPod serverless worker
#
# Build (from this folder; --platform matters on Apple Silicon Macs):
#   docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/qwen-edit-upscale:v1 .
#   docker push YOUR_DOCKERHUB_USER/qwen-edit-upscale:v1
#
# The 58 GB base model is NOT baked in: set the endpoint's "Model" field to
# Qwen/Qwen-Image-Edit-2511 so RunPod caches it on the host (fast cold starts,
# no download billing). The small files (LoRA ~0.8 GB, upscaler ~64 MB) are baked in.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

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
