# Qwen-Image-Edit-2511 + Lightning LoRA + Real-ESRGAN upscaling, on ComfyUI.
# Base image, ComfyUI install and model set follow wlsdml1114/qwen_image_edit.
FROM wlsdml1114/multitalk-base:1.7 as runtime

# wget: model downloads at build time, URL inputs at request time
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt

# ── Qwen-Image-Edit-2511 ────────────────────────────────────────────────────────
RUN wget -q https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors -O /ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
RUN wget -q https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors -O /ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
RUN wget -q https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors -O /ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
RUN wget -q https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -O /ComfyUI/models/vae/qwen_image_vae.safetensors

# ── Upscalers ───────────────────────────────────────────────────────────────────
# RealESRGAN_x4plus is the default ("upscale_model" in a request can name either).
# Both are x4 models and ~64 MB each, so keeping both costs almost nothing.
RUN mkdir -p /ComfyUI/models/upscale_models && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth && \
    wget -q https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth -O /ComfyUI/models/upscale_models/4x-UltraSharp.pth

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
