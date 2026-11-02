# Qwen-Image-Edit-2511 + Lightning LoRA + ESRGAN upscaling, on ComfyUI.
# Base image, ComfyUI install and model set follow wlsdml1114/qwen_image_edit.
#
# The models are NOT baked in. They total ~31 GB, and a RunPod hub build cannot
# export and upload a layer that big inside its 30-minute limit (it dies in
# "exporting layers" / "sending tarball"). entrypoint.sh fetches them on the first
# cold worker into MODEL_DIR instead - put that on a network volume and the download
# is paid once per volume. For a local build, bake them in with:
#
#   docker build --build-arg BAKE_MODELS=1 -t <you>/qwen-edit-upscale .
FROM wlsdml1114/multitalk-base:1.7 AS runtime

# wget: model downloads, both at startup and for URL inputs at request time
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

RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/text_encoders \
             /ComfyUI/models/vae /ComfyUI/models/loras /ComfyUI/models/upscale_models

# Optional: bake the model set into the image (local builds only - too slow for a
# RunPod hub build). Off by default; entrypoint.sh downloads them at startup.
COPY download_models.sh /download_models.sh
RUN chmod +x /download_models.sh
ARG BAKE_MODELS=0
RUN if [ "$BAKE_MODELS" = "1" ]; then /download_models.sh /ComfyUI/models; fi

COPY . .
RUN chmod +x /entrypoint.sh /download_models.sh

CMD ["/entrypoint.sh"]
