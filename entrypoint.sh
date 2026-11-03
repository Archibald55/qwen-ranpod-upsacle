#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# CUDA 검사 및 설정
echo "Checking CUDA availability..."

# Python을 통한 CUDA 검사
python_cuda_check() {
    python3 -c "
import torch
try:
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE')
        exit(0)
    else:
        print('CUDA_NOT_AVAILABLE')
        exit(1)
except Exception as e:
    print(f'CUDA_ERROR: {e}')
    exit(2)
" 2>/dev/null
}

# CUDA 검사 실행
cuda_status=$(python_cuda_check)
case $? in
    0)
        echo "✅ CUDA is available and working (Python check)"
        export CUDA_VISIBLE_DEVICES=0
        export FORCE_CUDA=1
        ;;
    1)
        echo "❌ CUDA is not available (Python check)"
        echo "Error: CUDA is required but not available. Exiting..."
        exit 1
        ;;
    2)
        echo "❌ CUDA check failed (Python check)"
        echo "Error: CUDA initialization failed. Exiting..."
        exit 1
        ;;
esac

# 추가적인 nvidia-smi 검사
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA driver working (nvidia-smi check)"
    else
        echo "❌ NVIDIA driver found but not working"
        echo "Error: NVIDIA driver is not working properly. Exiting..."
        exit 1
    fi
else
    echo "❌ NVIDIA driver not found"
    echo "Error: NVIDIA driver is required but not found. Exiting..."
    exit 1
fi

# CUDA 환경 변수 설정
echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"

# ── Models ──────────────────────────────────────────────────────────────────────
# The ~31 GB model set is not baked into the image: a RunPod hub build cannot export
# and upload that much inside its 30-minute limit. Instead it is fetched on the first
# cold worker into MODEL_DIR, which should live on a network volume so the download is
# paid once per volume rather than once per worker.
# One resolver for both, so the entrypoint and a manual run can never disagree:
# /runpod-volume on a serverless worker, /workspace on a pod, container disk otherwise.
MODEL_DIR="${MODEL_DIR:-$(/download_models.sh --print-dir)}"
export MODEL_DIR
echo "Model directory: $MODEL_DIR"

if [ "$MODEL_DIR" = "/ComfyUI/models" ]; then
    echo "⚠️  No network volume attached (neither /runpod-volume nor /workspace exists),"
    echo "⚠️  so models go to the container disk: EVERY cold worker re-downloads ~31 GB,"
    echo "⚠️  billed as runtime, and the disk is probably too small to hold them at all."
    echo "⚠️  Attach a network volume to the endpoint to fetch them once."
    if [ "${REQUIRE_VOLUME:-0}" = "1" ]; then
        echo "Error: REQUIRE_VOLUME=1 and no network volume is attached. Exiting..."
        exit 1
    fi
fi

if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo "Preparing models in $MODEL_DIR ..."
    /download_models.sh "$MODEL_DIR" || exit 1
fi

# ── Copy models to local disk ───────────────────────────────────────────────────
# ComfyUI streams weights on demand rather than loading them whole, and it reports
#   Model storage policy: fast_disk=False paths=['/runpod-volume/...']
#   Model QwenImage prepared for dynamic VRAM loading. 19582MB Staged.
# when the weights sit on a network volume - meaning ~20 GB is paged over NFS during
# sampling, on every request. Copying to the container disk first costs one sequential
# read per cold worker and makes every request after it fast.
if [ "$MODEL_DIR" != "/ComfyUI/models" ] && [ "${LOCAL_MODEL_COPY:-1}" = "1" ]; then
    need_mb=$(du -sm "$MODEL_DIR" 2>/dev/null | cut -f1)
    free_mb=$(df -Pm /ComfyUI/models 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "$need_mb" ] && [ -n "$free_mb" ] && \
       [ "$free_mb" -gt "$((need_mb + 5120))" ]; then
        echo "Copying $((need_mb / 1024)) GB of models from $MODEL_DIR to the container disk"
        echo "(local reads keep ComfyUI's weight streaming off the network volume)..."
        copy_start=$SECONDS
        if cp -rn "$MODEL_DIR"/. /ComfyUI/models/ 2>/dev/null; then
            echo "Copied in $((SECONDS - copy_start))s; using /ComfyUI/models"
            MODEL_DIR=/ComfyUI/models
            export MODEL_DIR
        else
            echo "⚠️  Copy failed; falling back to reading models from $MODEL_DIR"
        fi
    else
        echo "⚠️  Not copying models to the container disk: needs"
        echo "⚠️  $(( (${need_mb:-0} + 5120) / 1024 )) GB, $(( ${free_mb:-0} / 1024 )) GB free."
        echo "⚠️  ComfyUI will stream weights from the network volume, which is SLOW."
        echo "⚠️  Raise the endpoint's container disk to fix it."
    fi
fi

# Point ComfyUI at MODEL_DIR in addition to its own models directory. These paths are
# appended to the search list, so anything baked into /ComfyUI/models still resolves.
if [ "$MODEL_DIR" != "/ComfyUI/models" ]; then
    cat > /ComfyUI/extra_model_paths.yaml <<YAML
volume:
    base_path: $MODEL_DIR
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    vae: vae
    loras: loras
    upscale_models: upscale_models
YAML
    echo "Wrote /ComfyUI/extra_model_paths.yaml -> $MODEL_DIR"
fi

# ── Launch flags ────────────────────────────────────────────────────────────────
# On a 40 GB+ card the whole model set fits in VRAM, so --highvram keeps it resident
# instead of staging weights per request. Set COMFY_ARGS to override entirely.
if [ -z "${COMFY_ARGS:-}" ]; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${VRAM_MB:-}" ] && [ "$VRAM_MB" -ge 40000 ]; then
        COMFY_ARGS="--highvram"
        echo "GPU has ${VRAM_MB} MB: starting ComfyUI with --highvram"
    fi
fi

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention ${COMFY_ARGS:-} &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120  # 최대 2분 대기
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler in the foreground
# 이 스크립트가 컨테이너의 메인 프로세스가 됩니다.
echo "Starting the handler..."
exec python handler.py