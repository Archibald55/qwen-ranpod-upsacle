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
if [ -z "${MODEL_DIR:-}" ]; then
    if [ -d /runpod-volume ]; then
        MODEL_DIR=/runpod-volume/ComfyUI-models
    else
        MODEL_DIR=/ComfyUI/models
        echo "⚠️  No network volume attached (/runpod-volume is missing), so models go to"
        echo "⚠️  the container disk and EVERY cold worker re-downloads ~31 GB, billed as"
        echo "⚠️  runtime. Attach a network volume to the endpoint to fetch them once."
        if [ "${REQUIRE_VOLUME:-0}" = "1" ]; then
            echo "Error: REQUIRE_VOLUME=1 and no network volume is attached. Exiting..."
            exit 1
        fi
    fi
fi
export MODEL_DIR

if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo "Preparing models in $MODEL_DIR ..."
    /download_models.sh "$MODEL_DIR" || exit 1
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

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention &

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