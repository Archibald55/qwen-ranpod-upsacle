#!/bin/bash
# Fetch the Qwen edit model set into a model directory, skipping whatever is already
# there. Run at container start (entrypoint.sh) against a network volume so the
# ~31 GB is paid once per volume, or at build time with --build-arg BAKE_MODELS=1.
#
#   ./download_models.sh /runpod-volume/ComfyUI-models
#
# Safe to run concurrently: each file is downloaded under an flock and moved into
# place only once complete, so several cold workers sharing one volume cooperate
# instead of writing over each other.

set -uo pipefail

DEST="${1:-${MODEL_DIR:-/ComfyUI/models}}"
# Models already in the image (the upscalers) are not re-fetched onto the volume.
BAKED_DIR="${BAKED_DIR:-/ComfyUI/models}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-3600}"

# name|subdirectory|url|approximate size in MB (for the free-space check)
MANIFEST=(
"qwen_image_edit_2511_fp8mixed.safetensors|diffusion_models|https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors|20000"
"qwen_2.5_vl_7b_fp8_scaled.safetensors|text_encoders|https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors|9400"
"qwen_image_vae.safetensors|vae|https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors|260"
"Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors|loras|https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors|1200"
"RealESRGAN_x4plus.pth|upscale_models|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth|70"
"4x-UltraSharp.pth|upscale_models|https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth|70"
)

have() {   # already present in the volume or baked into the image?
    local sub="$1" name="$2" dir
    for dir in "$DEST" "$BAKED_DIR"; do
        [ -s "$dir/$sub/$name" ] && return 0
    done
    return 1
}

fetch() {
    local name="$1" sub="$2" url="$3"
    local target="$DEST/$sub/$name"

    if have "$sub" "$name"; then
        echo "cached    $sub/$name"
        return 0
    fi
    mkdir -p "$DEST/$sub"

    # Serialize per file: another worker on this volume may be mid-download.
    # Without flock (not on every base image) the .part file still keeps the result
    # correct; two workers would just duplicate the transfer.
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$target.lock"
        if ! flock -w "$LOCK_TIMEOUT" 9; then
            echo "ERROR     timed out waiting for another worker to finish $sub/$name" >&2
            return 1
        fi
        if [ -s "$target" ]; then
            echo "cached    $sub/$name (fetched by another worker)"
            return 0
        fi
    fi

    echo "download  $sub/$name"
    local start=$SECONDS
    if ! wget --tries=3 --timeout=60 --continue --no-verbose -O "$target.part" "$url"; then
        echo "ERROR     download failed: $url" >&2
        rm -f "$target.part"
        return 1
    fi
    mv -f "$target.part" "$target"
    local gb
    gb=$(du -m "$target" | cut -f1)
    echo "ready     $sub/$name ($((gb / 1024)).$(((gb % 1024) * 10 / 1024)) GB in $((SECONDS - start))s)"
}

echo "model directory: $DEST"
mkdir -p "$DEST"

# Check there is room before starting. Out of space, wget reports the baffling
# "Cannot write to '...' (Success)" and the real cause - usually models landing on a
# small container disk because no network volume is attached - stays invisible.
needed_mb=0
for entry in "${MANIFEST[@]}"; do
    IFS='|' read -r name sub url size_mb <<<"$entry"
    have "$sub" "$name" || needed_mb=$((needed_mb + size_mb))
done
free_mb=$(df -Pk "$DEST" 2>/dev/null | awk 'NR==2 {print int($4 / 1024)}')
if [ -n "$free_mb" ] && [ "$needed_mb" -gt 0 ]; then
    echo "need $((needed_mb / 1024)) GB, $((free_mb / 1024)) GB free on $DEST"
    if [ "$free_mb" -lt "$((needed_mb + 2048))" ]; then
        echo "ERROR: not enough disk space: $((needed_mb / 1024)) GB of models still to" >&2
        echo "       download but only $((free_mb / 1024)) GB free on $DEST." >&2
        case "$DEST" in
            /runpod-volume/*) echo "       Grow the network volume, or delete what else is on it." >&2 ;;
            *) echo "       The models are going to the CONTAINER DISK, which is far too small" >&2
               echo "       for them. Attach a network volume to the endpoint (it mounts at" >&2
               echo "       /runpod-volume) so they are downloaded once and cached, or raise" >&2
               echo "       the endpoint's container disk above $(((needed_mb + 2048) / 1024)) GB." >&2 ;;
        esac
        exit 1
    fi
fi

pids=()
for entry in "${MANIFEST[@]}"; do
    IFS='|' read -r name sub url size_mb <<<"$entry"
    fetch "$name" "$sub" "$url" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if [ "$status" -ne 0 ]; then
    echo "ERROR: one or more models could not be downloaded" >&2
    exit 1
fi
echo "all models present in $DEST"
