"""
RunPod serverless worker: Qwen-Image-Edit-2511 (+ Lightning LoRA) with built-in
Real-ESRGAN upscaling.

Same request/response shape as the FLUX.2-klein worker used by enhance_batch.py:

  {"input": {
      "prompt": "...",                       # required
      "image_base64": "<base64 or data URI>",  # required, the photo to edit (Picture 1)
      "reference_image_base64": "<base64>",  # optional style reference (Picture 2)
      "upscale": 2.0,                        # 1 = none, up to 4 (2 on a ~1 MP edit = ~4 MP)
      "seed": 42,                            # optional, omit / -1 = random
      "steps": 8,                            # optional, default DEFAULT_STEPS
      "true_cfg_scale": 1.0,                 # optional, 1.0 with Lightning, ~4 without
      "negative_prompt": " ",                # optional (only used when true_cfg_scale > 1)
      "edit_mp": 1.0,                        # optional, megapixels Qwen edits at (default 1)
      "output_format": "png"                 # png | jpeg | webp
  }}

  -> {"image": "data:image/png;base64,...", "width": W, "height": H, "seed": S,
      "timings": {...}}      or      {"error": "..."}
"""

import base64
import io
import logging
import math
import os
import random
import time

import numpy as np
import runpod
import torch
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwen-worker")

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
CACHE_ROOT = os.getenv("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")
USE_LIGHTNING = os.getenv("USE_LIGHTNING", "1") == "1"
LORA_PATH = os.getenv("LORA_PATH", "/models/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
UPSCALER_PATH = os.getenv("UPSCALER_PATH", "/models/upscale/RealESRGAN_x4plus.pth")
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"  # for <80 GB GPUs: slower, needs lots of RAM
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "8" if USE_LIGHTNING else "40"))
DEFAULT_CFG = float(os.getenv("DEFAULT_CFG", "1.0" if USE_LIGHTNING else "4.0"))
UPSCALE_TILE = int(os.getenv("UPSCALE_TILE", "512"))
MAX_UPSCALE = 4.0
DEVICE = "cuda"

# Scheduler settings published with Qwen-Image-Lightning for few-step inference.
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


# ───────────────────────── model loading (once per worker) ─────────────────────────

def resolve_model_path(model_id: str) -> str:
    """Use RunPod's cached model if the endpoint has one, else fall back to downloading."""
    snaps = os.path.join(CACHE_ROOT, "models--" + model_id.replace("/", "--"), "snapshots")
    if os.path.isdir(snaps):
        for name in sorted(os.listdir(snaps)):
            path = os.path.join(snaps, name)
            if os.path.isfile(os.path.join(path, "model_index.json")):
                log.info(f"using cached model at {path}")
                return path
    log.warning(f"{model_id} not found in {CACHE_ROOT}; downloading (~58 GB, slow and billed). "
                f"Set the endpoint's Model field to {model_id} to cache it.")
    return model_id


def load_pipeline():
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    t0 = time.time()
    kwargs = {"torch_dtype": torch.bfloat16}
    if USE_LIGHTNING:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    pipe = QwenImageEditPlusPipeline.from_pretrained(resolve_model_path(MODEL_ID), **kwargs)
    if not CPU_OFFLOAD:
        pipe.to(DEVICE)
    if USE_LIGHTNING:
        pipe.load_lora_weights(os.path.dirname(LORA_PATH), weight_name=os.path.basename(LORA_PATH),
                               adapter_name="lightning")
        pipe.fuse_lora(adapter_names=["lightning"], lora_scale=1.0)
        pipe.unload_lora_weights()  # fused weights stay, LoRA layers are removed
    if CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    log.info(f"pipeline ready in {time.time() - t0:.0f}s (lightning={USE_LIGHTNING}, offload={CPU_OFFLOAD})")
    return pipe


def load_upscaler():
    from spandrel import ImageModelDescriptor, ModelLoader

    model = ModelLoader().load_from_file(UPSCALER_PATH)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(f"{UPSCALER_PATH} is not an image-to-image upscaler")
    model = model.cuda().eval()
    half = bool(model.supports_half)
    if half:
        model = model.half()
    log.info(f"upscaler loaded: {model.architecture.name if hasattr(model, 'architecture') else ''} x{model.scale}")
    return model, half


PIPE = load_pipeline()
UPSCALER, UPSCALER_HALF = load_upscaler()


# ───────────────────────── helpers ─────────────────────────

def decode_image(data: str) -> Image.Image:
    raw = base64.b64decode(data.split(",", 1)[-1])
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def edit_size(w: int, h: int, mp: float, multiple: int = 32):
    """Width/height with the same aspect ratio as the input and ~mp megapixels."""
    target = mp * 1024 * 1024
    ratio = w / h
    nw = math.sqrt(target * ratio)
    nh = nw / ratio
    nw = max(multiple * 8, round(nw / multiple) * multiple)
    nh = max(multiple * 8, round(nh / multiple) * multiple)
    return int(nw), int(nh)


@torch.inference_mode()
def esrgan_upscale(img: Image.Image) -> Image.Image:
    """Run the x4 model in overlapping tiles (bounded VRAM), return the x4 image."""
    s = UPSCALER.scale
    dtype = torch.float16 if UPSCALER_HALF else torch.float32
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    _, _, H, W = x.shape
    out = torch.zeros((1, 3, H * s, W * s), dtype=torch.float32)
    tile, pad = UPSCALE_TILE, 16
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, H), min(x1 + pad, W)
            t = x[:, :, ya:yb, xa:xb].to(DEVICE, dtype)
            o = UPSCALER(t).float().clamp_(0, 1).cpu()
            oy, ox = (y0 - ya) * s, (x0 - xa) * s
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
    arr = (out[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def upscale(img: Image.Image, factor: float) -> Image.Image:
    if factor <= 1.0:
        return img
    target = (round(img.width * factor), round(img.height * factor))
    big = esrgan_upscale(img)  # x4
    return big.resize(target, Image.LANCZOS) if big.size != target else big


def encode(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, "JPEG", quality=95, subsampling=0)
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=95)
    else:
        img.save(buf, "PNG", compress_level=4)
    return f"data:image/{fmt};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ───────────────────────── handler ─────────────────────────

def handler(job):
    inp = job.get("input") or {}
    try:
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "prompt is required"}
        if not inp.get("image_base64"):
            return {"error": "image_base64 is required"}
        factor = float(inp.get("upscale", 1.0))
        if not (1.0 <= factor <= MAX_UPSCALE):
            return {"error": f"upscale must be between 1 and {MAX_UPSCALE}"}
        fmt = str(inp.get("output_format", "png")).lower().replace("jpg", "jpeg")
        if fmt not in ("png", "jpeg", "webp"):
            return {"error": "output_format must be png, jpeg or webp"}
        steps = int(inp.get("steps") or DEFAULT_STEPS)
        cfg = float(inp.get("true_cfg_scale") or DEFAULT_CFG)
        edit_mp = min(max(float(inp.get("edit_mp") or 1.0), 0.25), 2.0)
        seed = inp.get("seed")
        seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)

        t0 = time.time()
        main = decode_image(inp["image_base64"])
        images = [main]
        if inp.get("reference_image_base64"):
            images.append(decode_image(inp["reference_image_base64"]))
        w, h = edit_size(main.width, main.height, edit_mp)

        result = PIPE(
            image=images,
            prompt=prompt,
            negative_prompt=inp.get("negative_prompt") or " ",
            true_cfg_scale=cfg,
            guidance_scale=1.0,
            num_inference_steps=steps,
            width=w,
            height=h,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_images_per_prompt=1,
        ).images[0]
        t_edit = time.time() - t0

        t1 = time.time()
        result = upscale(result, factor)
        t_up = time.time() - t1

        return {
            "image": encode(result, fmt),
            "width": result.width,
            "height": result.height,
            "seed": seed,
            "timings": {"edit_s": round(t_edit, 2), "upscale_s": round(t_up, 2), "steps": steps},
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "CUDA out of memory - lower edit_mp, use an 80 GB GPU, or set CPU_OFFLOAD=1"}
    except Exception as e:  # report to the client instead of crashing the worker
        log.exception("job failed")
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
