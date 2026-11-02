"""
RunPod serverless worker: Qwen-Image-Edit-2511 (+ Lightning 4-step LoRA) on ComfyUI,
with optional Real-ESRGAN upscaling of the edited image.

Built on the wlsdml1114/qwen_image_edit worker: ComfyUI runs in the background
(see entrypoint.sh) and this handler drives it over its HTTP/websocket API with one
of the workflow graphs in ./workflow. The upscale pass is grafted onto the graph at
request time (SaveImage is rewired behind UpscaleModelLoader -> ImageUpscaleWithModel
-> ImageScaleBy), so the stock workflows stay usable unchanged in the ComfyUI editor.

Request:

  {"input": {
      "prompt": "...",                        # required: the edit instruction
      "negative_prompt": "",                  # optional (only matters at cfg > 1)

      # Picture 1 - the image being edited. One of:
      "image_base64": "<base64 or data URI>",
      "image_url":   "https://...",
      "image_path":  "/abs/path.png",         # path inside the container

      # Picture 2 / 3 - optional extra references, same three forms:
      "image_base64_2" | "image_url_2" | "image_path_2",
      "image_base64_3" | "image_url_3" | "image_path_3",
      "reference_image_base64" / "reference_image_url"   # aliases for Picture 2

      "upscale": 2.0,            # 1 = off, up to MAX_UPSCALE (2 on a 1 MP edit = ~4 MP)
      "upscale_model": "RealESRGAN_x4plus.pth",   # any file in models/upscale_models
      "edit_mp": 1.0,            # megapixels Qwen edits at (Qwen's native size is 1)
      "steps": 4,                # Lightning LoRA is trained for 4
      "true_cfg_scale": 1.0,     # 1.0 with Lightning
      "seed": 42,                # omit or -1 for random
      "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
      "lora_strength": 1.0, "shift": 3.0,
      "output_format": "png",    # png | jpeg | webp
      "jpeg_quality": 95
  }}

Response:

  {"image": "data:image/png;base64,...", "images": [...], "width": W, "height": H,
   "seed": S, "upscale": 2.0,
   "timings": {"total_s": .., "comfy_s": .., "gpu_s": .., "steps": ..}}

  or {"error": "..."}
"""

import base64
import binascii
import io
import json
import logging
import os
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import runpod
import websocket
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen-worker")

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT", "8188")
BASE_URL = f"http://{SERVER_ADDRESS}:{SERVER_PORT}"
CLIENT_ID = str(uuid.uuid4())

WORKFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow")
WORKFLOW_FILES = {
    1: "qwen_image_edit_1_1image.json",
    2: "qwen_image_edit_1_2image.json",
    3: "qwen_image_edit_1_3image.json",
}

# Node ids inside those graphs (identical across the three files).
NODE_IMAGE = {1: "78", 2: "117", 3: "119"}
NODE_SAMPLER = "3"          # KSampler: seed, steps, cfg, sampler_name, scheduler, denoise
NODE_POSITIVE = "111"       # TextEncodeQwenImageEditPlus (prompt)
NODE_NEGATIVE = "110"       # TextEncodeQwenImageEditPlus (negative prompt)
# ImageScaleToTotalPixels, one per picture (93 = Picture 1, 118/120 = the references)
NODE_EDIT_SCALE = ("93", "118", "120")
NODE_SHIFT = "66"           # ModelSamplingAuraFlow -> shift
NODE_LORA = "89"            # LoraLoaderModelOnly -> strength_model
NODE_SAVE = "60"            # SaveImage

# Ids used for the injected upscale chain; high enough not to collide.
NODE_UPSCALE_LOADER = "900"
NODE_UPSCALE_APPLY = "901"
NODE_UPSCALE_RESIZE = "902"

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "4"))
DEFAULT_CFG = float(os.getenv("DEFAULT_CFG", "1.0"))
DEFAULT_EDIT_MP = float(os.getenv("DEFAULT_EDIT_MP", "1.0"))
DEFAULT_UPSCALE = float(os.getenv("DEFAULT_UPSCALE", "1.0"))
UPSCALE_MODEL = os.getenv("UPSCALE_MODEL", "RealESRGAN_x4plus.pth")
# Native scale of UPSCALE_MODEL. The requested factor is reached by running the model
# once and then resampling by factor / UPSCALE_MODEL_SCALE.
UPSCALE_MODEL_SCALE = float(os.getenv("UPSCALE_MODEL_SCALE", "4"))
UPSCALE_RESAMPLE = os.getenv("UPSCALE_RESAMPLE", "lanczos")
MAX_UPSCALE = float(os.getenv("MAX_UPSCALE", "4"))
JOB_TIMEOUT = float(os.getenv("JOB_TIMEOUT", "900"))        # per-request ComfyUI budget
BOOT_TIMEOUT = float(os.getenv("BOOT_TIMEOUT", "300"))      # wait for ComfyUI on cold start


# ───────────────────────── input handling ─────────────────────────

def download_file(url: str, output_path: str) -> str:
    result = subprocess.run(["wget", "-O", output_path, "--no-verbose", url],
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"download failed for {url}: {result.stderr.strip()[:300]}")
    logger.info(f"downloaded {url} -> {output_path}")
    return output_path


def save_base64(data: str, output_path: str) -> str:
    try:
        raw = base64.b64decode(data.split(",", 1)[-1], validate=False)
    except (binascii.Error, ValueError) as e:
        raise RuntimeError(f"invalid base64 image: {e}")
    if not raw:
        raise RuntimeError("empty base64 image")
    with open(output_path, "wb") as f:
        f.write(raw)
    logger.info(f"wrote {len(raw) / 1e6:.1f} MB of base64 input -> {output_path}")
    return output_path


def resolve_image(job_input: dict, slot: int, temp_dir: str):
    """Materialize the image for Picture `slot` as a file, or return None if absent.

    Accepts image_path / image_url / image_base64 (slot 1) and the _2 / _3 variants,
    plus reference_image_* as an alias for slot 2.
    """
    suffix = "" if slot == 1 else f"_{slot}"
    keys = [(f"image_path{suffix}", "path"), (f"image_url{suffix}", "url"),
            (f"image_base64{suffix}", "base64")]
    if slot == 2:
        keys += [("reference_image_path", "path"), ("reference_image_url", "url"),
                 ("reference_image_base64", "base64")]
    for key, kind in keys:
        value = job_input.get(key)
        if not value:
            continue
        if kind == "path":
            if not os.path.isfile(value):
                raise RuntimeError(f"{key} does not exist in the container: {value}")
            return value
        os.makedirs(temp_dir, exist_ok=True)
        dst = os.path.abspath(os.path.join(temp_dir, f"input_{slot}.png"))
        return download_file(value, dst) if kind == "url" else save_base64(value, dst)
    return None


# ───────────────────────── ComfyUI API ─────────────────────────

def wait_for_comfyui(timeout: float):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/", timeout=5).read(1)
            return
        except Exception as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"ComfyUI did not answer on {BASE_URL} within {timeout:.0f}s ({last})")


def queue_prompt(prompt: dict) -> str:
    data = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        # ComfyUI returns the per-node validation errors in the body; without them a
        # rejected graph looks like a bare "HTTP Error 400: Bad Request".
        raise RuntimeError(f"ComfyUI rejected the workflow: {e.read().decode('utf-8', 'replace')[:1000]}")


def get_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())


def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder,
                                    "type": folder_type})
    with urllib.request.urlopen(f"{BASE_URL}/view?{query}") as response:
        return response.read()


def run_workflow(prompt: dict, timeout: float):
    """Queue the graph, wait for it to finish, return the output images as bytes."""
    ws = websocket.WebSocket()
    ws.settimeout(timeout)
    ws.connect(f"ws://{SERVER_ADDRESS}:{SERVER_PORT}/ws?clientId={CLIENT_ID}")
    try:
        prompt_id = queue_prompt(prompt)
        logger.info(f"queued prompt {prompt_id}")
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise RuntimeError(f"workflow did not finish within {timeout:.0f}s")
            out = ws.recv()
            if not isinstance(out, str):
                continue  # binary preview frame
            message = json.loads(out)
            data = message.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            if message.get("type") == "execution_error":
                raise RuntimeError(f"ComfyUI execution error in node "
                                   f"{data.get('node_type')} ({data.get('node_id')}): "
                                   f"{data.get('exception_message')}")
            if message.get("type") == "execution_interrupted":
                raise RuntimeError("ComfyUI execution was interrupted")
            if message.get("type") == "executing" and data.get("node") is None \
                    and data.get("prompt_id") == prompt_id:
                break
    finally:
        try:
            ws.close()
        except Exception:
            pass

    history = get_history(prompt_id).get(prompt_id) or {}
    status = history.get("status") or {}
    images = []
    for node_output in (history.get("outputs") or {}).values():
        for image in node_output.get("images") or []:
            if image.get("type") == "temp":
                continue  # preview copies of the same frame
            images.append(get_image(image["filename"], image.get("subfolder", ""),
                                    image.get("type", "output")))
    if not images:
        raise RuntimeError(f"the workflow produced no images (status: {status})")
    return images


# ───────────────────────── graph preparation ─────────────────────────

def load_workflow(num_images: int) -> dict:
    path = os.path.join(WORKFLOW_DIR, WORKFLOW_FILES[num_images])
    if not os.path.isfile(path):
        raise RuntimeError(f"workflow file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_upscale(prompt: dict, factor: float, model_name: str):
    """Rewire SaveImage through an ESRGAN upscale of the decoded image.

    The model runs at its native scale (x4 for RealESRGAN_x4plus); anything other
    than that exact factor is resampled afterwards, so upscale=2 is a x4 pass
    followed by a lanczos halving rather than a plain x2 resize.
    """
    source = prompt[NODE_SAVE]["inputs"]["images"]
    prompt[NODE_UPSCALE_LOADER] = {
        "inputs": {"model_name": model_name},
        "class_type": "UpscaleModelLoader",
        "_meta": {"title": "Load Upscale Model"},
    }
    prompt[NODE_UPSCALE_APPLY] = {
        "inputs": {"upscale_model": [NODE_UPSCALE_LOADER, 0], "image": source},
        "class_type": "ImageUpscaleWithModel",
        "_meta": {"title": "Upscale Image (using Model)"},
    }
    out = [NODE_UPSCALE_APPLY, 0]
    scale_by = factor / UPSCALE_MODEL_SCALE
    if abs(scale_by - 1.0) > 1e-3:
        prompt[NODE_UPSCALE_RESIZE] = {
            "inputs": {"upscale_method": UPSCALE_RESAMPLE,
                       "scale_by": round(scale_by, 4), "image": out},
            "class_type": "ImageScaleBy",
            "_meta": {"title": "Upscale Image By"},
        }
        out = [NODE_UPSCALE_RESIZE, 0]
    prompt[NODE_SAVE]["inputs"]["images"] = out
    logger.info(f"upscale x{factor} via {model_name} "
                f"(model x{UPSCALE_MODEL_SCALE:g}, then x{scale_by:.3f} {UPSCALE_RESAMPLE})")


def build_prompt(job_input: dict, image_paths: list, seed: int) -> dict:
    prompt = load_workflow(len(image_paths))

    for slot, path in enumerate(image_paths, start=1):
        prompt[NODE_IMAGE[slot]]["inputs"]["image"] = path

    prompt[NODE_POSITIVE]["inputs"]["prompt"] = job_input["prompt"]
    prompt[NODE_NEGATIVE]["inputs"]["prompt"] = job_input.get("negative_prompt") or ""

    sampler = prompt[NODE_SAMPLER]["inputs"]
    sampler["seed"] = seed
    sampler["steps"] = int(job_input.get("steps") or DEFAULT_STEPS)
    sampler["cfg"] = float(job_input.get("true_cfg_scale")
                           or job_input.get("cfg") or DEFAULT_CFG)
    if job_input.get("sampler_name"):
        sampler["sampler_name"] = job_input["sampler_name"]
    if job_input.get("scheduler"):
        sampler["scheduler"] = job_input["scheduler"]
    if job_input.get("denoise") is not None:
        sampler["denoise"] = float(job_input["denoise"])

    edit_mp = max(0.25, min(float(job_input.get("edit_mp") or DEFAULT_EDIT_MP), 4.0))
    for node in NODE_EDIT_SCALE:        # every picture, not just the one being edited
        if node in prompt:
            prompt[node]["inputs"]["megapixels"] = edit_mp
    if job_input.get("shift") is not None:
        prompt[NODE_SHIFT]["inputs"]["shift"] = float(job_input["shift"])
    if job_input.get("lora_strength") is not None:
        prompt[NODE_LORA]["inputs"]["strength_model"] = float(job_input["lora_strength"])

    factor = float(job_input.get("upscale", DEFAULT_UPSCALE) or 1.0)
    if factor > 1.0:
        add_upscale(prompt, factor, job_input.get("upscale_model") or UPSCALE_MODEL)
    return prompt


# ───────────────────────── output ─────────────────────────

def encode_output(raw: bytes, fmt: str, quality: int):
    """(data URI, width, height). ComfyUI saves PNG; convert only if asked to."""
    with Image.open(io.BytesIO(raw)) as im:
        width, height = im.size
        if fmt == "png":
            return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), width, height
        buf = io.BytesIO()
        if fmt == "jpeg":
            im.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0)
        else:
            im.save(buf, "WEBP", quality=quality)
        data = buf.getvalue()
    return f"data:image/{fmt};base64," + base64.b64encode(data).decode("ascii"), width, height


# ───────────────────────── handler ─────────────────────────

def handler(job):
    job_input = job.get("input") or {}
    t0 = time.time()
    image_paths = []
    try:
        prompt_text = (job_input.get("prompt") or "").strip()
        if not prompt_text:
            return {"error": "prompt is required"}
        job_input["prompt"] = prompt_text

        factor = float(job_input.get("upscale", DEFAULT_UPSCALE) or 1.0)
        if not (1.0 <= factor <= MAX_UPSCALE):
            return {"error": f"upscale must be between 1 and {MAX_UPSCALE:g}"}
        fmt = str(job_input.get("output_format", "png")).lower().replace("jpg", "jpeg")
        if fmt not in ("png", "jpeg", "webp"):
            return {"error": "output_format must be png, jpeg or webp"}
        quality = int(job_input.get("jpeg_quality") or 95)

        seed = job_input.get("seed")
        seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)

        temp_dir = os.path.join("/tmp", f"task_{uuid.uuid4().hex}")
        for slot in (1, 2, 3):
            path = resolve_image(job_input, slot, temp_dir)
            if path is None:
                break           # Picture 3 without Picture 2 is not a thing
            image_paths.append(path)
        if not image_paths:
            return {"error": "an image is required (image_base64, image_url or image_path)"}

        logger.info(f"job: {len(image_paths)} image(s), seed {seed}, upscale x{factor:g}, "
                    f"steps {job_input.get('steps') or DEFAULT_STEPS}, format {fmt}")
        graph = build_prompt(job_input, image_paths, seed)

        wait_for_comfyui(BOOT_TIMEOUT)
        t1 = time.time()
        images = run_workflow(graph, JOB_TIMEOUT)
        comfy_s = time.time() - t1

        data_uri, width, height = encode_output(images[0], fmt, quality)
        extra = [encode_output(raw, fmt, quality)[0] for raw in images[1:]]
        logger.info(f"done: {width}x{height} in {comfy_s:.1f}s on the GPU")
        return {
            "image": data_uri,
            "images": [data_uri] + extra,
            "width": width,
            "height": height,
            "seed": seed,
            "upscale": factor,
            "timings": {"total_s": round(time.time() - t0, 2),
                        "comfy_s": round(comfy_s, 2),
                        "gpu_s": round(comfy_s, 2),
                        "steps": graph[NODE_SAMPLER]["inputs"]["steps"]},
        }
    except Exception as e:
        logger.exception("job failed")
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        # only the files this request materialized; an image_path input is the caller's
        for path in image_paths:
            if path.startswith("/tmp/task_"):
                try:
                    os.remove(path)
                    os.rmdir(os.path.dirname(path))
                except OSError:
                    pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
