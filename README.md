# Qwen-Image-Edit-2511 + upscaler — RunPod serverless worker

Edits each photo with Qwen-Image-Edit-2511 (Q4_K_M GGUF + Lightning LoRA, 8 steps by default) at ~1 MP,
then upscales with Real-ESRGAN x4 and resizes to the requested factor
(`upscale: 2` → ~4 MP). The API matches the FLUX worker, so `qwen_enhance_batch.py` works with it.

## 0. Test on a GPU pod first

Do not debug this on serverless: every fix costs an image build plus a cold start.
Rent an on-demand **GPU Pod** (24 GB, billed per minute), and iterate there in seconds.

```bash
git clone YOUR_REPO && cd qwen-worker
pip install -r requirements.txt
# either let it download the quant, or put it somewhere and point GGUF_PATH at it
export GGUF_PATH=/workspace/qwen-image-edit-2511-Q4_K_M.gguf
python smoke_gpu.py                 # loads the model, runs 1 edit, prints per-stage timing
python smoke_gpu.py photo.jpg       # with a real photo
```

`smoke_gpu.py` reports GPU name and VRAM, whether the GGUF was found, how long each
load stage took, free host RAM at each step, peak VRAM, and a warm second pass (the
steady-state time per image). Exit code 0 means an image came back. Only build the
serverless image once this passes.

## 1. Build and push the image

```bash
cd qwen-worker
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/qwen-edit-upscale:v2 .
docker push YOUR_DOCKERHUB_USER/qwen-edit-upscale:v2
```

No Docker locally? Push this folder to a GitHub repo and use **Serverless → New Endpoint →
GitHub repo**; RunPod builds the Dockerfile for you.

## 2. Create the endpoint (RunPod console → Serverless → New Endpoint → Docker image)

| Setting | Value |
|---|---|
| Container image | `YOUR_DOCKERHUB_USER/qwen-edit-upscale:v2` |
| **Model (cached model)** | `Qwen/Qwen-Image-Edit-2511` ← important, avoids a 16 GB text-encoder download per worker |
| GPU | **24 GB** (RTX 4090, L4 or A5000). 48 GB also works and is less contended |
| Container disk | 40 GB (the image carries the 12.8 GB quant) |
| Active workers | 0 |
| Max workers | 4 (match `--workers` in the batch script) |
| Idle timeout | 10 s |
| Execution timeout | 600 s |
| FlashBoot | on |

### Optional environment variables

| Var | Default | Meaning |
|---|---|---|
| `DEFAULT_STEPS` | 8 | steps when the request doesn't send `steps` (try 4 for cheaper) |
| `USE_LIGHTNING` | 1 | 0 = plain model (use ~40 steps, cfg 4; much slower) |
| `QUANT` | gguf | `gguf` = Q4_K_M transformer (24 GB GPU). `bnb4` = NF4. `off` = bf16, needs 80 GB |
| `GGUF_FILE` | Q4_K_M | Q5_K_M (15.4 GB) or Q6_K (17.9 GB) on a 32-48 GB GPU. Only Q4_K_M is baked in — others download on cold start |
| `TEXT_ENCODER_4BIT` | 1 | NF4 Qwen2.5-VL: 16 GB → ~5.5 GB. Required on 24 GB |
| `REQUIRE_LOCAL` | 0 | 1 = refuse to start instead of downloading the text encoder at runtime |
| `LOCAL_BASE` | /models/base | components baked into the image (used before RunPod's cache) |
| `CPU_OFFLOAD` | 0 | Last resort if the GPU still OOMs (slow; needs ~64 GB RAM) |
| `UPSCALE_TILE` | 512 | upscaler tile size; lower if you ever hit OOM there |

### Stopping the per-worker download

`Fetching 4 files` in the worker log is the text encoder (~16 GB) coming from Hugging
Face, billed as runtime, on every cold worker. The 40 GB transformer is not downloaded
(the GGUF replaces it, and diffusers skips folders for components passed in directly).
Three ways to avoid it, best first:

1. **Cache it on the endpoint** (free, no rebuild): Manage -> Edit Endpoint -> Model =
   `Qwen/Qwen-Image-Edit-2511`. Mounts at `/runpod-volume/huggingface-cache/hub/` and the
   handler loads from there. Confirm it worked: the log should say `using cached model at ...`.
2. **Bake it into the image**: `docker build --build-arg BAKE_BASE=1 ...`. Adds ~17 GB to
   the image (text encoder, VAE, tokenizer; not the transformer) and a cold worker then
   downloads nothing at all. Use this if the cached-model field keeps missing.
3. **Network volume**: attach one and point `HF_HOME` at it, so the download happens once
   and persists across workers.

Set `REQUIRE_LOCAL=1` once one of those is in place: the worker then fails loudly on a
cache miss instead of quietly re-downloading 16 GB per worker.

## 3. Request format

```json
{"input": {
  "prompt": "…",
  "image_base64": "<base64>",
  "reference_image_base64": "<base64, optional style reference>",
  "upscale": 2.0,
  "seed": 42,
  "steps": 8,
  "true_cfg_scale": 1.0,
  "edit_mp": 1.0,
  "output_format": "png"
}}
```

Response: `{"image": "data:image/png;base64,…", "width", "height", "seed", "timings"}` or `{"error": "…"}`.

In prompts, the photo is **Picture 1** and the reference is **Picture 2** (Qwen's naming).

## 4. Run the batch

```bash
pip install requests pillow pillow-heif
export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
python qwen_enhance_batch.py ./photos ./test_out --limit 10          # test 10 first
python qwen_enhance_batch.py ./photos ./photos_enhanced              # all, ~4 MP PNG
python qwen_enhance_batch.py ./photos ./photos_enhanced --steps 4    # cheaper
python qwen_enhance_batch.py ./photos ./photos_enhanced --format jpg # smaller files
```

Each log line shows output size and GPU seconds (`gpu 10.7s`). Cost per image ≈ GPU seconds ×
the GPU's per-second rate (a 24 GB flex worker was ~$0.00031/s; check current pricing).
For 800 images at ~12 s that is roughly $3, plus a cold start per worker.

## Notes

- First request on a fresh worker loads the model (cached: roughly 1–2 min). Then ~10–20 s per image
  depending on GPU and steps (measure with a few test images; `timings` is in every response).
  The startup log line reports allocated and free VRAM — check it after changing `GGUF_FILE`.
- A 4 MP PNG is ~10–15 MB as base64. If you hit response-size errors, use `output_format: "jpeg"`
  (batch script: `--format jpg`).
- If colours look off with the Lightning LoRA, compare against `USE_LIGHTNING=0` on a few images;
  there are community reports of colour shifts with diffusers + Lightning.
- Q4 is a real quality trade-off. For fidelity-critical work (exact colours, fine texture) compare
  Q4_K_M against Q6_K or `QUANT=off` on a handful of images before committing to a full batch.
- With a quantized transformer the Lightning LoRA stays an active adapter instead of being fused
  into the weights (`fuse_lora` cannot write into quantized tensors). Same output, a few % slower.
