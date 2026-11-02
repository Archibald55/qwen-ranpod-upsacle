# Qwen-Image-Edit-2511 + upscaler — RunPod serverless worker

Edits each photo with Qwen-Image-Edit-2511 (Lightning LoRA, 8 steps by default) at ~1 MP,
then upscales with Real-ESRGAN x4 and resizes to the requested factor
(`upscale: 2` → ~4 MP). The API matches the FLUX worker, so `qwen_enhance_batch.py` works with it.

## 1. Build and push the image

```bash
cd qwen-worker
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/qwen-edit-upscale:v1 .
docker push YOUR_DOCKERHUB_USER/qwen-edit-upscale:v1
```

No Docker locally? Push this folder to a GitHub repo and use **Serverless → New Endpoint →
GitHub repo**; RunPod builds the Dockerfile for you.

## 2. Create the endpoint (RunPod console → Serverless → New Endpoint → Docker image)

| Setting | Value |
|---|---|
| Container image | `YOUR_DOCKERHUB_USER/qwen-edit-upscale:v1` |
| **Model (cached model)** | `Qwen/Qwen-Image-Edit-2511` ← important, avoids a 58 GB download per worker |
| GPU | **80 GB** (A100 80GB or H100). The bf16 model needs ~58 GB VRAM |
| Container disk | 20 GB |
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
| `CPU_OFFLOAD` | 0 | 1 = run on a 48 GB GPU by offloading to RAM (slow; needs ~64 GB RAM) |
| `UPSCALE_TILE` | 512 | upscaler tile size; lower if you ever hit OOM there |

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
the GPU's per-second rate (A100 80GB flex was ~$0.00076/s; check current pricing).
For 800 images at ~15 s that is roughly $9 on an A100, plus a cold start per worker.

## Notes

- First request on a fresh worker loads the model (cached: roughly 1–2 min). Then ~10–25 s per image
  depending on GPU and steps (measure with a few test images; `timings` is in every response).
- A 4 MP PNG is ~10–15 MB as base64. If you hit response-size errors, use `output_format: "jpeg"`
  (batch script: `--format jpg`).
- If colours look off with the Lightning LoRA, compare against `USE_LIGHTNING=0` on a few images;
  there are community reports of colour shifts with diffusers + Lightning.
