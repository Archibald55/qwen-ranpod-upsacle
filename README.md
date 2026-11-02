# Qwen-Image-Edit-2511 + upscaling — RunPod serverless worker

Instruction-driven image editing with **Qwen-Image-Edit-2511** (fp8) and the
**Lightning 4-step LoRA**, plus an optional **Real-ESRGAN / 4x-UltraSharp upscale pass**
on the result.

The worker runs ComfyUI inside the container and drives it over its HTTP/websocket API,
following the [wlsdml1114/qwen_image_edit](https://github.com/wlsdml1114/qwen_image_edit)
template. Everything (model, LoRA, text encoder, VAE, upscalers) is baked into the image,
so a cold worker downloads nothing — no network volume needed.

## Request

```json
{
  "input": {
    "prompt": "put the woman in a black coat, full body shot",
    "image_base64": "<base64 or data URI>",
    "upscale": 2.0,
    "steps": 4,
    "seed": 42
  }
}
```

| field | default | meaning |
| --- | --- | --- |
| `prompt` | **required** | the edit instruction |
| `negative_prompt` | `""` | only matters at `true_cfg_scale > 1` |
| `image_base64` / `image_url` / `image_path` | **required** | Picture 1 — the image being edited |
| `…_2`, `…_3` | – | Picture 2 / 3, extra references (`reference_image_base64` is an alias for Picture 2) |
| `upscale` | `1.0` (`DEFAULT_UPSCALE`) | output scale on top of the edit, up to `MAX_UPSCALE` (4). `2` on a 1 MP edit ≈ 4 MP |
| `upscale_model` | `RealESRGAN_x4plus.pth` | `RealESRGAN_x4plus.pth` or `4x-UltraSharp.pth` |
| `edit_mp` | `1.0` | megapixels Qwen edits at; 1.0 is the model's native size |
| `steps` | `4` (`DEFAULT_STEPS`) | the Lightning LoRA is trained for 4 |
| `true_cfg_scale` | `1.0` | 1.0 with Lightning; ~4 without |
| `seed` | random | omit or `-1` for random |
| `sampler_name`, `scheduler`, `denoise`, `shift`, `lora_strength` | from the workflow | advanced overrides |
| `output_format` | `png` | `png`, `jpeg` or `webp` |
| `jpeg_quality` | `95` | for `jpeg` / `webp` |

Images may be given three ways, for every slot: `image_base64` (raw base64 or a
`data:image/...` URI), `image_url` (fetched with wget) or `image_path` (a path that
already exists inside the container).

## Response

```json
{
  "image": "data:image/png;base64,...",
  "images": ["data:image/png;base64,..."],
  "width": 2048,
  "height": 2048,
  "seed": 42,
  "upscale": 2.0,
  "timings": {"total_s": 14.2, "comfy_s": 12.9, "gpu_s": 12.9, "steps": 4}
}
```

Errors come back as `{"error": "..."}` with the failing ComfyUI node named, rather than
killing the worker.

## How the upscale pass works

The three graphs in `workflow/` are the stock Qwen edit workflows and end at
`SaveImage`. When a request asks for `upscale > 1`, the handler grafts

```
VAEDecode → UpscaleModelLoader → ImageUpscaleWithModel → ImageScaleBy → SaveImage
```

onto the graph in memory. The ESRGAN model always runs at its native x4, and
`ImageScaleBy` resamples to the factor actually requested — so `upscale: 2` is a x4
model pass followed by a lanczos halving, which is visibly cleaner than a plain x2
resize. With `upscale: 1` the graph is left untouched.

Because the upscale runs as ComfyUI nodes, it is tiled and VRAM-bounded by ComfyUI
itself; a 4x pass on a 1 MP edit fits on a 24 GB card.

## Environment variables

| var | default | |
| --- | --- | --- |
| `DEFAULT_STEPS` | `4` | used when a request omits `steps` |
| `DEFAULT_CFG` | `1.0` | used when a request omits `true_cfg_scale` |
| `DEFAULT_EDIT_MP` | `1.0` | used when a request omits `edit_mp` |
| `DEFAULT_UPSCALE` | `1.0` | used when a request omits `upscale` |
| `UPSCALE_MODEL` | `RealESRGAN_x4plus.pth` | default upscaler |
| `UPSCALE_MODEL_SCALE` | `4` | native scale of that model |
| `UPSCALE_RESAMPLE` | `lanczos` | resample method after the model pass |
| `MAX_UPSCALE` | `4` | requests above this are rejected |
| `JOB_TIMEOUT` | `900` | seconds one workflow may run |
| `BOOT_TIMEOUT` | `300` | seconds to wait for ComfyUI on a cold worker |
| `SERVER_ADDRESS`, `SERVER_PORT` | `127.0.0.1`, `8188` | where ComfyUI listens |

## Build

```bash
docker build -t <you>/qwen-edit-upscale:latest .
docker push <you>/qwen-edit-upscale:latest
```

The image is large (~70 GB: fp8 transformer, 7B text encoder, VAE, LoRA, upscalers), so
`containerDiskInGb` is set to 180 in `.runpod/hub.json`.

## Batch client

`qwen_enhance_batch.py` (one directory up in this working tree) walks a folder of
photos, sends each one to this endpoint with upscaling, and writes the results into a
mirror folder tree:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python qwen_enhance_batch.py ./photos ./photos_enhanced --upscale 2
```

## Credits

- Worker scaffold, ComfyUI setup and workflows: [wlsdml1114/qwen_image_edit](https://github.com/wlsdml1114/qwen_image_edit)
- [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) · [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) · [4x-UltraSharp](https://huggingface.co/Kim2091/UltraSharp)
