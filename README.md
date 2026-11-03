# Qwen-Image-Edit-2511 + upscaling — RunPod serverless worker

Instruction-driven image editing with **Qwen-Image-Edit-2511** (fp8) and the
**Lightning 4-step LoRA**, plus an optional **Real-ESRGAN / 4x-UltraSharp upscale pass**
on the result.

The worker runs ComfyUI inside the container and drives it over its HTTP/websocket API,
following the [wlsdml1114/qwen_image_edit](https://github.com/wlsdml1114/qwen_image_edit)
template.

> **Attach a network volume to the endpoint.** The model set (~31 GB) is not baked into
> the image — see [Models](#models) — and is downloaded onto the volume by the first
> cold worker, once per volume.

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

## Which GPU

The fp8 transformer is ~20 GB and the fp8 text encoder ~9 GB. On a 24 GB card they
cannot both stay resident, so ComfyUI evicts one and reloads it on the next request -
paid on every job, and easily mistaken for slow generation. 48 GB keeps both loaded.

| GPU | RunPod id | |
| --- | --- | --- |
| L40S / RTX 6000 Ada, 48 GB | `ADA_48_PRO` | **preferred**: everything resident, no arch risk |
| A6000 / A40, 48 GB | `AMPERE_48` | same headroom, slower per step |
| RTX 5090, 32 GB | `ADA_32_PRO` | enough VRAM, but needs a torch with `sm_120` kernels - check the startup log |
| 4090 / L4, 24 GB | `ADA_24` | cheapest; reloads a model every request |

Startup logs the card, its compute capability and the arch list torch was built for,
and warns when the two do not match - a Blackwell card on a pre-CUDA-12.8 torch runs
via PTX JIT at a fraction of its real speed.


## Models

| file | size | where |
| --- | --- | --- |
| `qwen_image_edit_2511_fp8mixed.safetensors` | ~20 GB | `diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~9 GB | `text_encoders/` |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | ~1.2 GB | `loras/` |
| `qwen_image_vae.safetensors` | ~250 MB | `vae/` |
| `RealESRGAN_x4plus.pth`, `4x-UltraSharp.pth` | ~64 MB each | `upscale_models/` |

None of this is baked into the image. A RunPod **hub build cannot export and upload a
~31 GB layer inside its 30-minute limit** — it dies in `exporting layers` /
`sending tarball` with `Build timeout exceeded`. So `entrypoint.sh` runs
`download_models.sh` before starting ComfyUI, which fetches whatever is missing from
`MODEL_DIR` in parallel, under a per-file lock and via `.part` files, so several cold
workers sharing one volume cooperate instead of fighting. ComfyUI is then pointed at
`MODEL_DIR` through a generated `extra_model_paths.yaml`.

`MODEL_DIR` is detected from where the volume actually is, because that differs by
how you run the container:

1. `MODEL_DIR`, if set — always wins.
2. Any candidate that **already holds the models** (checked by looking for the
   transformer), so a full-but-populated volume is never passed over.
3. `/runpod-volume` (serverless), then `/workspace` (pod), then **any other real mount**
   read out of `/proc/mounts` — that last step is what finds a global or otherwise
   custom-mounted volume. A candidate needs `MIN_VOLUME_GB` (40) free to be chosen.
4. `/ComfyUI/models` — the container disk, with a warning.

The last row is the bad one: **every cold worker re-downloads ~31 GB, billed as
runtime**, and the container disk is usually too small to hold them anyway. Startup
warns loudly, the free-space check refuses rather than failing mid-download, and
`REQUIRE_VOLUME=1` turns it into a refusal to start at all. Set `MODEL_DIR` explicitly
to override the detection.

To pre-warm a volume from a cheap CPU pod, mount it and run the same script — it picks
`/workspace/ComfyUI-models` on its own, which the serverless workers then see at
`/runpod-volume/ComfyUI-models`:

```bash
./download_models.sh                # or: ./download_models.sh --print-dir  to check first
```

For a local build you can still bake everything in, since no 30-minute limit applies:

```bash
docker build --build-arg BAKE_MODELS=1 -t <you>/qwen-edit-upscale .
```

## Environment variables

| var | default | |
| --- | --- | --- |
| `MODEL_DIR` | detected (see [Models](#models)) | where models are cached |
| `REQUIRE_VOLUME` | `0` | `1` = refuse to start without a network volume |
| `MIN_VOLUME_GB` | `40` | free space a mount needs before it is chosen automatically |
| `SKIP_MODEL_DOWNLOAD` | `0` | `1` = assume the models are already in place |
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

Without the models the image is just the base plus ComfyUI, which builds in minutes and
fits comfortably inside a RunPod hub build. `containerDiskInGb` is 100 in
`.runpod/hub.json`: enough for the image, and enough to hold the model set on the
container disk if no network volume is attached.

The first cold worker on a fresh volume spends a few minutes downloading models before
ComfyUI starts, so give the endpoint a generous execution timeout for its first request.

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
