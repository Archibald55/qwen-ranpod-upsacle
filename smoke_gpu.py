#!/usr/bin/env python3
"""Load the pipeline and run one edit, printing what each stage costs.

Run this on a RunPod GPU Pod (or any 24 GB GPU) BEFORE building a serverless
image: a pod is billed per minute and you can re-run this in seconds, whereas
every serverless fix costs an image build plus a cold start.

  # on a 24 GB pod, in this folder
  pip install -r requirements.txt
  export GGUF_PATH=/workspace/qwen-image-edit-2511-Q4_K_M.gguf   # or let it download
  python smoke_gpu.py                 # synthetic test image
  python smoke_gpu.py photo.jpg       # a real photo
  QUANT=off python smoke_gpu.py       # compare against bf16 (needs 80 GB)

Exit code is 0 only if an image came back.
"""
import base64
import io
import logging
import os
import sys
import time

log = logging.getLogger("smoke")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        log.error("no CUDA device: run this on a GPU pod, not locally")
        return 2
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"gpu: {name}, {total:.1f} GB")

    import handler  # importing no longer loads the models

    log.info(f"config: QUANT={handler.QUANT} GGUF_PATH={handler.GGUF_PATH} "
             f"exists={os.path.isfile(handler.GGUF_PATH)} "
             f"TEXT_ENCODER_4BIT={handler.TEXT_ENCODER_4BIT} "
             f"USE_LIGHTNING={handler.USE_LIGHTNING}")
    log.info(f"host: {handler.mem()}")

    t0 = time.time()
    try:
        handler.init()
    except Exception:
        log.exception("model load FAILED")
        return 1
    log.info(f"=== loaded in {time.time() - t0:.0f}s ===")

    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1]).convert("RGB")
        log.info(f"input: {sys.argv[1]} {src.width}x{src.height}")
    else:
        src = Image.new("RGB", (1024, 768), (90, 110, 130))
        for x in range(300, 724):           # a shape to check it is not returning noise
            for y in range(200, 568):
                src.putpixel((x, y), (210, 180, 60))
        log.info("input: synthetic 1024x768")
    buf = io.BytesIO()
    src.save(buf, "JPEG", quality=95)

    job = {"input": {
        "prompt": "Replace the background with a clean light gray studio backdrop. "
                  "Keep the object exactly as it is.",
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "upscale": 1.0,                      # skip the upscaler on the first run
        "steps": 4,
        "seed": 42,
        "output_format": "jpeg",
    }}
    t1 = time.time()
    out = handler.handler(job)
    elapsed = time.time() - t1

    if not isinstance(out, dict) or out.get("error"):
        log.error(f"job FAILED after {elapsed:.1f}s: "
                  f"{out.get('error') if isinstance(out, dict) else out}")
        return 1
    img = out.get("image")
    if not img:
        log.error(f"no image in output: {str(out)[:300]}")
        return 1
    raw = base64.b64decode(img.split(",", 1)[-1])
    dst = "smoke_out.jpg"
    with open(dst, "wb") as f:
        f.write(raw)
    log.info(f"OK: {out.get('width')}x{out.get('height')} in {elapsed:.1f}s, "
             f"timings={out.get('timings')}, {len(raw) / 1e6:.1f} MB -> {dst}")
    log.info(f"peak gpu {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of {total:.1f} GB; "
             f"{handler.mem()}")

    log.info("second pass (warm, measures steady-state time per image)")
    t2 = time.time()
    out2 = handler.handler(job)
    if isinstance(out2, dict) and out2.get("error"):
        log.error(f"warm pass FAILED: {out2['error']}")
        return 1
    log.info(f"warm: {time.time() - t2:.1f}s, timings={out2.get('timings')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
