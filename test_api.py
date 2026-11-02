#!/usr/bin/env python3
"""One-image smoke test against a deployed endpoint (or a local `runpod` test server).

  pip install requests
  export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
  python test_api.py photo.jpg "make the background a clean studio backdrop" --upscale 2
"""

import argparse
import base64
import os
import sys
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("prompt")
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--upscale-model")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out_test.png")
    ap.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    ap.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    ap.add_argument("--api-base", default=os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2"))
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    if not args.endpoint_id or not args.api_key:
        ap.error("set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY")

    payload = {
        "prompt": args.prompt,
        "image_base64": base64.b64encode(open(args.image, "rb").read()).decode("ascii"),
        "upscale": args.upscale,
        "seed": args.seed,
    }
    if args.upscale_model:
        payload["upscale_model"] = args.upscale_model
    if args.steps:
        payload["steps"] = args.steps

    base = f"{args.api_base.rstrip('/')}/{args.endpoint_id}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    t0 = time.time()
    job = requests.post(f"{base}/run", json={"input": payload}, headers=headers, timeout=120).json()
    job_id = job.get("id")
    if not job_id:
        print(f"no job id: {job}", file=sys.stderr)
        return 1
    print(f"job {job_id} submitted")

    while time.time() - t0 < args.timeout:
        time.sleep(2)
        st = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=120).json()
        status = st.get("status")
        print(f"  {time.time() - t0:5.0f}s  {status}")
        if status == "COMPLETED":
            out = st.get("output") or {}
            if out.get("error"):
                print(f"worker error: {out['error']}", file=sys.stderr)
                return 1
            raw = base64.b64decode(out["image"].split(",", 1)[-1])
            open(args.out, "wb").write(raw)
            print(f"{out.get('width')}x{out.get('height')} -> {args.out}  "
                  f"timings {out.get('timings')}")
            return 0
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"job {status}: {st.get('error') or st.get('output')}", file=sys.stderr)
            return 1
    print("timed out waiting for the job", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
