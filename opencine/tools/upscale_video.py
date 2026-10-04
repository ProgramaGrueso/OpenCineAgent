#!/usr/bin/env python3
"""Streams a video through a spandrel upscaler (e.g. 4x-ClearRealityV1) in batches and area-downsamples
to the target size (supersampling: sharper than lanczos, no ringing). Writes video only; the caller
muxes the original audio back.

Runs with any python that has torch + spandrel + numpy (ComfyUI's venv works):
  python upscale_video.py in.mp4 out.mp4 --model 4x-ClearRealityV1.pth --width 1920 --height 1080
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def probe_size(path: Path):
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"])


def main() -> None:
    import numpy as np
    import torch
    from spandrel import ModelLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    in_w, in_h = probe_size(a.src)
    model = ModelLoader().load_from_file(str(a.model)).model.cuda().half().eval()
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
                            "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", str(a.dst)],
                           stdin=subprocess.PIPE)
    frame_bytes = in_w * in_h * 3
    total, t0 = 0, time.time()
    try:
        while True:
            batch = []
            for _ in range(a.batch):
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                batch.append(np.frombuffer(raw, np.uint8).reshape(in_h, in_w, 3))
            if not batch:
                break
            x = torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).half() / 255.0
            with torch.inference_mode():
                y = model(x).clamp(0, 1)
                y = torch.nn.functional.interpolate(y.float(), size=(a.height, a.width), mode="area")
            for frame in (y.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy():
                enc.stdin.write(frame.tobytes())
            total += len(batch)
    finally:
        enc.stdin.close()
        enc.wait()
        dec.wait()
    if enc.returncode != 0 or total == 0:
        sys.exit(f"upscale failed after {total} frames")
    print(f"{total} frames in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
