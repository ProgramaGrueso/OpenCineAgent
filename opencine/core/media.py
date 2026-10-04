"""FFmpeg / FFprobe helpers shared by QC, mixing and assembly.

Every command raises MediaError on failure: a silent fallback that produces a
"successful" but wrong file is worse than a crash.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class MediaError(RuntimeError):
    pass


async def run_cmd(cmd: Sequence[str]) -> str:
    proc = await asyncio.create_subprocess_exec(
        *[str(c) for c in cmd], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise MediaError(f"{Path(str(cmd[0])).name} failed ({proc.returncode}): {err.decode(errors='replace')[-1200:]}")
    return out.decode(errors="replace") + err.decode(errors="replace")


def probe(path: Path) -> Dict[str, float]:
    """Duration, size, fps, frame count and audio presence of a media file."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise MediaError(f"ffprobe could not read {path}: {res.stderr[-400:]}")
    info = json.loads(res.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    duration = float(info.get("format", {}).get("duration", 0.0))
    out: Dict[str, float] = {"duration": duration, "has_audio": float(audio is not None),
                             "has_video": float(video is not None)}
    if video:
        num, den = (video.get("r_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        frames = video.get("nb_frames")
        out.update(width=float(video["width"]), height=float(video["height"]), fps=fps,
                   frames=float(frames) if frames and frames.isdigit() else round(duration * fps))
    return out


def gray_frames(video: Path, count: int = 12, size: int = 96) -> np.ndarray:
    """`count` evenly spaced grayscale thumbnails as an array (count, size, size)."""
    dur = probe(video)["duration"]
    frames = []
    for i in range(count):
        t = dur * (i + 0.5) / count
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
             "-vf", f"scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
            capture_output=True,
        ).stdout
        if len(raw) == size * size:
            frames.append(np.frombuffer(raw, np.uint8).reshape(size, size))
    return np.stack(frames) if frames else np.zeros((0, size, size), np.uint8)


async def extract_frames(video: Path, out_dir: Path, count: int = 4, width: int = 768) -> List[Path]:
    """Evenly spaced PNG frames (avoids the first/last 5% where H3 still sticks to the reference)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe(video)["duration"]
    paths = []
    for i in range(count):
        t = dur * (0.05 + 0.9 * (i + 0.5) / count)
        p = out_dir / f"{video.stem}_f{i:02d}.png"
        await run_cmd(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video,
                       "-frames:v", "1", "-vf", f"scale={width}:-2", p])
        paths.append(p)
    return paths


def audio_samples(path: Path, rate: int = 16000, start: float = 0.0, end: Optional[float] = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(0.01, end - start):.3f}"]
    cmd += ["-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return 20 * math.log10(rms) if rms > 0 else -120.0


async def loudnorm_two_pass(src: Path, dst: Path, target_lufs: float = -14.0, true_peak: float = -1.5) -> Path:
    """EBU R128 normalization measured first and applied linearly (single-pass loudnorm pumps ambience)."""
    measure = await run_cmd(["ffmpeg", "-hide_banner", "-i", src, "-af",
                             f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:print_format=json", "-f", "null", "-"])
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", measure, re.S)
    if not match:
        raise MediaError("loudnorm measurement returned no statistics")
    m = json.loads(match.group(0))
    af = (f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:measured_I={m['input_i']}:"
          f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
          f"offset={m['target_offset']}:linear=true")
    await run_cmd(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", "48000", dst])
    return dst
