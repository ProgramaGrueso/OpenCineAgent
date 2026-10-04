"""Agent 9: Super-Resolution Agent (Layer C).

Upscales approved clips to the master resolution and keeps H3's native audio
(an upscaler that writes video only must not drop the dialogue).
  lanczos       ffmpeg scale, no extra dependencies
  clearreality  4x-ClearRealityV1 through spandrel, streamed in batches (opencine/tools/upscale_video.py);
                needs a python with torch + spandrel (OPENCINE_UPSCALE_PYTHON, e.g. ComfyUI's venv)
Frame rate stays at 24 fps: blended frame interpolation produces ghosting.
"""

from __future__ import annotations

import sys
from pathlib import Path

from opencine.agents.base import BaseAgent
from opencine.core import media
from opencine.core.comfy_client import comfy_client
from opencine.core.config import settings

TOOL = Path(__file__).resolve().parent.parent / "tools" / "upscale_video.py"


class SuperResolutionAgent(BaseAgent):
    def __init__(self, mode: str | None = None) -> None:
        super().__init__(name="SuperResolutionAgent", layer="Layer C")
        self.mode = mode or settings.upscale_mode

    async def execute(self, clip: Path, out_dir: Path, shot_id: str) -> Path:
        if self.mode == "none":
            return clip
        w, h = settings.upscale_width, settings.upscale_height
        target = out_dir / "masters" / f"{shot_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == "clearreality":
            if not settings.upscale_model_path:
                raise ValueError("OPENCINE_UPSCALE_MODEL_PATH must point to 4x-ClearRealityV1.pth")
            await comfy_client.free()  # the upscaler needs the VRAM ComfyUI is holding
            silent = target.with_suffix(".video.mp4")
            await media.run_cmd([settings.upscale_python or sys.executable, TOOL, clip, silent,
                                 "--model", settings.upscale_model_path, "--width", w, "--height", h,
                                 "--fps", settings.fps])
            await media.run_cmd(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", clip, "-map", "0:v:0",
                                 "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", target])
            silent.unlink(missing_ok=True)
        elif self.mode == "lanczos":
            await media.run_cmd(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf",
                                 f"scale={w}:{h}:flags=lanczos,unsharp=3:3:0.4", "-c:v", "libx264", "-crf", "16",
                                 "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", target])
        else:
            raise ValueError(f"Unknown upscale mode '{self.mode}' (none | lanczos | clearreality)")

        await self.emit("superres.done", {"shot_id": shot_id, "path": str(target), "mode": self.mode})
        return target
