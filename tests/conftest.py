"""Test fixtures.

FakeComfyClient stands in for a ComfyUI server in tests only: it inspects each workflow and writes a
real file with ffmpeg (an image for SaveImage, a moving clip with audio for SaveVideo, a tone for
SaveAudio). This exercises orchestration, QC, cutting and mixing end to end without a GPU.
Production code has no simulated path.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from opencine.core.config import settings

ROOT = Path(__file__).resolve().parent.parent


class FakeComfyClient:
    def __init__(self) -> None:
        self.jobs: List[Dict[str, Any]] = []
        self.labels: List[str] = []
        self.freed = 0
        self.uploads: List[str] = []
        self.models = {"refmods": set()}

    async def upload(self, path: Path, subfolder: str = "") -> str:
        name = f"fake_{Path(path).name}"
        self.uploads.append(str(path))
        return name

    async def free(self) -> None:
        self.freed += 1

    async def is_alive(self) -> bool:
        return True

    async def check_system(self, required_nodes=()):
        return {"alive": True, "node_count": 2000, "missing_nodes": [], "partial": False}

    async def available(self, node: str, field: str) -> List[str]:
        if node == "MiniMaxH3RefModsLoader":
            return ["(none)", *self.models["refmods"]]
        return [settings.krea2_unet, settings.krea2_clip, settings.krea2_vae, settings.krea2_identity_edit_lora,
                settings.h3_unet, settings.h3_clip, settings.h3_video_vae, settings.h3_audio_vae,
                settings.h3_turbo_lora or "", settings.music3_unet, settings.music3_clip, settings.music3_vae]

    async def object_info(self, refresh: bool = False) -> Dict[str, Any]:
        return {}

    async def run(self, workflow, dest_dir: Path, label: str, timeout=None, suffixes=(), expect_files=True):
        self.jobs.append(workflow)
        self.labels.append(label)
        dest_dir.mkdir(parents=True, exist_ok=True)
        classes = {n["class_type"]: n for n in workflow.values()}
        n = len(self.jobs)
        if "SaveVideo" in classes:
            r2v = classes["MiniMaxH3ReferenceToVideo"]["inputs"]
            seconds = r2v["length"] / settings.fps
            out = dest_dir / f"clip_{n:03d}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"testsrc2=s={r2v['width']}x{r2v['height']}:r={settings.fps}:d={seconds}",
                            "-f", "lavfi", "-i", f"sine=f=330:d={seconds}", "-shortest",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out], check=True)
            return [out]
        if "SaveImage" in classes:
            latent = next(v for k, v in classes.items() if "Latent" in k and "Upscale" not in k)["inputs"]
            scale = 2 if "LatentUpscaleBy" in classes else 1
            out = dest_dir / f"image_{n:03d}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"testsrc2=s={latent['width'] * scale}x{latent['height'] * scale}",
                            "-frames:v", "1", out], check=True)
            return [out]
        if "SaveAudio" in classes:
            seconds = classes["EmptyMiniMaxMusic3LatentAudio"]["inputs"]["seconds"]
            out = dest_dir / f"music_{n:03d}.flac"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=110:d={seconds}", out],
                           check=True)
            return [out]
        if "MiniMaxH3RefModExtract" in classes:
            self.models["refmods"].add(classes["MiniMaxH3RefModExtract"]["inputs"]["name"])
            return []
        raise AssertionError(f"FakeComfyClient got an unknown workflow: {sorted(classes)}")


@pytest.fixture(autouse=True)
def test_settings(tmp_path, monkeypatch):
    """Small, fast, deterministic settings; ignores the developer's .env."""
    overrides = dict(
        workspace_dir=tmp_path / "workspace", video_width=320, video_height=180, keyframe_width=320,
        keyframe_height=180, fps=24, upscale_mode="none", music_mode="none", h3_sla=False,
        h3_turbo_lora="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        h3_unet="minimax_h3_ref2va_pruned_int8_convrot.safetensors", krea2_hires_pass=False,
        krea2_unet="krea2_turbo_fp8_scaled.safetensors", max_qc_retries=2, dialogue_language="English",
        vlm_reviewer_model="opencine-test-no-such-vlm", ollama_url="http://127.0.0.1:9",
    )
    for key, value in overrides.items():
        monkeypatch.setattr(settings, key, value)
    return settings


@pytest.fixture
def fake_comfy() -> FakeComfyClient:
    return FakeComfyClient()


@pytest.fixture
def registry_file(tmp_path) -> Path:
    """A copy of the shipped cast with no portraits approved (independent of local state)."""
    target = tmp_path / "characters" / "registry.json"
    target.parent.mkdir(parents=True)
    cast = json.loads((ROOT / "config" / "characters" / "registry.json").read_text(encoding="utf-8"))
    for person in cast:
        person.pop("reference_image", None)
        person.pop("refmod_name", None)
        person["approved"] = False
    target.write_text(json.dumps(cast), encoding="utf-8")
    return target


@pytest.fixture
def example_script() -> Path:
    return ROOT / "examples" / "rooftop_confrontation.json"
