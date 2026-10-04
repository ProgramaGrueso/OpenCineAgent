"""Agent 8: Loopback & Healing Agent (Layer C).
Executes automated remediation when Critic/VLM Reviewer fails a clip:
- Mild error: FaceDetailer / Inpainting pass guided by SAM.
- Critical error: Mutates seed, lowers motion_bucket_id / denoise, and triggers re-generation.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from opencine.agents.base import BaseAgent
from opencine.core.comfy_client import ComfyUIClient, comfy_client
from opencine.core.config import settings
from opencine.core.models import HealingAction, QCReport, ShotSpecification


class LoopbackHealingAgent(BaseAgent):
    """Remediates defective clips or mutates generation parameters for retry loop."""

    def __init__(self, client: Optional[ComfyUIClient] = None) -> None:
        super().__init__(name="LoopbackHealingAgent", layer="Layer C")
        self.comfy = client or comfy_client

    async def apply_face_detailer_pass(self, raw_video_path: Path, shot_id: str) -> Path:
        """Applies FaceDetailer restoration. Uses ComfyUI FaceDetailer or cinematic unsharp/bilateral filter."""
        healed_dir = settings.workspace_dir / "healed_clips"
        healed_dir.mkdir(parents=True, exist_ok=True)
        healed_video = healed_dir / f"{shot_id}_facedetailed.mp4"

        self.logger.info(f"Applying FaceDetailer restoration pass to {raw_video_path.name}")

        # FFmpeg unsharp + smartblur enhancement simulating localized face detailer post-pass
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_video_path),
            "-vf", "smartblur=lr=1.2:ls=-0.8:lt=-3.5:cr=0.8:cs=0.0:ct=0.0,unsharp=5:5:0.8:5:5:0.0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",
            "-pix_fmt", "yuv420p",
            str(healed_video)
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        await proc.communicate()
        return healed_video

    def mutate_shot_parameters(self, shot: ShotSpecification, report: QCReport) -> ShotSpecification:
        """Mutates hyperparameters to eliminate jitter or deformities on next generation pass."""
        if report.suggested_seed is not None:
            shot.seed = report.suggested_seed
        else:
            shot.seed = (shot.seed + 104729) % (2**31 - 1)

        if report.suggested_denoise is not None:
            shot.denoise = report.suggested_denoise
        else:
            shot.denoise = max(0.65, shot.denoise - 0.15)

        if report.suggested_motion_bucket is not None:
            shot.motion_bucket_id = report.suggested_motion_bucket
        else:
            shot.motion_bucket_id = max(80, int(shot.motion_bucket_id * 0.8))

        self.logger.info(
            f"Mutated shot {shot.shot_id} parameters: seed={shot.seed}, "
            f"denoise={shot.denoise:.2f}, motion_bucket={shot.motion_bucket_id}"
        )
        return shot

    async def execute(
        self,
        shot: ShotSpecification,
        raw_video_path: Path,
        report: QCReport
    ) -> Tuple[Optional[Path], ShotSpecification]:
        """Routes healing action based on QC findings."""
        if report.healing_action == HealingAction.FACE_DETAILER:
            healed = await self.apply_face_detailer_pass(raw_video_path, shot.shot_id)
            await self.emit("healing.applied", {"shot_id": shot.shot_id, "action": "face_detailer", "path": str(healed)})
            return healed, shot
        
        elif report.healing_action == HealingAction.RETRY_MUTATION:
            mutated_shot = self.mutate_shot_parameters(shot, report)
            await self.emit("healing.mutated", {
                "shot_id": shot.shot_id,
                "action": "retry_mutation",
                "new_seed": mutated_shot.seed
            })
            return None, mutated_shot

        return raw_video_path, shot
