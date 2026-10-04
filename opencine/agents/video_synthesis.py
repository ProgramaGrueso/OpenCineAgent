"""Agent 6: Video Synthesis Agent (Layer B).

Renders each shot with MiniMax H3 reference-to-video: picture, voice and diegetic sound in one pass.
  <Picture 1> keyframe, <Picture 2..> the cast's master portraits (identity anchors, same in every shot)
  RefMod only when exactly one face is on screen
  voice_track (pre-recorded lines) switches to lipsync mode; otherwise H3 speaks the <d> lines itself
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from opencine.agents.base import BaseAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.comfy_client import ComfyUIClient, comfy_client
from opencine.core.config import settings
from opencine.core.models import ShotSpecification
from opencine.workflows import h3_reference_to_video


class VideoSynthesisAgent(BaseAgent):
    def __init__(self, registry_agent: Optional[AssetIdentityRegistryAgent] = None,
                 client: Optional[ComfyUIClient] = None) -> None:
        super().__init__(name="VideoSynthesisAgent", layer="Layer B")
        self.registry = registry_agent or AssetIdentityRegistryAgent()
        self.comfy = client or comfy_client

    async def execute(self, shot: ShotSpecification, keyframe: Path, out_dir: Path, attempt: int = 1) -> Path:
        cast = self.registry.cast(shot.character_ids)
        refs = [await self.comfy.upload(keyframe)]
        for person in cast:
            refs.append(await self.comfy.upload(self.registry.reference_path(person)))
        refmod = cast[0].refmod_name if len(cast) == 1 else None
        voice = await self.comfy.upload(Path(shot.voice_track)) if shot.voice_track else None

        wf = h3_reference_to_video(
            prompt=shot.video_prompt, ref_images=refs, width=settings.video_width, height=settings.video_height,
            length=shot.length_frames, seed=shot.seed, prefix=f"opencine/clips/{shot.shot_id}",
            refmod=refmod, voice_track=voice,
        )
        clips = out_dir / "clips"
        out = await self.comfy.run(wf, clips / "raw", f"video:{shot.shot_id}:t{attempt}", suffixes=(".mp4",))
        target = clips / f"{shot.shot_id}_t{attempt}_s{shot.seed}.mp4"
        out[0].replace(target)
        await self.emit("video.synthesized", {"shot_id": shot.shot_id, "path": str(target), "seed": shot.seed,
                                              "frames": shot.length_frames, "refmod": refmod,
                                              "lipsync": bool(voice)})
        return target
