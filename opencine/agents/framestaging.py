"""Agent 5: Frame Staging Agent (Keyframes) (Layer B).

Builds the first frame of each shot, which H3 receives as <Picture 1>:
  * people in shot  -> Krea 2 Identity Edit: scene = locked stage plate, subject = approved master
    portrait. Two or three people are added one edit at a time ("keep everyone already in the
    image"), which avoids the merged faces a single multi-person prompt produces.
  * nobody in shot  -> Krea 2 text-to-image of the shot.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from opencine.agents.base import BaseAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.comfy_client import ComfyUIClient, comfy_client
from opencine.core.config import settings
from opencine.core.models import EnvironmentStage, ShotSpecification
from opencine.core.prompts import keyframe_prompt
from opencine.workflows import krea2_identity_edit, krea2_text_to_image


class FrameStagingAgent(BaseAgent):
    def __init__(self, registry_agent: Optional[AssetIdentityRegistryAgent] = None,
                 client: Optional[ComfyUIClient] = None) -> None:
        super().__init__(name="FrameStagingAgent", layer="Layer B")
        self.registry = registry_agent or AssetIdentityRegistryAgent()
        self.comfy = client or comfy_client

    async def execute(self, shot: ShotSpecification, stage: EnvironmentStage, out_dir: Path,
                      location: str, style: str, force: bool = False) -> Path:
        target = out_dir / "keyframes" / f"{shot.shot_id}.png"
        if target.exists() and not force:
            return target
        work = out_dir / "keyframes" / "raw"
        w, h = settings.keyframe_width, settings.keyframe_height
        cast = self.registry.cast(shot.character_ids)

        # Identical inputs give an identical image: reuse it instead of rendering again.
        key = hashlib.sha1(json.dumps([shot.keyframe_prompt, shot.seed, stage.locked_background_image,
                                       [str(self.registry.reference_path(c)) for c in cast], w, h,
                                       settings.krea2_unet]).encode()).hexdigest()[:16]
        cached = out_dir / "keyframes" / "cache" / f"{key}.png"
        if cached.exists() and not force:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target)
            await self.emit("keyframe.reused", {"shot_id": shot.shot_id, "path": str(target)})
            return target

        if not cast:
            wf = krea2_text_to_image(shot.keyframe_prompt, w, h, shot.seed, f"opencine/kf/{shot.shot_id}")
            current = (await self.comfy.run(wf, work, f"keyframe:{shot.shot_id}", suffixes=(".png",)))[0]
        else:
            current = Path(stage.locked_background_image or "")
            if not current.exists():
                raise FileNotFoundError(f"Stage plate missing for {shot.stage_id}: {current}")
            for step, person in enumerate(cast):
                portrait = self.registry.reference_path(person)
                if not portrait or not portrait.exists():
                    raise FileNotFoundError(f"No master portrait for '{person.id}'")
                prompt = shot.keyframe_prompt if step == 0 else keyframe_prompt(shot, cast, location, style, step)
                wf = krea2_identity_edit(prompt, await self.comfy.upload(current), await self.comfy.upload(portrait),
                                         w, h, shot.seed + step, f"opencine/kf/{shot.shot_id}_{step}")
                label = f"keyframe:{shot.shot_id}:{person.id}"
                current = (await self.comfy.run(wf, work, label, suffixes=(".png",)))[0]

        target.parent.mkdir(parents=True, exist_ok=True)
        current.replace(target)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, cached)
        await self.emit("keyframe.generated", {"shot_id": shot.shot_id, "path": str(target)})
        return target
