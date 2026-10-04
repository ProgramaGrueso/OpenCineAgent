"""Agent 4: Environment & Stage Agent (Layer B).

Locks one empty plate per location (Krea 2 t2i, no people) and reuses it as the scene image of
every keyframe in that location, so architecture and light stay consistent across angles.
A photo of a real location can replace the generated plate (SceneScript.location_image).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

from opencine.agents.base import BaseAgent
from opencine.core.comfy_client import ComfyUIClient, comfy_client
from opencine.core.config import settings
from opencine.core.models import EnvironmentStage, SceneScript
from opencine.core.prompts import stage_plate_prompt
from opencine.core.seeds import stable_seed
from opencine.workflows import krea2_text_to_image


class EnvironmentStageAgent(BaseAgent):
    def __init__(self, client: Optional[ComfyUIClient] = None) -> None:
        super().__init__(name="EnvironmentStageAgent", layer="Layer B")
        self.comfy = client or comfy_client
        self._stages: Dict[str, EnvironmentStage] = {}

    async def execute(self, stage_id: str, scene: SceneScript, out_dir: Path, style: str) -> EnvironmentStage:
        if stage_id in self._stages:
            return self._stages[stage_id]
        plate = out_dir / "stages" / f"{stage_id}.png"
        plate.parent.mkdir(parents=True, exist_ok=True)
        prompt = stage_plate_prompt(scene.location, scene.time_of_day, scene.mood, style)
        if not plate.exists():
            if scene.location_image:
                shutil.copy2(scene.location_image, plate)
                self.logger.info(f"Stage '{stage_id}' uses the location photo {scene.location_image}")
            else:
                wf = krea2_text_to_image(prompt, settings.keyframe_width, settings.keyframe_height,
                                         stable_seed(stage_id, "plate"), f"opencine/stages/{stage_id}")
                out = await self.comfy.run(wf, plate.parent / "raw", f"stage:{stage_id}", suffixes=(".png",))
                out[0].replace(plate)
        stage = EnvironmentStage(stage_id=stage_id, name=scene.title, base_prompt=prompt,
                                 locked_background_image=str(plate))
        self._stages[stage_id] = stage
        await self.emit("environment.locked", {"stage_id": stage_id, "plate": str(plate)})
        return stage
