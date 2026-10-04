"""Agent 2: Shotlist & Prompt Engineer Agent (Layer A).

Compiles every shot into render-ready specs: the Krea 2 keyframe prompt, the MiniMax H3 prompt
(validated against the contract in opencine/core/prompts.py), the H3 frame length and a
deterministic seed. Invalid shots stop the production before any GPU time is spent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from opencine.agents.base import BaseAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.config import settings
from opencine.core.models import ProductionScript, SceneScript, ShotSpecification
from opencine.core.prompts import keyframe_prompt, validate_video_prompt, video_prompt
from opencine.core.seeds import stable_seed
from opencine.workflows import h3_length


class ShotlistError(ValueError):
    pass


class ShotlistPromptAgent(BaseAgent):
    def __init__(self, registry_agent: Optional[AssetIdentityRegistryAgent] = None) -> None:
        super().__init__(name="ShotlistPromptAgent", layer="Layer A")
        self.registry = registry_agent or AssetIdentityRegistryAgent()

    def location_line(self, scene: SceneScript) -> str:
        location = scene.location.strip().rstrip(".")
        return f"{location[:1].upper()}{location[1:]}, {scene.time_of_day}, {scene.mood} mood"

    def compile_shot(self, shot: ShotSpecification, scene: SceneScript, script: ProductionScript) -> ShotSpecification:
        cast = self.registry.cast(shot.character_ids)
        location = self.location_line(scene)
        shot.keyframe_prompt = keyframe_prompt(shot, cast, location, script.visual_style)
        shot.video_prompt = video_prompt(shot, cast, script.language, location, script.visual_style)
        shot.length_frames = h3_length(shot.duration_sec, settings.fps)
        shot.seed = script.seed if script.seed is not None else stable_seed(script.title, shot.shot_id)
        return shot

    async def execute(self, script: ProductionScript) -> ProductionScript:
        problems: Dict[str, List[str]] = {}
        for scene in script.scenes:
            for shot in scene.shots:
                if not shot.stage_id:
                    shot.stage_id = f"stage_{scene.scene_id.lower()}"
                self.compile_shot(shot, scene, script)
                issues = validate_video_prompt(shot.video_prompt)
                if len(shot.character_ids) > 3:
                    issues.append("more than three people on screen; render in layers or split the shot")
                if issues:
                    problems[shot.shot_id] = issues
        if problems:
            report = "\n".join(f"  {sid}: {'; '.join(v)}" for sid, v in problems.items())
            raise ShotlistError(f"Shots violate the H3 prompt contract:\n{report}")
        await self.emit("shotlist.compiled", {"shots": len(script.all_shots()), "scenes": len(script.scenes)})
        return script
