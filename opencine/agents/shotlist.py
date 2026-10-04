"""Agent 2: Shotlist & Prompt Engineer Agent (Layer A).
Translates shot descriptions into structured video prompts, camera syntax (Wan 2.1/Hunyuan/AnimateDiff),
and deterministic generation parameters.
"""

from __future__ import annotations

from typing import List, Optional
from opencine.agents.base import BaseAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.models import (
    CameraMotion,
    ProductionScript,
    ShotSpecification,
    ShotType
)


class ShotlistPromptAgent(BaseAgent):
    """Refines shot definitions with cinematic video generation prompts and parameters."""

    def __init__(self, registry_agent: Optional[AssetIdentityRegistryAgent] = None) -> None:
        super().__init__(name="ShotlistPromptAgent", layer="Layer A")
        self.registry = registry_agent or AssetIdentityRegistryAgent()

    def _compile_camera_directive(self, motion: CameraMotion, shot_type: ShotType) -> str:
        """Translate motion enum into specialized video model camera prompt keywords."""
        motion_map = {
            CameraMotion.STATIC: "locked-off tripod shot, subtle natural breathing movement",
            CameraMotion.PAN_LEFT: "smooth cinematic horizontal pan from right to left, fluid dolly track",
            CameraMotion.PAN_RIGHT: "cinematic smooth pan from left to right, steadycam sweep",
            CameraMotion.TILT_UP: "vertical tilt upwards from low angle to eye-level, heroic perspective",
            CameraMotion.TILT_DOWN: "vertical tilt downwards, looking down onto subject, atmospheric tension",
            CameraMotion.DOLLY_IN: "slow controlled dolly push-in towards subject, intensifying intimacy, 35mm lens",
            CameraMotion.DOLLY_OUT: "slow cinematic dolly pull-out, revealing surrounding scale and environment",
            CameraMotion.CRANE_UP: "sweeping crane boom moving upwards, dramatic vertical perspective",
            CameraMotion.ORBIT_360: "cinematic circular orbit around subject, 360-degree parallax shift",
            CameraMotion.TRACKING: "dynamic tracking shot moving parallel with subject motion, steadycam"
        }

        framing_map = {
            ShotType.EXTREME_WIDE: "extreme wide establishing shot, vast architectural scale, deep depth of field",
            ShotType.WIDE: "wide cinematic shot, full body visible in environment, atmospheric haze",
            ShotType.MEDIUM: "medium shot from waist up, natural conversational framing, soft background bokeh",
            ShotType.CLOSE_UP: "intimate cinematic close-up, sharp eye focus, shallow depth of field, f/1.8",
            ShotType.EXTREME_CLOSE_UP: "extreme macro close-up, hyper-detailed skin texture, iris reflection, 85mm"
        }

        return f"{framing_map.get(shot_type, 'cinematic shot')}, {motion_map.get(motion, 'static shot')}"

    def enhance_shot(self, shot: ShotSpecification) -> ShotSpecification:
        """Injects identity descriptors, camera grammar, and model hyperparameters."""
        # 1. Identity Reinforcement
        char_prompts = self.registry.compile_character_prompt_enhancement(shot.character_ids)
        
        # 2. Camera syntax
        cam_directive = self._compile_camera_directive(shot.camera_motion, shot.shot_type)

        # 3. Model Engine tags (Wan 2.1 / Hunyuan cinematic lighting)
        cinematic_style = (
            "photorealistic 35mm cinema film still, Arri Alexa Mini LF, Master Anamorphic lenses, "
            "volumetric fog, subtle film grain, raytraced reflections, highly detailed, masterwork"
        )

        full_positive = (
            f"{shot.action_description}. {char_prompts} {cam_directive}. {cinematic_style}."
        ).strip()

        negative = (
            "blurry, cartoon, 3d render, deformed hands, extra fingers, mutated anatomy, disfigured face, "
            "low quality, artifacts, watermark, signature, flicker, jitter, plastic skin, oversaturated"
        )

        # 4. Set deterministic parameters
        shot.positive_prompt = full_positive
        shot.negative_prompt = negative
        
        # Deterministic seed hashing based on shot ID to allow reproducible mutations
        base_seed = abs(hash(shot.shot_id)) % (2**31 - 1)
        shot.seed = base_seed
        shot.motion_bucket_id = 127
        shot.denoise = 1.0

        return shot

    async def execute(self, script: ProductionScript) -> ProductionScript:
        """Process all shots in the script, compiling video prompt specifications."""
        total_shots = 0
        for scene in script.scenes:
            enhanced_shots: List[ShotSpecification] = []
            for shot in scene.shots:
                enhanced = self.enhance_shot(shot)
                enhanced_shots.append(enhanced)
                total_shots += 1
            scene.shots = enhanced_shots

        await self.emit("shotlist.compiled", {"total_shots": total_shots, "scenes": len(script.scenes)})
        return script
