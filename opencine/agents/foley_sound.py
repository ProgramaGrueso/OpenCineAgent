"""Agent 11: Foley & Sound Design Agent (Layer D).

MiniMax H3 renders diegetic sound together with the picture, driven by the prompt's
`overall_soundscape` field. This agent makes sure every shot describes its sound: a broadband
location ambience plus the concrete action sounds, so H3 does not fall back to a hum or silence.
"""

from __future__ import annotations

from opencine.agents.base import BaseAgent
from opencine.core.models import ProductionScript


class FoleySoundAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="FoleySoundAgent", layer="Layer D")

    async def execute(self, script: ProductionScript) -> ProductionScript:
        filled = 0
        for scene in script.scenes:
            ambience = f"Natural room tone and ambience of {scene.location.strip().rstrip('.').lower()}"
            for shot in scene.shots:
                if not shot.sound_description.strip():
                    shot.sound_description = f"{ambience}, with the sounds of the action in the shot"
                    filled += 1
        await self.emit("foley.described", {"filled": filled})
        return script
