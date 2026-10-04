"""Agent 1: Scriptwriter / Adapter Agent (Layer A).
Converts premises into technical scripts formatted with scenes, shots, dialogues, and emotions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import httpx
from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import (
    DialogueLine,
    ProductionScript,
    SceneScript,
    ShotSpecification,
    ShotType,
    CameraMotion
)


class ScriptwriterAgent(BaseAgent):
    """Generates structured cinematic screenplays using Ollama LLMs."""

    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__(name="ScriptwriterAgent", layer="Layer A")
        self.ollama_url = ollama_url or settings.ollama_url
        self.model = model or settings.llm_scriptwriter_model

    def _build_system_prompt(self, character_ids: List[str]) -> str:
        return (
            "You are an elite Hollywood technical director and screenwriter. "
            "Convert user premises into strict, production-ready cinematic scripts. "
            "Output MUST be valid JSON conforming to the requested schema. "
            f"Available character IDs: {', '.join(character_ids)}. "
            "Ensure dramatic tension, precise camera direction, and distinct character dialogue."
        )

    def _build_fallback_script(self, premise: str, character_ids: List[str]) -> ProductionScript:
        """High quality deterministic fallback if Ollama endpoint is offline."""
        c1 = character_ids[0] if len(character_ids) > 0 else "elena_vance"
        c2 = character_ids[1] if len(character_ids) > 1 else "victor_cross"

        return ProductionScript(
            title="The Protocol Incident",
            premise=premise,
            genre="Cyberpunk Noir Thriller",
            target_duration_sec=16.0,
            scenes=[
                SceneScript(
                    scene_id="SC_01",
                    title="The High Altitude Confrontation",
                    location="Skyline Penthouse Balcony",
                    time_of_day="night",
                    mood="tense and cinematic",
                    action_summary="Rain lashes against floor-to-ceiling glass. Neon reflections shimmer as two operatives face off.",
                    shots=[
                        ShotSpecification(
                            shot_id="SH_001",
                            scene_id="SC_01",
                            shot_number=1,
                            shot_type=ShotType.WIDE,
                            camera_motion=CameraMotion.DOLLY_IN,
                            action_description=f"Wide establishing shot. Rain streaks across panoramic glass. {c1} steps forward from the shadows into harsh blue rim light.",
                            stage_id="stage_penthouse_terrace",
                            character_ids=[c1],
                            dialogue=[
                                DialogueLine(
                                    speaker_id=c1,
                                    speaker_name="Elena Vance",
                                    text="You knew the encryption was already breached, Victor.",
                                    emotion="tense, cold certainty",
                                    start_offset=0.5,
                                    duration=3.0
                                )
                            ],
                            duration_sec=4.0
                        ),
                        ShotSpecification(
                            shot_id="SH_002",
                            scene_id="SC_01",
                            shot_number=2,
                            shot_type=ShotType.CLOSE_UP,
                            camera_motion=CameraMotion.STATIC,
                            action_description=f"Close up of {c2}. His platinum hair catches the magenta billboard glare as he adjusts his signet ring with unflinching composure.",
                            stage_id="stage_penthouse_terrace",
                            character_ids=[c2],
                            dialogue=[
                                DialogueLine(
                                    speaker_id=c2,
                                    speaker_name="Victor Cross",
                                    text="Breached? No Elena. It was designed to open.",
                                    emotion="calm, chillingly arrogant",
                                    start_offset=0.3,
                                    duration=3.2
                                )
                            ],
                            duration_sec=4.0
                        ),
                        ShotSpecification(
                            shot_id="SH_003",
                            scene_id="SC_01",
                            shot_number=3,
                            shot_type=ShotType.MEDIUM,
                            camera_motion=CameraMotion.PAN_LEFT,
                            action_description=f"Medium two-shot profile. {c1} raises her sidearm, tactical laser cutting through the mist as lightning flashes.",
                            stage_id="stage_penthouse_terrace",
                            character_ids=[c1, c2],
                            dialogue=[
                                DialogueLine(
                                    speaker_id=c1,
                                    speaker_name="Elena Vance",
                                    text="Then this ends right now.",
                                    emotion="determined, fierce breath",
                                    start_offset=0.4,
                                    duration=2.5
                                )
                            ],
                            duration_sec=4.0
                        ),
                        ShotSpecification(
                            shot_id="SH_004",
                            scene_id="SC_01",
                            shot_number=4,
                            shot_type=ShotType.EXTREME_CLOSE_UP,
                            camera_motion=CameraMotion.CRANE_UP,
                            action_description=f"Extreme close up on {c2}'s ocular cybernetic glint as thunder rumbles. A cold smirk forms.",
                            stage_id="stage_penthouse_terrace",
                            character_ids=[c2],
                            dialogue=[
                                DialogueLine(
                                    speaker_id=c2,
                                    speaker_name="Victor Cross",
                                    text="Nothing ever ends in this city.",
                                    emotion="whispering, sardonic",
                                    start_offset=0.2,
                                    duration=2.8
                                )
                            ],
                            duration_sec=4.0
                        )
                    ]
                )
            ]
        )

    async def execute(self, premise: str, character_ids: Optional[List[str]] = None) -> ProductionScript:
        """Call Ollama LLM to generate production script with scenes and shots."""
        active_cids = character_ids or ["elena_vance", "victor_cross"]

        prompt = (
            f"Premise: {premise}\n"
            f"Generate a screenplay with 1 scene containing exactly 3 to 4 sequential shots. "
            f"Characters to feature: {active_cids}.\n"
            "Format the output strictly as valid JSON matching this schema:\n"
            "{\n"
            '  "title": "Short Film Title",\n'
            '  "premise": "...",\n'
            '  "genre": "...",\n'
            '  "target_duration_sec": 16.0,\n'
            '  "scenes": [\n'
            "    {\n"
            '      "scene_id": "SC_01",\n'
            '      "title": "...",\n'
            '      "location": "...",\n'
            '      "time_of_day": "night",\n'
            '      "mood": "...",\n'
            '      "action_summary": "...",\n'
            '      "shots": [\n'
            "        {\n"
            '          "shot_id": "SH_001",\n'
            '          "shot_number": 1,\n'
            '          "shot_type": "wide",\n'
            '          "camera_motion": "dolly_in",\n'
            '          "action_description": "...",\n'
            '          "stage_id": "stage_main",\n'
            f'          "character_ids": ["{active_cids[0]}"],\n'
            '          "dialogue": [\n'
            "            {\n"
            f'              "speaker_id": "{active_cids[0]}",\n'
            '              "speaker_name": "...",\n'
            '              "text": "...",\n'
            '              "emotion": "...",\n'
            '              "start_offset": 0.5,\n'
            '              "duration": 2.5\n'
            "            }\n"
            "          ],\n"
            '          "duration_sec": 4.0\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Return ONLY raw JSON, no markdown codeblocks, no explanations."
        )

        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
                res = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self._build_system_prompt(active_cids)},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "format": "json"
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("message", {}).get("content", "").strip()
                    # Strip any potential markdown wrappers
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    parsed = json.loads(content)
                    script = ProductionScript(**parsed)
                    self.logger.info(f"Generated script '{script.title}' via Ollama ({self.model})")
                    return script
        except Exception as exc:
            self.logger.warning(f"Ollama script generation error: {exc}. Using deterministic high-fidelity fallback.")

        return self._build_fallback_script(premise, active_cids)
