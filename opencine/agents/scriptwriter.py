"""Agent 1: Scriptwriter / Adapter Agent (Layer A).

Turns a premise into a shot-by-shot screenplay with a local LLM (Ollama), or loads a script
written by a human (JSON). There is no canned fallback: a script that ignores the premise is
worse than an error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import ValidationError

from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import CharacterProfile, ProductionScript
from opencine.core.prompts import MAX_LINE_CHARS

SCHEMA_EXAMPLE = {
    "title": "Short title",
    "premise": "...",
    "genre": "...",
    "scenes": [{
        "scene_id": "SC_01",
        "title": "...",
        "location": "the empty location in detail: place, era, materials, lighting",
        "time_of_day": "night",
        "mood": "tense",
        "action_summary": "...",
        "music_description": "score direction, e.g. 'low cello drone, slow pulse'",
        "shots": [{
            "shot_id": "SH_001",
            "shot_number": 1,
            "shot_type": "wide | medium | close_up | extreme_close_up | extreme_wide",
            "camera_motion": "static | dolly_in | dolly_out | pan_left | pan_right | tracking_shot | handheld",
            "action_description": "present tense, what happens in order, physical and visible",
            "stage_id": "stage_sc01",
            "character_ids": ["<id>"],
            "dialogue": [{"speaker_id": "<id>", "speaker_name": "...", "text": "...",
                          "emotion": "delivery direction", "start_offset": 0.5, "duration": 3.0}],
            "sound_description": "diegetic sound: rain on glass, distant traffic",
            "duration_sec": 6.0,
        }],
    }],
}


class ScriptGenerationError(RuntimeError):
    pass


class ScriptwriterAgent(BaseAgent):
    def __init__(self, ollama_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__(name="ScriptwriterAgent", layer="Layer A")
        self.ollama_url = ollama_url or settings.ollama_url
        self.model = model or settings.llm_scriptwriter_model

    @staticmethod
    def load_script(path: Path) -> ProductionScript:
        return ProductionScript(**json.loads(Path(path).read_text(encoding="utf-8")))

    def _system_prompt(self, cast: List[CharacterProfile], language: str, shots: int) -> str:
        roster = "\n".join(f"- {c.id}: {c.name}, {c.description}" for c in cast)
        return (
            "You are a film director writing a shooting script for an AI video model that renders one clip "
            "per shot (5 to 15 seconds, with native dialogue audio).\n"
            f"Cast (use ONLY these ids):\n{roster}\n"
            "Rules:\n"
            f"- About {shots} shots in total, each 4 to 15 seconds (duration_sec).\n"
            "- At most ONE character speaks per shot; split exchanges into shot / reverse shot.\n"
            f"- Each line at most {MAX_LINE_CHARS} characters, at most two lines per shot. "
            f"Dialogue text is written in {language}.\n"
            "- A speaking character faces the camera and is not eating or covering the mouth.\n"
            "- action_description is concrete and visible, in present tense; no camera jargon numbers.\n"
            "- character_ids lists who is on screen, the speaker first.\n"
            "- Every scene has one location reused by all its shots (stage_id).\n"
            "Return ONLY JSON matching this example structure:\n" + json.dumps(SCHEMA_EXAMPLE, indent=1)
        )

    async def _ask(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            res = await client.post(f"{self.ollama_url}/api/chat", json={
                "model": self.model, "stream": False, "format": "json",
                "options": {"temperature": 0.7},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            })
        res.raise_for_status()
        content = res.json().get("message", {}).get("content", "")
        return re.sub(r"^```(?:json)?|```$", "", content.strip()).strip()

    async def execute(self, premise: str, cast: List[CharacterProfile], language: Optional[str] = None,
                      shots: int = 4) -> ProductionScript:
        language = language or settings.dialogue_language
        system = self._system_prompt(cast, language, shots)
        user = f"Premise: {premise}"
        allowed = {c.id for c in cast}
        last_error = ""
        for attempt in range(2):
            try:
                raw = await self._ask(system, user if not last_error else
                                      f"{user}\n\nYour previous answer was invalid: {last_error}\nFix it.")
                script = ProductionScript(**json.loads(raw))
            except httpx.HTTPError as exc:
                raise ScriptGenerationError(
                    f"Ollama ({self.ollama_url}, model {self.model}) is not available: {exc}. "
                    "Start Ollama or pass --script with a hand-written script."
                ) from exc
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:800]
                continue
            unknown = {cid for shot in script.all_shots() for cid in shot.character_ids} - allowed
            if not script.all_shots():
                last_error = "the script has no shots"
            elif unknown:
                last_error = f"unknown character ids {sorted(unknown)}; allowed: {sorted(allowed)}"
            else:
                script.premise = script.premise or premise
                script.language = language
                self.logger.info(f"Script '{script.title}': {len(script.all_shots())} shots via {self.model}")
                await self.emit("script.generated", {"title": script.title, "shots": len(script.all_shots())})
                return script
        raise ScriptGenerationError(f"The LLM did not return a valid script after 2 attempts: {last_error}")
