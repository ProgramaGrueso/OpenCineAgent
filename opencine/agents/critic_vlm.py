"""Agent 7: Critic / Quality Control Agent (Layer C).

Measurable checks on every rendered clip:
  technical      container, frame count and audio stream match the shot spec
  motion         the clip is not frozen or black
  audio          speech-level audio exists when the shot has dialogue
  dialogue       faster-whisper transcript vs the script lines (optional dependency)
  identity       a local VLM (Ollama, e.g. qwen2.5vl) compares frames with the approved master
                 portraits and looks for deformed faces/hands and extra people
A check that cannot run is reported as SKIPPED, never as passed: "verified" means measured.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from opencine.agents.base import BaseAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core import media
from opencine.core.config import settings
from opencine.core.models import CheckStatus, HealingAction, QCCheck, QCReport, ShotSpecification

_whisper_model = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def dialogue_similarity(expected: str, heard: str) -> float:
    a, b = _normalize(expected), _normalize(heard)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


class CriticVlmAgent(BaseAgent):
    def __init__(self, registry_agent: Optional[AssetIdentityRegistryAgent] = None,
                 ollama_url: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__(name="CriticVlmAgent", layer="Layer C")
        self.registry = registry_agent or AssetIdentityRegistryAgent()
        self.ollama_url = ollama_url or settings.ollama_url
        self.model = model or settings.vlm_reviewer_model
        self._vlm_available: Optional[bool] = None

    # ------------------------------------------------------------ checks
    def check_technical(self, shot: ShotSpecification, clip: Path) -> QCCheck:
        info = media.probe(clip)
        need = shot.duration_sec * settings.fps
        problems = []
        if not info.get("has_video"):
            problems.append("no video stream")
        if info.get("frames", 0) + 1 < need:
            problems.append(f"{info.get('frames', 0):.0f} frames < {need:.0f} needed for {shot.duration_sec}s")
        if not info.get("has_audio"):
            problems.append("no audio stream")
        detail = "; ".join(problems) or f"{info.get('width', 0):.0f}x{info.get('height', 0):.0f}, " \
                                         f"{info.get('frames', 0):.0f} frames, audio ok"
        return QCCheck(name="technical", status=CheckStatus.FAILED if problems else CheckStatus.PASSED, detail=detail)

    def check_motion(self, clip: Path) -> QCCheck:
        frames = media.gray_frames(clip, count=12).astype(np.float32)
        if len(frames) < 2:
            return QCCheck(name="motion", status=CheckStatus.FAILED, detail="could not decode frames")
        brightness = float(frames.mean())
        change = float(np.abs(np.diff(frames, axis=0)).mean())
        if brightness < 6:
            return QCCheck(name="motion", status=CheckStatus.FAILED, detail=f"black clip (mean luma {brightness:.1f})")
        if change < 0.3:
            return QCCheck(name="motion", status=CheckStatus.FAILED, detail=f"frozen clip (frame change {change:.2f})")
        return QCCheck(name="motion", status=CheckStatus.PASSED, detail=f"frame change {change:.1f}")

    def check_audio(self, shot: ShotSpecification, clip: Path) -> QCCheck:
        level = media.rms_db(media.audio_samples(clip, end=shot.duration_sec))
        if shot.dialogue and level < -45:
            return QCCheck(name="audio", status=CheckStatus.FAILED, detail=f"near silent ({level:.0f} dBFS) with dialogue")
        return QCCheck(name="audio", status=CheckStatus.PASSED, detail=f"{level:.0f} dBFS RMS")

    async def check_dialogue(self, shot: ShotSpecification, clip: Path) -> tuple[QCCheck, str]:
        if not shot.dialogue:
            return QCCheck(name="dialogue", status=CheckStatus.PASSED, detail="no dialogue in shot"), ""
        try:
            transcript = await asyncio.to_thread(self._transcribe, clip, shot.duration_sec)
        except ImportError:
            return QCCheck(name="dialogue", status=CheckStatus.SKIPPED,
                           detail="faster-whisper not installed (pip install -e '.[qc]')"), ""
        expected = " ".join(line.text for line in shot.dialogue)
        score = dialogue_similarity(expected, transcript)
        status = CheckStatus.PASSED if score >= settings.dialogue_match_threshold else CheckStatus.FAILED
        return QCCheck(name="dialogue", status=status, detail=f"similarity {score:.2f}: heard '{transcript[:120]}'"), transcript

    @staticmethod
    def _transcribe(clip: Path, duration: float) -> str:
        global _whisper_model
        from faster_whisper import WhisperModel  # optional dependency
        if _whisper_model is None:
            _whisper_model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
        segments, _ = _whisper_model.transcribe(str(clip), vad_filter=True)
        return " ".join(s.text.strip() for s in segments if s.start < duration)

    async def _vlm_ready(self) -> bool:
        if self._vlm_available is None:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    tags = (await client.get(f"{self.ollama_url}/api/tags")).json().get("models", [])
                names = [m.get("name", "") for m in tags]
                self._vlm_available = any(n == self.model or n.startswith(self.model + ":") for n in names)
            except httpx.HTTPError:
                self._vlm_available = False
        return self._vlm_available

    async def check_identity(self, shot: ShotSpecification, frames: List[Path]) -> QCCheck:
        cast = self.registry.cast(shot.character_ids)
        if not await self._vlm_ready():
            return QCCheck(name="identity", status=CheckStatus.SKIPPED,
                           detail=f"VLM '{self.model}' not available in Ollama (ollama pull {self.model})")
        refs = [self.registry.reference_path(c) for c in cast]
        images = [base64.b64encode(p.read_bytes()).decode() for p in [*refs, *frames] if p]
        roster = "; ".join(f"reference image {i + 1} is {c.name}" for i, c in enumerate(cast)) or "no reference people"
        question = (
            f"The first {len(cast)} image(s) are reference portraits: {roster}. The remaining {len(frames)} images are "
            f"frames of one generated video shot. Expected on screen: {len(cast)} person(s).\n"
            "Answer ONLY JSON: {\"same_identity\": [true/false per reference person], \"deformed_face\": bool, "
            "\"deformed_hands\": bool, \"extra_people\": bool, \"notes\": \"short reason\"}. "
            "same_identity is false if the face in the frames is clearly a different person."
        )
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
                res = await client.post(f"{self.ollama_url}/api/chat", json={
                    "model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": question, "images": images}]})
            res.raise_for_status()
            verdict: Dict[str, Any] = json.loads(res.json()["message"]["content"])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            return QCCheck(name="identity", status=CheckStatus.SKIPPED, detail=f"VLM error: {exc}")
        same = verdict.get("same_identity", [])
        same = same if isinstance(same, list) else [same]
        failures = [cast[i].name for i, ok in enumerate(same[:len(cast)]) if ok is False]
        if verdict.get("deformed_face"):
            failures.append("deformed face")
        if verdict.get("deformed_hands"):
            failures.append("deformed hands")
        if verdict.get("extra_people"):
            failures.append("extra people")
        detail = f"{', '.join(failures) or 'consistent'} - {verdict.get('notes', '')}"[:300]
        return QCCheck(name="identity", status=CheckStatus.FAILED if failures else CheckStatus.PASSED, detail=detail)

    # ------------------------------------------------------------ main
    async def execute(self, shot: ShotSpecification, clip: Path, out_dir: Path, attempt: int = 1) -> QCReport:
        checks = [self.check_technical(shot, clip), self.check_motion(clip), self.check_audio(shot, clip)]
        dialogue_check, transcript = await self.check_dialogue(shot, clip)
        checks.append(dialogue_check)
        frames = await media.extract_frames(clip, out_dir / "qc" / f"{shot.shot_id}_t{attempt}", count=4)
        checks.append(await self.check_identity(shot, frames))

        failed = [c for c in checks if c.status == CheckStatus.FAILED]
        report = QCReport(
            shot_id=shot.shot_id, attempt=attempt, pass_qc=not failed, checks=checks,
            error_codes=[f"ERR_{c.name.upper()}" for c in failed],
            healing_action=HealingAction.RETRY_NEW_SEED if failed else HealingAction.NONE,
            transcript=transcript, inspected_frames=[str(p) for p in frames],
        )
        await self.emit("qc.approved" if report.pass_qc else "qc.rejected", report.model_dump(mode="json"))
        return report
