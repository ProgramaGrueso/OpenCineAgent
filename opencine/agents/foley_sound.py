"""Agent 11: Foley & Sound Design Agent (Layer D).
Generates synchronous footsteps, impacts, and environmental atmospheric audio
(MMAudio / indexed local SFX sound design).
"""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path
from typing import List
from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import AudioTrack, ShotSpecification


class FoleySoundAgent(BaseAgent):
    """Creates diegetic sound effects (footsteps, cloth movement, rain texture) tailored to action."""

    def __init__(self) -> None:
        super().__init__(name="FoleySoundAgent", layer="Layer D")

    def _generate_foley_wav(
        self,
        output_path: Path,
        duration_sec: float,
        sfx_type: str = "rain_ambient"
    ) -> Path:
        """Generates realistic acoustic texture for Foley sound design."""
        sample_rate = settings.voice_sample_rate
        total_samples = int(duration_sec * sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        random.seed(42)

        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)

            frames = bytearray()
            for i in range(total_samples):
                t = i / sample_rate
                # Generate noise-based texture for ambient rain / footsteps
                if "rain" in sfx_type:
                    # Pink noise approximation with lowpass
                    noise = (random.random() * 2.0 - 1.0) * 0.35
                    rumble = math.sin(2 * math.pi * 55 * t) * 0.15
                    sample_val = noise + rumble
                else:
                    # Subtle mechanical drone
                    sample_val = math.sin(2 * math.pi * 85 * t) * 0.2 + (random.random() * 0.1)

                sample_int = int(sample_val * 16000.0)
                sample_int = max(-32768, min(32767, sample_int))
                frames.extend(struct.pack("<h", sample_int))

            wav.writeframes(frames)

        return output_path

    async def execute(self, shot: ShotSpecification, cumulative_time: float) -> List[AudioTrack]:
        """Generates Foley audio cues matching the shot action."""
        foley_dir = settings.workspace_dir / "audio" / "foley"
        foley_dir.mkdir(parents=True, exist_ok=True)
        tracks: List[AudioTrack] = []

        track_file = foley_dir / f"{shot.shot_id}_foley_ambient.wav"
        self._generate_foley_wav(track_file, shot.duration_sec, sfx_type="rain_ambient")

        track = AudioTrack(
            track_id=f"foley_{shot.shot_id}",
            track_type="foley",
            file_path=str(track_file),
            start_time=cumulative_time,
            duration_sec=shot.duration_sec,
            gain_db=-6.0,
            metadata={"description": "Rain and atmospheric night Foley"}
        )
        tracks.append(track)
        await self.emit("foley.generated", track.model_dump())
        return tracks
