"""Agent 12: Music & Scoring Agent (Layer D).
Generates mood-based thematic background scores separated per scene.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import AudioTrack, SceneScript


class MusicScoringAgent(BaseAgent):
    """Composes mood-driven ambient music score tracks."""

    def __init__(self) -> None:
        super().__init__(name="MusicScoringAgent", layer="Layer D")

    def _synthesize_score_wav(
        self,
        output_path: Path,
        duration_sec: float,
        mood: str = "tense"
    ) -> Path:
        """Synthesizes rich cinematic atmospheric score (deep pads, minor chords, drone)."""
        sample_rate = settings.voice_sample_rate
        total_samples = int(duration_sec * sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Minor triad chord frequencies: D Minor (D2 = 73.42Hz, F2 = 87.31Hz, A2 = 110.0Hz)
        frequencies = [73.42, 87.31, 110.0, 146.83]

        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(2)  # Stereo
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)

            frames = bytearray()
            for i in range(total_samples):
                t = i / sample_rate
                # Cinematic slow swell envelope
                swell = 0.5 + 0.5 * math.sin(2 * math.pi * 0.08 * t)

                # Left channel synthesis
                val_left = (
                    0.4 * math.sin(2 * math.pi * frequencies[0] * t) +
                    0.3 * math.sin(2 * math.pi * frequencies[1] * t + 0.2) +
                    0.2 * math.sin(2 * math.pi * frequencies[3] * t)
                )

                # Right channel with detune / chorus effect
                val_right = (
                    0.4 * math.sin(2 * math.pi * (frequencies[0] * 1.003) * t) +
                    0.3 * math.sin(2 * math.pi * frequencies[2] * t + 0.5) +
                    0.2 * math.sin(2 * math.pi * (frequencies[3] * 0.997) * t)
                )

                int_l = max(-32768, min(32767, int(val_left * swell * 18000.0)))
                int_r = max(-32768, min(32767, int(val_right * swell * 18000.0)))

                frames.extend(struct.pack("<hh", int_l, int_r))

            wav.writeframes(frames)

        return output_path

    async def execute(self, scene: SceneScript, cumulative_start: float, duration_sec: float) -> AudioTrack:
        """Generates continuous score track for the scene."""
        music_dir = settings.workspace_dir / "audio" / "music"
        music_dir.mkdir(parents=True, exist_ok=True)

        music_file = music_dir / f"{scene.scene_id}_score_bed.wav"
        self.logger.info(f"Composing score for scene '{scene.title}' ({duration_sec:.1f}s, mood: {scene.mood})")
        self._synthesize_score_wav(music_file, duration_sec, mood=scene.mood)

        track = AudioTrack(
            track_id=f"score_{scene.scene_id}",
            track_type="music",
            file_path=str(music_file),
            start_time=cumulative_start,
            duration_sec=duration_sec,
            gain_db=-8.0,
            metadata={"scene_id": scene.scene_id, "mood": scene.mood}
        )
        await self.emit("score.composed", track.model_dump())
        return track
