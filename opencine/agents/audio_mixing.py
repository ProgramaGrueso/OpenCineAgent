"""Agent 13: Audio Mixing Agent (Layer D).

Production audio (H3's native dialogue + sound, cut with the picture) is the lead; the score sits
at a fixed level and is ducked under it with sidechain compression. Loudness is normalized at the
end in two passes, linear: a single-pass loudnorm over a mix with ambience-only stretches pumps
the ambience up to the music's level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from opencine.agents.base import BaseAgent
from opencine.core import media
from opencine.core.config import settings
from opencine.core.models import AudioTrack


class AudioMixingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="AudioMixingAgent", layer="Layer D")

    async def execute(self, production_stem: Path, music: Optional[AudioTrack], total_duration: float,
                      output_master_wav: Path) -> Path:
        output_master_wav.parent.mkdir(parents=True, exist_ok=True)
        premix = output_master_wav.with_name(output_master_wav.stem + "_premix.wav")
        if music:
            graph = (
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,apad,atrim=0:{total_duration:.3f},"
                f"asplit=2[voice][key];"
                f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={music.gain_db}dB[bed];"
                f"[bed][key]sidechaincompress=threshold=0.03:ratio=6:attack={settings.ducking_attack_ms}:"
                f"release={settings.ducking_release_ms}[ducked];"
                f"[voice][ducked]amix=inputs=2:duration=first:normalize=0[mix]"
            )
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", production_stem, "-i", music.file_path,
                   "-filter_complex", graph, "-map", "[mix]", premix]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", production_stem, "-af",
                   f"aformat=sample_rates=48000:channel_layouts=stereo,apad,atrim=0:{total_duration:.3f}", premix]
        await media.run_cmd(cmd)
        await media.loudnorm_two_pass(premix, output_master_wav, settings.target_lufs)
        premix.unlink(missing_ok=True)
        await self.emit("audio_mix.mastered", {"master": str(output_master_wav), "music": bool(music)})
        return output_master_wav
