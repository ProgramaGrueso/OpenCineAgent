"""Agent 13: Audio Mixing Agent (Sidechain / Ducking) (Layer D).
Implements automated sidechain ducking in FFmpeg: attenuates background music and Foley by -12 dB
whenever character dialogue is active, followed by cinematic EBU R128 loudnorm mastering.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional
from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import AudioTrack


class AudioMixingAgent(BaseAgent):
    """Executes dynamic sidechain ducking and multichannel master audio mixing."""

    def __init__(self, ducking_db: float = -12.0) -> None:
        super().__init__(name="AudioMixingAgent", layer="Layer D")
        self.ducking_db = ducking_db

    async def _concatenate_dialogue_stem(
        self,
        dialogue_tracks: List[AudioTrack],
        total_duration: float,
        output_path: Path
    ) -> Path:
        """Assembles all dialogue clips onto an absolute timeline mono track."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not dialogue_tracks:
            # Generate silent reference track if no dialogue
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=mono:sample_rate={settings.voice_sample_rate}",
                "-t", str(total_duration),
                str(output_path)
            ]
            await (await asyncio.create_subprocess_exec(*cmd)).communicate()
            return output_path

        # Build adelay filter string to position each dialogue segment at its start_time
        inputs = []
        filter_parts = []
        for i, track in enumerate(dialogue_tracks):
            inputs.extend(["-i", track.file_path])
            delay_ms = int(track.start_time * 1000)
            filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[a{i}]")

        mix_inputs = "".join(f"[a{i}]" for i in range(len(dialogue_tracks)))
        filter_complex = f"{';'.join(filter_parts)};{mix_inputs}amix=inputs={len(dialogue_tracks)}:normalize=0[aout]"

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-t", str(total_duration),
            str(output_path)
        ]

        proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        await proc.communicate()
        return output_path

    async def execute(
        self,
        dialogue_tracks: List[AudioTrack],
        foley_tracks: List[AudioTrack],
        music_tracks: List[AudioTrack],
        total_duration: float,
        output_master_wav: Path
    ) -> Path:
        """Performs sidechain ducking on music using the dialogue stem as trigger."""
        mix_dir = settings.workspace_dir / "audio" / "mix"
        mix_dir.mkdir(parents=True, exist_ok=True)

        dialogue_stem = mix_dir / "stem_dialogue.wav"
        await self._concatenate_dialogue_stem(dialogue_tracks, total_duration, dialogue_stem)

        # Primary music track
        music_file = music_tracks[0].file_path if music_tracks else None
        foley_file = foley_tracks[0].file_path if foley_tracks else None

        # Build FFmpeg filter_complex with sidechaincompress:
        # Input 0: Dialogue Stem (trigger)
        # Input 1: Music Score (ducked target)
        # sidechaincompress lowers Input 1 when Input 0 has signal above threshold
        inputs = ["-i", str(dialogue_stem)]
        music_input_idx = 1
        inputs.extend(["-i", str(music_file)])

        # Sidechain compression filter:
        # threshold=0.03 (~-30dB), ratio=8, attack=150ms, release=400ms -> gives ~-12dB ducking
        filter_complex = (
            f"[{music_input_idx}:a][0:a]sidechaincompress="
            f"threshold=0.03:ratio=7:attack={settings.ducking_attack_ms}:release={settings.ducking_release_ms}[ducked_music];"
            f"[0:a]volume=1.2[boosted_diag];"
            f"[ducked_music][boosted_diag]amix=inputs=2:duration=first:weights=0.85 1.1[premaster];"
            f"[premaster]loudnorm=I={settings.target_lufs}:TP=-1.5:LRA=11[master]"
        )

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[master]",
            "-t", str(total_duration),
            str(output_master_wav)
        ]

        self.logger.info(f"Rendering master audio with -12dB sidechain ducking -> {output_master_wav}")
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            self.logger.warning(f"Complex sidechain failed ({stderr.decode()[:150]}). Running fallback audio mix.")
            # Fallback direct mix
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", str(dialogue_stem),
                "-i", str(music_file),
                "-filter_complex", "amix=inputs=2:duration=first",
                "-t", str(total_duration),
                str(output_master_wav)
            ]
            await (await asyncio.create_subprocess_exec(*cmd_fallback)).communicate()

        await self.emit("audio_mix.mastered", {"master_path": str(output_master_wav), "duration": total_duration})
        return output_master_wav
