"""Agent 14: Assembly & Final Cut Agent (Layer D).
Sequentially stitches QC-approved video clips in FFmpeg, multiplexes the sidechain master audio,
and generates/burns synchronized SRT/ASS subtitles with cinematic typography.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import List, Tuple
from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import DialogueLine, ProductionResult, ShotRenderResult, ShotSpecification


class AssemblyFinalCutAgent(BaseAgent):
    """Assembles final cinematic master deliverable with burned or embedded subtitles."""

    def __init__(self) -> None:
        super().__init__(name="AssemblyFinalCutAgent", layer="Layer D")

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Converts float seconds to SRT time format: 00:00:00,000"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    def generate_srt_subtitles(
        self,
        shots: List[ShotSpecification],
        output_srt_path: Path
    ) -> Path:
        """Generates valid SubRip (.srt) subtitle document from dialogue events."""
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_lines: List[str] = []
        subtitle_index = 1
        cumulative_time = 0.0

        for shot in shots:
            for diag in shot.dialogue:
                start_sec = cumulative_time + diag.start_offset
                end_sec = start_sec + diag.duration
                srt_lines.append(f"{subtitle_index}")
                srt_lines.append(f"{self._format_srt_timestamp(start_sec)} --> {self._format_srt_timestamp(end_sec)}")
                speaker = f"[{diag.speaker_name}]" if diag.speaker_name else ""
                srt_lines.append(f"{speaker} {diag.text}".strip())
                srt_lines.append("")
                subtitle_index += 1

            cumulative_time += shot.duration_sec

        output_srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        self.logger.info(f"Generated subtitles at {output_srt_path} ({subtitle_index - 1} cues)")
        return output_srt_path

    async def execute(
        self,
        approved_shots: List[Tuple[ShotSpecification, Path]],
        master_audio_path: Path,
        production_title: str
    ) -> ProductionResult:
        """Stitches all video clips, synchronizes audio, and produces final release MP4."""
        output_dir = settings.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        concat_list_file = settings.workspace_dir / "concat_list.txt"
        srt_file = output_dir / f"{production_title.lower().replace(' ', '_')}.srt"
        final_video_file = output_dir / f"{production_title.lower().replace(' ', '_')}_master.mp4"

        # 1. Write FFmpeg concat manifest
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for shot, clip_path in approved_shots:
                f.write(f"file '{clip_path.resolve()}'\n")

        # 2. Extract dialogue lines and build subtitles
        shots_specs = [s[0] for s in approved_shots]
        self.generate_srt_subtitles(shots_specs, srt_file)

        # 3. Concatenate video clips + attach audio stem + burn subtitles
        total_duration = sum(s.duration_sec for s in shots_specs)

        # Use subtitles filter with cinematic font styling
        subtitle_filter = f"subtitles={srt_file.resolve()}:force_style='FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=1,MarginV=30'"

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file),
            "-i", str(master_audio_path),
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            str(final_video_file)
        ]

        self.logger.info(f"Rendering final master movie: {final_video_file.name}")
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            self.logger.warning(f"Subtitle burn failed ({stderr.decode()[:150]}). Re-running without burned subtitles.")
            # Fallback without burn
            cmd_fallback = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-i", str(master_audio_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "256k",
                "-shortest",
                str(final_video_file)
            ]
            await (await asyncio.create_subprocess_exec(*cmd_fallback)).communicate()

        result = ProductionResult(
            production_id=f"PROD_{len(approved_shots)}SH",
            title=production_title,
            final_master_path=str(final_video_file),
            subtitles_srt_path=str(srt_file),
            total_duration_sec=total_duration,
            qc_pass_rate=1.0
        )

        await self.emit("production.finished", result.model_dump())
        return result
