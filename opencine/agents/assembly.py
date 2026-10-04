"""Agent 14: Assembly & Final Cut Agent (Layer D).

  cut       each approved clip is trimmed to its exact frame count (H3 renders 5+17k frames, longer
            than the shot) and conformed to one size/fps, then concatenated. Trimming per clip keeps
            cuts from drifting; the clip's own audio is trimmed with it, so lips stay in sync.
  stem      the production audio of the cut, handed to the mixing agent
  master    picture + mixed audio, SRT sidecar (optionally burned), duration verified
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from opencine.agents.base import BaseAgent
from opencine.core import media
from opencine.core.config import settings
from opencine.core.models import ShotSpecification


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


class AssemblyFinalCutAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="AssemblyFinalCutAgent", layer="Layer D")

    def generate_srt_subtitles(self, shots: List[ShotSpecification], output_srt_path: Path) -> Path:
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        cues, t = [], 0.0
        for shot in shots:
            for line in shot.dialogue:
                start = t + min(line.start_offset, shot.duration_sec)
                end = min(start + line.duration, t + shot.duration_sec)
                if end - start < 0.3:
                    continue
                speaker = f"[{line.speaker_name}] " if line.speaker_name else ""
                cues.append(f"{len(cues) + 1}\n{_srt_time(start)} --> {_srt_time(end)}\n{speaker}{line.text}\n")
            t += shot.duration_sec
        output_srt_path.write_text("\n".join(cues), encoding="utf-8")
        return output_srt_path

    def generate_text_overlay(self, shots: List[ShotSpecification], output_ass_path: Path,
                              width: int, height: int) -> Optional[Path]:
        """ASS file with every shot's on-screen text at absolute times (None if there is none)."""
        # \an alignment (numpad), margins keep text inside a ~7% safe area
        layout = {"left": (4, 0.07, 0), "right": (6, 0.07, 0), "center": (5, 0, 0),
                  "bottom": (2, 0, 0.08), "title": (5, 0, 0)}
        events, t = [], 0.0
        for shot in shots:
            for cue in shot.on_screen_text:
                start, end = t + cue.start, t + min(cue.end, shot.duration_sec)
                if end <= start:
                    continue
                align, mx, my = layout.get(cue.position, layout["right"])
                style = "Title" if cue.position == "title" else "Label"
                text = cue.text.replace("\n", "\\N")
                events.append(
                    f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,"
                    f"{int(width * mx)},{int(width * mx)},{int(height * my)},,"
                    f"{{\\an{align}\\fad(250,250)}}{text}")
            t += shot.duration_sec
        if not events:
            return None
        label = max(18, height // 30)
        header = (
            "[Script Info]\nScriptType: v4.00+\n"
            f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Label,Noto Sans,{label},&H00303030,&H00FFFFFF,&H00000000,0,0,0,0,100,100,1,0,1,0,0,6,0,0,0,1\n"
            f"Style: Title,Noto Sans,{label * 2},&H002020E0,&H00FFFFFF,&H00000000,1,0,0,0,100,100,2,0,1,0,0,5,0,0,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        output_ass_path.parent.mkdir(parents=True, exist_ok=True)
        output_ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
        return output_ass_path

    async def cut(self, approved: List[Tuple[ShotSpecification, Path]], out_dir: Path) -> Tuple[Path, Path, float]:
        edit = out_dir / "edit"
        edit.mkdir(parents=True, exist_ok=True)
        first = media.probe(approved[0][1])
        width, height = int(first["width"]), int(first["height"])
        segments = []
        for i, (shot, clip) in enumerate(approved):
            frames = round(shot.duration_sec * settings.fps)
            seconds = frames / settings.fps
            seg = edit / f"{i:03d}_{shot.shot_id}.mp4"
            await media.run_cmd([
                "ffmpeg", "-v", "error", "-y", "-i", clip,
                "-vf", f"fps={settings.fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
                "-af", f"aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{seconds:.4f}",
                "-frames:v", str(frames), "-c:v", "libx264", "-crf", "15", "-preset", "medium",
                "-c:a", "pcm_s16le", "-shortest", seg.with_suffix(".mkv"),
            ])
            segments.append(seg.with_suffix(".mkv"))
        listing = edit / "concat.txt"
        listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segments), encoding="utf-8")
        picture = edit / "picture_cut.mkv"
        await media.run_cmd(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
                             "-c", "copy", picture])
        stem = edit / "production_audio.wav"
        await media.run_cmd(["ffmpeg", "-v", "error", "-y", "-i", picture, "-vn", "-c:a", "pcm_s16le", stem])
        total = sum(round(s.duration_sec * settings.fps) for s, _ in approved) / settings.fps
        return picture, stem, total

    async def master(self, picture: Path, master_audio: Path, srt: Path, total: float, out_dir: Path,
                     title_slug: str, burn_subtitles: bool = False, text_overlay: Optional[Path] = None) -> Path:
        final = out_dir / f"{title_slug}_master.mp4"
        filters = []
        if text_overlay:
            filters.append(f"ass={text_overlay.resolve()}")
        if burn_subtitles:
            style = "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,MarginV=40"
            filters.append(f"subtitles={srt.resolve()}:force_style='{style}'")
        vf = ["-c:v", "copy"]
        if filters:
            vf = ["-vf", ",".join(filters), "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]
        await media.run_cmd(["ffmpeg", "-v", "error", "-y", "-i", picture, "-i", master_audio,
                             "-map", "0:v:0", "-map", "1:a:0", *vf, "-c:a", "aac", "-b:a", "256k",
                             "-t", f"{total:.4f}", "-movflags", "+faststart", final])
        got = media.probe(final)["duration"]
        if abs(got - total) > 0.15:
            raise media.MediaError(f"Master is {got:.2f}s, expected {total:.2f}s")
        await self.emit("production.master", {"path": str(final), "duration": got})
        return final

    async def execute(self, approved: List[Tuple[ShotSpecification, Path]], out_dir: Path) -> Tuple[Path, Path, float]:
        return await self.cut(approved, out_dir)
