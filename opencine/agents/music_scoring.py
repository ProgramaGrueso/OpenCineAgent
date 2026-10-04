"""Agent 12: Music & Scoring Agent (Layer D).

H3 clips are rendered with "non_diegetic_music: No music." and the score is laid in post, so it
runs continuously across cuts instead of restarting every shot.
  none     no score
  file     a music file you own or licensed (--music): trimmed to the film with fades
  minimax  MiniMax Music 3 in ComfyUI, from the scenes' music directions
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from opencine.agents.base import BaseAgent
from opencine.core import media
from opencine.core.comfy_client import ComfyUIClient, comfy_client
from opencine.core.config import settings
from opencine.core.models import AudioTrack, ProductionScript
from opencine.core.seeds import stable_seed
from opencine.workflows import minimax_music3


class MusicScoringAgent(BaseAgent):
    def __init__(self, client: Optional[ComfyUIClient] = None, mode: Optional[str] = None) -> None:
        super().__init__(name="MusicScoringAgent", layer="Layer D")
        self.comfy = client or comfy_client
        self.mode = mode or settings.music_mode

    def caption(self, script: ProductionScript) -> str:
        directions = [s.music_description for s in script.scenes if s.music_description]
        moods = ", ".join(dict.fromkeys(s.mood for s in script.scenes))
        base = "; ".join(directions) if directions else f"cinematic {script.genre.lower()} film score, {moods}"
        return f"Instrumental film score, no vocals. {base}"

    async def execute(self, script: ProductionScript, total_duration: float, out_dir: Path,
                      music_file: Optional[Path] = None) -> Optional[AudioTrack]:
        mode = "file" if music_file else self.mode
        if mode == "none":
            return None
        raw: Path
        if mode == "file":
            if not music_file or not Path(music_file).exists():
                raise FileNotFoundError(f"Music file not found: {music_file}")
            raw = Path(music_file)
        elif mode == "minimax":
            wf = minimax_music3(self.caption(script), "", total_duration + 2.0,
                                stable_seed(script.title, "score"), "opencine/score")
            raw = (await self.comfy.run(wf, out_dir / "audio" / "raw", "music:score",
                                        suffixes=(".flac", ".wav", ".mp3", ".opus")))[0]
        else:
            raise ValueError(f"Unknown music mode '{mode}' (none | file | minimax)")

        target = out_dir / "audio" / "score.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        fade_out = max(0.0, total_duration - 2.5)
        await media.run_cmd(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", raw, "-t", f"{total_duration:.3f}",
                             "-af", f"afade=t=in:d=1.5,afade=t=out:st={fade_out:.3f}:d=2.5", "-ar", "48000",
                             "-ac", "2", target])
        track = AudioTrack(track_id="score", track_type="music", file_path=str(target),
                           duration_sec=total_duration, gain_db=settings.music_gain_db,
                           metadata={"mode": mode, "source": str(raw)})
        await self.emit("score.ready", track.model_dump())
        return track
