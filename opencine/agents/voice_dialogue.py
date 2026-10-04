"""Agent 10: Voice & Dialogue Agent (Layer D).

Two modes per shot:
  native   (default) H3 speaks the <d>[Language] ...</d> lines itself, in the voice described in the
           character profile. Nothing to do here; the QC agent checks the words with Whisper.
  lipsync  every line of the shot has `audio_path` (an actor recording or a TTS take). This agent
           lays the lines on one WAV per clip at their exact offsets (sample-accurate, +30 ms tail so
           the render never truncates them). The same track drives PixaromaH3AudioSync and the mix,
           so trimming a clip can never desynchronize lips and voice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from opencine.agents.base import BaseAgent
from opencine.core import media
from opencine.core.config import settings
from opencine.core.models import ShotSpecification

SAMPLE_RATE = 48000


class VoiceDialogueAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="VoiceDialogueAgent", layer="Layer D")

    async def execute(self, shot: ShotSpecification, out_dir: Path) -> Optional[Path]:
        recorded = [line for line in shot.dialogue if line.audio_path]
        if not recorded:
            return None
        if len(recorded) != len(shot.dialogue):
            raise ValueError(f"{shot.shot_id}: either every line has audio_path (lipsync) or none (native voice)")
        clip_seconds = shot.length_frames / settings.fps + 0.03
        target = out_dir / "voice" / f"{shot.shot_id}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-v", "error", "-y"]
        chains, labels = [], []
        for i, line in enumerate(shot.dialogue):
            if not Path(line.audio_path).exists():
                raise FileNotFoundError(f"{shot.shot_id}: missing recording {line.audio_path}")
            cmd += ["-i", line.audio_path]
            delay = round(line.start_offset * SAMPLE_RATE)
            chains.append(f"[{i}:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=mono,"
                          f"adelay={delay}S:all=1[l{i}]")
            labels.append(f"[l{i}]")
        chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,apad,"
                      f"atrim=0:{clip_seconds:.3f}[out]")
        cmd += ["-filter_complex", ";".join(chains), "-map", "[out]", "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-c:a", "pcm_s16le", str(target)]
        await media.run_cmd(cmd)
        shot.voice_track = str(target)
        await self.emit("dialogue.track", {"shot_id": shot.shot_id, "path": str(target)})
        return target
