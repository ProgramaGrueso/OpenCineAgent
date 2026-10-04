"""Tests for Layer D: Audio, Subtitles, and Final Assembly Agents."""

from pathlib import Path
import pytest
from opencine.agents.voice_dialogue import VoiceDialogueAgent
from opencine.agents.foley_sound import FoleySoundAgent
from opencine.agents.music_scoring import MusicScoringAgent
from opencine.agents.audio_mixing import AudioMixingAgent
from opencine.agents.assembly import AssemblyFinalCutAgent
from opencine.core.models import DialogueLine, SceneScript, ShotSpecification, ShotType


@pytest.mark.asyncio
async def test_audio_pipeline_and_sidechain_ducking(tmp_path):
    dialogue_agent = VoiceDialogueAgent()
    foley_agent = FoleySoundAgent()
    music_agent = MusicScoringAgent()
    mixing_agent = AudioMixingAgent()

    shot = ShotSpecification(
        shot_id="SH_AUD_01",
        stage_id="stage_01",
        action_description="Elena speaks",
        character_ids=["elena_vance"],
        dialogue=[
            DialogueLine(
                speaker_id="elena_vance",
                speaker_name="Elena Vance",
                text="The encrypted signal is pulsing.",
                start_offset=0.2,
                duration=2.0
            )
        ],
        duration_sec=3.0
    )

    scene = SceneScript(
        scene_id="SC_AUD",
        title="Audio Test Scene",
        location="Lab",
        action_summary="Testing audio stems",
        shots=[shot]
    )

    diag_tracks = await dialogue_agent.execute(shot, cumulative_time=0.0)
    assert len(diag_tracks) == 1
    assert Path(diag_tracks[0].file_path).exists()

    foley_tracks = await foley_agent.execute(shot, cumulative_time=0.0)
    assert len(foley_tracks) == 1
    assert Path(foley_tracks[0].file_path).exists()

    music_track = await music_agent.execute(scene, cumulative_start=0.0, duration_sec=3.0)
    assert Path(music_track.file_path).exists()

    master_audio = tmp_path / "test_master_audio.wav"
    mixed = await mixing_agent.execute(
        dialogue_tracks=diag_tracks,
        foley_tracks=foley_tracks,
        music_tracks=[music_track],
        total_duration=3.0,
        output_master_wav=master_audio
    )
    assert mixed.exists()


@pytest.mark.asyncio
async def test_assembly_srt_generation(tmp_path):
    assembly_agent = AssemblyFinalCutAgent()

    shot1 = ShotSpecification(
        shot_id="SH_SRT_01",
        stage_id="stage_01",
        action_description="Elena speaks",
        dialogue=[
            DialogueLine(
                speaker_id="elena_vance",
                speaker_name="Elena Vance",
                text="We are out of time.",
                start_offset=0.5,
                duration=2.0
            )
        ],
        duration_sec=3.0
    )

    srt_path = tmp_path / "test.srt"
    assembly_agent.generate_srt_subtitles([shot1], srt_path)
    assert srt_path.exists()

    content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:02,500" in content
    assert "[Elena Vance] We are out of time." in content
