"""End-to-end production with the test ComfyUI double: identity -> keyframes -> H3 -> QC -> cut -> mix -> master."""

import json
import subprocess
from pathlib import Path

import pytest

from opencine.agents.identity_registry import IdentityNotReady
from opencine.agents.scriptwriter import ScriptGenerationError, ScriptwriterAgent
from opencine.agents.shotlist import ShotlistError
from opencine.core import media
from opencine.core.config import settings
from opencine.core.models import CheckStatus, ProductionScript
from opencine.orchestrator import OpenCineOrchestrator


@pytest.fixture
def orchestrator(fake_comfy, registry_file):
    return OpenCineOrchestrator(client=fake_comfy, registry_file=registry_file)


async def test_full_production(orchestrator, fake_comfy, example_script):
    result = await orchestrator.run_pipeline(script_path=example_script, auto_approve=True)

    master = Path(result.final_master_path)
    info = media.probe(master)
    assert info["has_audio"] and abs(info["duration"] - 17.0) < 0.1  # 5 + 6 + 6 seconds, trimmed exactly
    assert len(result.shot_results) == 3 and result.qc_pass_rate == 1.0
    srt = Path(result.subtitles_srt_path).read_text(encoding="utf-8")
    assert "00:00:05,600 --> 00:00:08,800" in srt and "[Victor Cross]" in srt
    report = (master.parent / "REPORT.md").read_text(encoding="utf-8")
    assert "identity" in report  # the VLM was not available: reported as not verified

    labels = fake_comfy.labels
    assert sum(label.startswith("portrait:") for label in labels) == 2      # Elena and Victor, once each
    assert sum(label.startswith("stage:") for label in labels) == 1         # one plate per location
    assert sum(label.startswith("video:") for label in labels) == 3
    # every keyframe is an Identity Edit from the shared plate + the approved portrait
    edits = [wf for wf in fake_comfy.jobs if any(n["class_type"] == "Krea2EditModelPatch" for n in wf.values())]
    assert len(edits) == 3
    for wf in fake_comfy.jobs:
        if any(n["class_type"] == "MiniMaxH3ReferenceToVideo" for n in wf.values()):
            assert len([k for k in wf if k.startswith("img")]) == 2   # keyframe + master portrait

    # resume: approved shots are not rendered again
    renders = len(fake_comfy.jobs)
    await orchestrator.run_pipeline(script_path=example_script, auto_approve=True)
    assert len(fake_comfy.jobs) == renders


async def test_only_one_shot_needs_only_its_cast(orchestrator, fake_comfy, example_script):
    result = await orchestrator.run_pipeline(script_path=example_script, only=["SH_003"], auto_approve=True)
    assert [r.shot_id for r in result.shot_results] == ["SH_003"]
    assert not any(label == "portrait:elena_vance:0" for label in fake_comfy.labels)


async def test_identity_gate_blocks_unapproved_cast(orchestrator, example_script):
    with pytest.raises(IdentityNotReady):
        await orchestrator.run_pipeline(script_path=example_script)


async def test_failed_qc_retries_with_new_seed(orchestrator, fake_comfy, example_script, monkeypatch):
    calls = {"n": 0}
    original = orchestrator.critic_agent.check_motion

    def flaky_motion(clip):
        calls["n"] += 1
        check = original(clip)
        if calls["n"] == 1:
            check.status, check.detail = CheckStatus.FAILED, "forced failure"
        return check

    monkeypatch.setattr(orchestrator.critic_agent, "check_motion", flaky_motion)
    result = await orchestrator.run_pipeline(script_path=example_script, only=["SH_001"], auto_approve=True)
    shot = result.shot_results[0]
    assert shot.attempts == 2 and shot.approved
    seeds = [wf["ks"]["inputs"]["seed"] for wf in fake_comfy.jobs if "r2v" in wf]
    assert len(set(seeds)) == 2


async def test_lipsync_mode_uses_voice_track(orchestrator, fake_comfy, example_script, tmp_path):
    line = tmp_path / "line.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=200:d=2", line], check=True)
    data = json.loads(example_script.read_text(encoding="utf-8"))
    data["scenes"][0]["shots"][1]["dialogue"][0]["audio_path"] = str(line)
    script = tmp_path / "script.json"
    script.write_text(json.dumps(data), encoding="utf-8")

    await orchestrator.run_pipeline(script_path=script, only=["SH_002"], auto_approve=True)
    h3 = next(wf for wf in fake_comfy.jobs if "r2v" in wf)
    assert "sync" in h3 and h3["r2v"]["inputs"]["ref_audios.ref_audio_0"] == ["voice", 0]
    voice = settings.workspace_dir / "productions" / "the_rooftop_protocol" / "voice" / "SH_002.wav"
    assert abs(media.probe(voice)["duration"] - (158 / 24 + 0.03)) < 0.02


async def test_music_from_minimax_is_ducked_into_the_mix(orchestrator, fake_comfy, example_script, monkeypatch):
    monkeypatch.setattr(settings, "music_mode", "minimax")
    monkeypatch.setattr(orchestrator.music_agent, "mode", "minimax")
    result = await orchestrator.run_pipeline(script_path=example_script, only=["SH_001"], auto_approve=True)
    assert any(label == "music:score" for label in fake_comfy.labels)
    assert media.probe(Path(result.final_master_path))["has_audio"]


async def test_shotlist_rejects_contract_violations(orchestrator, example_script, tmp_path):
    data = json.loads(example_script.read_text(encoding="utf-8"))
    data["scenes"][0]["shots"][1]["dialogue"][0]["text"] = "word " * 40
    script = tmp_path / "bad.json"
    script.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ShotlistError):
        await orchestrator.plan(script_path=script)


async def test_scriptwriter_validates_llm_output(registry_file, monkeypatch):
    from opencine.agents.identity_registry import AssetIdentityRegistryAgent
    cast = AssetIdentityRegistryAgent(registry_file).cast(["elena_vance"])
    writer = ScriptwriterAgent()
    good = {"title": "T", "scenes": [{"scene_id": "SC_01", "title": "a", "location": "a room", "shots": [
        {"shot_id": "SH_001", "action_description": "she waits", "stage_id": "s", "character_ids": ["elena_vance"]}]}]}
    bad = json.loads(json.dumps(good))
    bad["scenes"][0]["shots"][0]["character_ids"] = ["someone_else"]
    answers = iter([json.dumps(bad), json.dumps(good)])

    async def fake_ask(system, user):
        return next(answers)
    monkeypatch.setattr(writer, "_ask", fake_ask)
    script = await writer.execute("premise", cast, "Spanish")
    assert isinstance(script, ProductionScript) and script.language == "Spanish"

    async def always_bad(system, user):
        return json.dumps(bad)
    monkeypatch.setattr(writer, "_ask", always_bad)
    with pytest.raises(ScriptGenerationError):
        await writer.execute("premise", cast)


async def test_refmod_extraction_is_verified(orchestrator, fake_comfy, tmp_path):
    image = tmp_path / "face.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1", image],
                   check=True)
    name = await orchestrator.registry_agent.extract_refmod("elena_vance", [image])
    assert name == "opencine_elena_vance_refmod"
    assert orchestrator.registry_agent.get_character("elena_vance").refmod_name == name


async def test_global_seed_keyframe_action_and_on_screen_text(orchestrator, fake_comfy, example_script, tmp_path):
    data = json.loads(example_script.read_text(encoding="utf-8"))
    data["seed"] = 777
    first = data["scenes"][0]["shots"][0]
    first["keyframe_action"] = "Elena stands still with a neutral face"
    first["on_screen_text"] = [{"text": "Neutral", "start": 0.2, "end": 2.0, "position": "left"},
                               {"text": "TITLE", "start": 0.0, "end": 1.5, "position": "title"}]
    script = tmp_path / "script.json"
    script.write_text(json.dumps(data), encoding="utf-8")

    plan, _ = await orchestrator.plan(script_path=script)
    shots = plan.all_shots()
    assert {s.seed for s in shots} == {777}
    assert "neutral face" in shots[0].keyframe_prompt and "steps out" not in shots[0].keyframe_prompt
    assert "steps out" in shots[0].video_prompt

    result = await orchestrator.run_pipeline(script_path=script, only=["SH_001"], auto_approve=True)
    ass = Path(result.final_master_path).parent / "on_screen_text.ass"
    text = ass.read_text(encoding="utf-8")
    assert "0:00:00.20,0:00:02.00,Label" in text and "\\an4" in text and "Title" in text
    assert media.probe(Path(result.final_master_path))["duration"] == pytest.approx(5.0, abs=0.1)
