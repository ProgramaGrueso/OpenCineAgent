"""Quality control checks on real (synthetic) clips."""

import subprocess

import pytest

from opencine.agents.critic_vlm import CriticVlmAgent, dialogue_similarity
from opencine.agents.healing import LoopbackHealingAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.config import settings
from opencine.core.models import CheckStatus, DialogueLine, HealingAction, QCCheck, QCReport, ShotSpecification


def make_clip(path, source, seconds=6, audio="sine=f=300"):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}=s=320x180:r=24:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"{audio}:d={seconds}", "-shortest", "-c:a", "aac"]
    subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def critic(registry_file):
    return CriticVlmAgent(AssetIdentityRegistryAgent(registry_file))


def shot(**kw):
    base = dict(shot_id="SH_1", action_description="x", stage_id="s", duration_sec=5.0)
    base.update(kw)
    return ShotSpecification(**base)


def test_moving_clip_passes_technical_checks(critic, tmp_path):
    clip = make_clip(tmp_path / "ok.mp4", "testsrc2")
    s = shot()
    assert critic.check_technical(s, clip).status == CheckStatus.PASSED
    assert critic.check_motion(clip).status == CheckStatus.PASSED
    assert critic.check_audio(s, clip).status == CheckStatus.PASSED


def test_frozen_short_and_silent_clips_fail(critic, tmp_path):
    frozen = make_clip(tmp_path / "frozen.mp4", "color", audio=None)
    assert critic.check_motion(frozen).status == CheckStatus.FAILED
    short = make_clip(tmp_path / "short.mp4", "testsrc2", seconds=2)
    assert critic.check_technical(shot(), short).status == CheckStatus.FAILED
    silent = make_clip(tmp_path / "silent.mp4", "testsrc2", audio="anullsrc=r=48000")
    talking = shot(dialogue=[DialogueLine(speaker_id="elena_vance", text="hello")])
    assert critic.check_audio(talking, silent).status == CheckStatus.FAILED


async def test_unavailable_tools_are_skipped_not_passed(critic, tmp_path, monkeypatch):
    clip = make_clip(tmp_path / "ok.mp4", "testsrc2")
    s = shot(character_ids=["elena_vance"], dialogue=[DialogueLine(speaker_id="elena_vance", text="hello")])

    def no_whisper(*_):
        raise ImportError("faster_whisper")
    monkeypatch.setattr(CriticVlmAgent, "_transcribe", staticmethod(no_whisper))
    report = await critic.execute(s, clip, tmp_path, attempt=1)
    status = {c.name: c.status for c in report.checks}
    assert status["dialogue"] == CheckStatus.SKIPPED
    assert status["identity"] == CheckStatus.SKIPPED
    assert set(report.unverified) == {"dialogue", "identity"}
    assert report.pass_qc  # skipped checks are reported, not failed
    assert len(report.inspected_frames) == 4


async def test_dialogue_mismatch_fails(critic, tmp_path, monkeypatch):
    clip = make_clip(tmp_path / "ok.mp4", "testsrc2")
    s = shot(dialogue=[DialogueLine(speaker_id="elena_vance", text="You knew the encryption was breached")])
    monkeypatch.setattr(CriticVlmAgent, "_transcribe", staticmethod(lambda *_: "the weather is lovely today"))
    check, heard = await critic.check_dialogue(s, clip)
    assert check.status == CheckStatus.FAILED and heard
    monkeypatch.setattr(CriticVlmAgent, "_transcribe",
                        staticmethod(lambda *_: "You knew the encryption was breached."))
    check, _ = await critic.check_dialogue(s, clip)
    assert check.status == CheckStatus.PASSED


def test_dialogue_similarity_ignores_accents_and_punctuation():
    assert dialogue_similarity("¿Qué pasó, causa?", "que paso causa") > 0.95
    assert dialogue_similarity("hola", "") == 0.0


async def test_healing_retries_then_stops():
    healer = LoopbackHealingAgent()
    s = shot(seed=11, length_frames=192, duration_sec=8)
    failed = QCReport(shot_id="SH_1", attempt=1, pass_qc=False, error_codes=["ERR_IDENTITY"],
                      healing_action=HealingAction.RETRY_NEW_SEED,
                      checks=[QCCheck(name="identity", status=CheckStatus.FAILED)])
    retried = await healer.execute(s, failed)
    assert retried is not None and retried.seed != 11
    last = failed.model_copy(update={"attempt": settings.max_qc_retries + 1})
    assert await healer.execute(s, last) is None
    assert healer.same_failure_twice([failed, failed])
    assert healer.shorten_for_oom(s) and s.length_frames == settings.h3_min_frames
    assert s.duration_sec <= settings.h3_min_frames / settings.fps
    assert not healer.shorten_for_oom(s)


def test_best_attempt_prefers_fewest_failures():
    def report(attempt, failures):
        return QCReport(shot_id="S", attempt=attempt, pass_qc=failures == 0,
                        checks=[QCCheck(name=f"c{i}", status=CheckStatus.FAILED) for i in range(failures)])
    takes = [(report(1, 2), "a"), (report(2, 1), "b"), (report(3, 1), "c")]
    assert LoopbackHealingAgent.best_attempt(takes)[1] == "c"
