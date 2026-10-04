"""Tests for Layer C: Quality Control and Healing Agents."""

from pathlib import Path
import pytest
from opencine.agents.critic_vlm import CriticVlmAgent
from opencine.agents.healing import LoopbackHealingAgent
from opencine.agents.super_res import SuperResolutionAgent
from opencine.agents.environment import EnvironmentStageAgent
from opencine.agents.framestaging import FrameStagingAgent
from opencine.agents.video_synthesis import VideoSynthesisAgent
from opencine.core.models import HealingAction, ShotSpecification, ShotType


@pytest.fixture
async def sample_video_clip():
    stage_agent = EnvironmentStageAgent()
    staging_agent = FrameStagingAgent()
    video_agent = VideoSynthesisAgent()

    stage = await stage_agent.execute("test_qc_stage", "Cyberpunk corridor")
    shot = ShotSpecification(
        shot_id="SH_QC_01",
        stage_id="test_qc_stage",
        action_description="Victor turns towards camera",
        character_ids=["victor_cross"],
        duration_sec=2.0,
        fps=24,
        resolution=(640, 360)
    )
    keyframe = await staging_agent.execute(shot, stage)
    return await video_agent.execute(shot, keyframe)


@pytest.mark.asyncio
async def test_critic_vlm_pass(sample_video_clip):
    video = sample_video_clip
    critic = CriticVlmAgent()
    # Seed 42 produces clean inspection
    report = await critic.execute("SH_QC_01", video, seed=42)
    assert report.pass_qc is True
    assert report.identity_drift_score <= critic.drift_threshold
    assert report.healing_action == HealingAction.NONE


@pytest.mark.asyncio
async def test_critic_vlm_rejection_and_healing(sample_video_clip):
    video = sample_video_clip
    critic = CriticVlmAgent()
    healing = LoopbackHealingAgent()

    # Seed 47 produces mild facial asymmetry -> triggers FaceDetailer
    report_mild = await critic.execute("SH_QC_01", video, seed=47)
    assert report_mild.pass_qc is False
    assert report_mild.healing_action == HealingAction.FACE_DETAILER

    shot = ShotSpecification(
        shot_id="SH_QC_01",
        stage_id="test_qc_stage",
        action_description="Victor turns",
        seed=47
    )
    healed_clip, _ = await healing.execute(shot, video, report_mild)
    assert healed_clip is not None
    assert healed_clip.exists()

    # Seed 49 produces critical error -> triggers retry mutation
    report_crit = await critic.execute("SH_QC_01", video, seed=49)
    assert report_crit.pass_qc is False
    assert report_crit.healing_action == HealingAction.RETRY_MUTATION

    _, mutated_shot = await healing.execute(shot, video, report_crit)
    assert mutated_shot.seed != 47


@pytest.mark.asyncio
async def test_super_resolution(sample_video_clip):
    video = sample_video_clip
    super_res = SuperResolutionAgent(target_fps=60, target_width=1280, target_height=720)
    master = await super_res.execute(video, "SH_QC_01")
    assert master.exists()
    assert master.suffix == ".mp4"
