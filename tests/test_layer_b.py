"""Tests for Layer B: Generation and Spatial Consistency Agents."""

from pathlib import Path
import pytest
from opencine.agents.environment import EnvironmentStageAgent
from opencine.agents.framestaging import FrameStagingAgent
from opencine.agents.video_synthesis import VideoSynthesisAgent
from opencine.core.models import CameraMotion, ShotSpecification, ShotType


@pytest.mark.asyncio
async def test_environment_stage_lock():
    agent = EnvironmentStageAgent()
    stage = await agent.execute("test_stage_terrace", "Cinematic neon cityscape terrace")
    assert stage.stage_id == "test_stage_terrace"
    assert stage.locked_background_image is not None
    assert Path(stage.locked_background_image).exists()


@pytest.mark.asyncio
async def test_frame_staging_keyframe():
    stage_agent = EnvironmentStageAgent()
    staging_agent = FrameStagingAgent()

    stage = await stage_agent.execute("test_stage_lab", "Minimalist cyan lab")
    shot = ShotSpecification(
        shot_id="SH_TEST_01",
        stage_id=stage.stage_id,
        action_description="Elena inspects holographic terminal",
        character_ids=["elena_vance"],
        shot_type=ShotType.CLOSE_UP,
        camera_motion=CameraMotion.STATIC,
        resolution=(640, 360)
    )

    keyframe = await staging_agent.execute(shot, stage)
    assert keyframe.exists()
    assert keyframe.suffix == ".png"


@pytest.mark.asyncio
async def test_video_synthesis_motion():
    stage_agent = EnvironmentStageAgent()
    staging_agent = FrameStagingAgent()
    video_agent = VideoSynthesisAgent()

    stage = await stage_agent.execute("test_stage_lab", "Minimalist cyan lab")
    shot = ShotSpecification(
        shot_id="SH_TEST_02",
        stage_id=stage.stage_id,
        action_description="Marcus looks around alleyway",
        character_ids=["marcus_reid"],
        shot_type=ShotType.MEDIUM,
        camera_motion=CameraMotion.DOLLY_IN,
        duration_sec=2.0,
        fps=24,
        resolution=(640, 360)
    )

    keyframe = await staging_agent.execute(shot, stage)
    video = await video_agent.execute(shot, keyframe)
    assert video.exists()
    assert video.suffix == ".mp4"
