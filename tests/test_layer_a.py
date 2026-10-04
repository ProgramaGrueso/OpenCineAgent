"""Tests for Layer A: Preproduction and Breakdown Agents."""

import pytest
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.agents.scriptwriter import ScriptwriterAgent
from opencine.agents.shotlist import ShotlistPromptAgent
from opencine.core.models import CameraMotion, ProductionScript, ShotType


@pytest.mark.asyncio
async def test_identity_registry():
    registry = AssetIdentityRegistryAgent()
    chars = registry.list_characters()
    assert len(chars) >= 4

    elena = registry.get_character("elena_vance")
    assert elena is not None
    assert elena.name == "Elena Vance"
    assert "trench coat" in elena.invariable_clothing

    prompt_enhancement = registry.compile_character_prompt_enhancement(["elena_vance"])
    assert "Elena Vance" in prompt_enhancement
    assert "trench coat" in prompt_enhancement


@pytest.mark.asyncio
async def test_scriptwriter_fallback():
    scriptwriter = ScriptwriterAgent()
    script = await scriptwriter.execute(
        premise="Test confrontation in high-altitude cyberpunk tower",
        character_ids=["elena_vance", "victor_cross"]
    )
    assert isinstance(script, ProductionScript)
    assert len(script.scenes) >= 1
    assert len(script.scenes[0].shots) >= 3
    assert script.scenes[0].shots[0].character_ids == ["elena_vance"]


@pytest.mark.asyncio
async def test_shotlist_prompt_engineer():
    registry = AssetIdentityRegistryAgent()
    shotlist_agent = ShotlistPromptAgent(registry)
    scriptwriter = ScriptwriterAgent()
    
    script = await scriptwriter.execute("Cyberpunk rain alley", ["elena_vance"])
    enhanced_script = await shotlist_agent.execute(script)

    shot0 = enhanced_script.scenes[0].shots[0]
    assert "Elena Vance" in shot0.positive_prompt
    assert "photorealistic" in shot0.positive_prompt
    assert shot0.seed > 0
    assert shot0.negative_prompt != ""
