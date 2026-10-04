"""End-to-End Test for the Complete 14-Agent Production Pipeline."""

from pathlib import Path
import pytest
from opencine.core.models import ProductionResult
from opencine.orchestrator import OpenCineOrchestrator


@pytest.mark.asyncio
async def test_end_to_end_production():
    orchestrator = OpenCineOrchestrator()

    result = await orchestrator.run_pipeline(
        premise="Elena Vance and Victor Cross confront each other under neon rainfall.",
        character_ids=["elena_vance", "victor_cross"]
    )

    assert isinstance(result, ProductionResult)
    assert Path(result.final_master_path).exists()
    assert Path(result.subtitles_srt_path).exists()
    assert len(result.shot_results) >= 3
    assert result.total_duration_sec > 0
    assert result.qc_pass_rate == 1.0
