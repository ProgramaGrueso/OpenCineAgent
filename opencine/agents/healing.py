"""Agent 8: Loopback & Healing Agent (Layer C).

What actually fixes generated shots in practice:
  * a new seed with the same prompt (two seeds usually yield one clearly better take; when two seeds
    fail the same way the prompt is the problem, and that is reported instead of retried forever);
  * CUDA out of memory -> a shorter clip (frames x reference images is what fills VRAM);
  * if every attempt fails, the take with the fewest failed checks is kept and flagged.
Every attempt stays on disk (<shot>_t<N>_s<seed>.mp4); nothing approved is overwritten.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import CheckStatus, QCReport, ShotSpecification
from opencine.core.seeds import stable_seed


class LoopbackHealingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="LoopbackHealingAgent", layer="Layer C")

    def shorten_for_oom(self, shot: ShotSpecification) -> bool:
        """Drops to the shortest trained length. Returns False if it is already there."""
        if shot.length_frames <= settings.h3_min_frames:
            return False
        shot.length_frames = settings.h3_min_frames
        shot.duration_sec = min(shot.duration_sec, settings.h3_min_frames / settings.fps)
        self.logger.warning(f"{shot.shot_id}: OOM, retrying at {shot.length_frames} frames ({shot.duration_sec:.2f}s)")
        return True

    @staticmethod
    def same_failure_twice(reports: List[QCReport]) -> bool:
        return len(reports) >= 2 and set(reports[-1].error_codes) == set(reports[-2].error_codes) != set()

    @staticmethod
    def best_attempt(attempts: List[Tuple[QCReport, object]]) -> Tuple[QCReport, object]:
        def score(item: Tuple[QCReport, object]) -> Tuple[int, int]:
            report = item[0]
            failed = sum(c.status == CheckStatus.FAILED for c in report.checks)
            return failed, -report.attempt
        return min(attempts, key=score)

    async def execute(self, shot: ShotSpecification, report: QCReport) -> Optional[ShotSpecification]:
        """Returns the shot to render next, or None when retrying will not help."""
        if report.pass_qc or report.attempt > settings.max_qc_retries:
            return None
        shot.seed = stable_seed(shot.shot_id, "retry", report.attempt, shot.seed)
        await self.emit("healing.retry", {"shot_id": shot.shot_id, "new_seed": shot.seed,
                                          "errors": report.error_codes})
        return shot
