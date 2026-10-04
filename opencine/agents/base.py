"""Base Agent specification for OpenCineAgent pipeline.
All 14 agents inherit from this interface.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict
from opencine.core.events import EventBus, PipelineEvent, event_bus


class BaseAgent(abc.ABC):
    """Abstract base class for all OpenCine agents."""

    def __init__(self, name: str, layer: str, bus: EventBus = event_bus) -> None:
        self.name = name
        self.layer = layer  # "Layer A", "Layer B", "Layer C", "Layer D"
        self.bus = bus
        self.logger = logging.getLogger(f"OpenCine.{self.name}")

    async def emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Helper to emit an event from this agent."""
        event = PipelineEvent(name=event_name, sender=self.name, payload=payload)
        await self.bus.publish(event)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Lifecycle wrapper that tracks execution time and errors."""
        start_time = time.perf_counter()
        self.logger.info(f"Starting {self.name} [{self.layer}]")
        await self.emit(f"{self.name.lower()}.started", {"args": str(args)})
        try:
            result = await self.execute(*args, **kwargs)
            duration = time.perf_counter() - start_time
            self.logger.info(f"Completed {self.name} in {duration:.2f}s")
            await self.emit(f"{self.name.lower()}.completed", {"duration_sec": duration})
            return result
        except Exception as exc:
            duration = time.perf_counter() - start_time
            self.logger.error(f"Failed {self.name} after {duration:.2f}s: {exc}", exc_info=True)
            await self.emit(f"{self.name.lower()}.failed", {"error": str(exc), "duration_sec": duration})
            raise

    @abc.abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Core execution logic implemented by child agents."""
        raise NotImplementedError
