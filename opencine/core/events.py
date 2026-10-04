"""OpenCineAgent Asynchronous Event Bus.
Enables event-driven coordination across all 14 agents in the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List
from pydantic import BaseModel, Field

logger = logging.getLogger("OpenCine.EventBus")


class PipelineEvent(BaseModel):
    name: str = Field(..., description="Unique event identifier (e.g. 'shot.rendered', 'qc.rejected')")
    sender: str = Field(..., description="Agent name producing the event")
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


EventHandler = Callable[[PipelineEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Central asynchronous pub/sub event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._history: List[PipelineEvent] = []

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe an async handler to a specific event topic or '*' for wildcard."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)
        logger.debug(f"Subscribed handler {handler.__name__} to event '{event_name}'")

    async def publish(self, event: PipelineEvent) -> None:
        """Publish an event to all matching subscribers."""
        self._history.append(event)
        logger.info(f"[{event.sender}] -> Event '{event.name}' published")

        # Handlers for specific event + wildcard handlers
        targets = self._subscribers.get(event.name, []) + self._subscribers.get("*", [])
        if targets:
            tasks = [asyncio.create_task(handler(event)) for handler in targets]
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_history(self) -> List[PipelineEvent]:
        return list(self._history)

    def clear(self) -> None:
        self._subscribers.clear()
        self._history.clear()


# Global event bus singleton instance
event_bus = EventBus()
