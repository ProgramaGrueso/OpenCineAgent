"""ComfyUI Headless REST / WebSocket API Client.
Interfaces with ComfyUI running at --listen --port 8188.
Includes robust polling, timeout recovery, and mock fallback for tests/environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from opencine.core.config import settings

logger = logging.getLogger("OpenCine.ComfyClient")


class ComfyUIClient:
    """Headless client for ComfyUI REST API."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host = host or settings.comfyui_host
        self.port = port or settings.comfyui_port
        self.base_url = f"http://{self.host}:{self.port}"
        self.client_id = str(uuid.uuid4())

    async def is_alive(self) -> bool:
        """Check if ComfyUI headless daemon is responding."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.base_url}/system_stats")
                return res.status_code == 200
        except Exception:
            return False

    async def queue_prompt(self, workflow_prompt: Dict[str, Any]) -> Optional[str]:
        """Post a workflow graph to ComfyUI /prompt and return the prompt_id."""
        payload = {
            "prompt": workflow_prompt,
            "client_id": self.client_id
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(f"{self.base_url}/prompt", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    prompt_id = data.get("prompt_id")
                    logger.info(f"Queued ComfyUI job prompt_id={prompt_id}")
                    return prompt_id
                else:
                    logger.error(f"ComfyUI prompt error: {res.status_code} - {res.text}")
                    return None
        except Exception as exc:
            logger.warning(f"Failed to post to ComfyUI: {exc}")
            return None

    async def wait_for_completion(self, prompt_id: str, timeout_sec: float = 300.0) -> Optional[Dict[str, Any]]:
        """Poll /history/{prompt_id} until execution is finished."""
        start = asyncio.get_event_loop().time()
        poll_interval = 2.0

        async with httpx.AsyncClient(timeout=5.0) as client:
            while (asyncio.get_event_loop().time() - start) < timeout_sec:
                try:
                    res = await client.get(f"{self.base_url}/history/{prompt_id}")
                    if res.status_code == 200:
                        data = res.json()
                        if prompt_id in data:
                            history_entry = data[prompt_id]
                            # Check if status completed
                            status = history_entry.get("status", {})
                            if status.get("completed", False) or "outputs" in history_entry:
                                logger.info(f"Job {prompt_id} completed successfully")
                                return history_entry
                except Exception as exc:
                    logger.debug(f"Polling history error: {exc}")

                await asyncio.sleep(poll_interval)

        logger.warning(f"Timeout waiting for ComfyUI prompt_id={prompt_id}")
        return None

    async def download_output_file(self, filename: str, subfolder: str = "", folder_type: str = "output", dest_path: Optional[Path] = None) -> Optional[Path]:
        """Download rendered image or video from ComfyUI /view."""
        dest = dest_path or (settings.temp_dir / filename)
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.get(f"{self.base_url}/view", params=params)
                if res.status_code == 200:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(res.content)
                    return dest
        except Exception as exc:
            logger.error(f"Failed to download ComfyUI output {filename}: {exc}")
        return None


# Global client singleton
comfy_client = ComfyUIClient()
