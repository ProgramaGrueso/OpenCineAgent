"""ComfyUI headless API client.

Production rules baked in (each one comes from a real failure, see docs/PRODUCTION_NOTES.md):
  * A workflow is validated against /object_info BEFORE it is queued: unknown nodes,
    missing required inputs and model files that are not installed fail fast with a clear message.
  * One heavy job at a time: we wait for an empty queue and call /free whenever the model set
    changes (the queue serializes execution but does not release VRAM between clients). Consecutive
    jobs on the same models keep them loaded: reloading the 14 GB H3 UNET costs minutes per clip.
  * A job is only successful when its output file was downloaded and is non-empty.
    An exit code, an empty queue or a "success" status are not evidence.
  * Execution errors are raised with the node and the exception text; CUDA OOM is flagged
    so the caller can retry with a shorter clip.
  * Inputs are uploaded through /upload/image, so ComfyUI can live on another machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from opencine.core.config import settings

logger = logging.getLogger("OpenCine.ComfyClient")

Workflow = Dict[str, Dict[str, Any]]
OUTPUT_KINDS = ("images", "gifs", "videos", "audio")


class ComfyError(RuntimeError):
    """Base error for ComfyUI interactions."""


class ComfyUnavailable(ComfyError):
    """ComfyUI is not reachable or is only partially started."""


class WorkflowInvalid(ComfyError):
    """The workflow does not match the server's node definitions or installed models."""


class JobFailed(ComfyError):
    def __init__(self, message: str, oom: bool = False) -> None:
        super().__init__(message)
        self.oom = oom


class JobTimeout(ComfyError):
    """The job exceeded its time budget and was removed from the queue."""


class ComfyUIClient:
    """Async client for the ComfyUI REST API."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or settings.comfyui_url).rstrip("/")
        self.client_id = f"opencine-{uuid.uuid4().hex[:8]}"
        self._object_info: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()
        self._loaded_models: Optional[tuple] = None
        self.journal_path = settings.workspace_dir / "comfy_journal.jsonl"

    # ------------------------------------------------------------------ basics
    async def _get(self, path: str, timeout: float = 30.0, **params: Any) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(f"{self.base_url}{path}", params=params or None)

    async def _post(self, path: str, payload: Dict[str, Any], timeout: float = 60.0) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(f"{self.base_url}{path}", json=payload)

    async def is_alive(self) -> bool:
        try:
            return (await self._get("/system_stats", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def object_info(self, refresh: bool = False) -> Dict[str, Any]:
        if self._object_info is None or refresh:
            try:
                res = await self._get("/object_info", timeout=60.0)
                res.raise_for_status()
            except httpx.HTTPError as exc:
                raise ComfyUnavailable(f"ComfyUI not reachable at {self.base_url}: {exc}") from exc
            self._object_info = res.json()
        return self._object_info

    async def check_system(self, required_nodes: Iterable[str] = ()) -> Dict[str, Any]:
        """Health report: reachability, node count (partial start detection) and missing nodes."""
        if not await self.is_alive():
            return {"alive": False, "node_count": 0, "missing_nodes": list(required_nodes), "partial": False}
        info = await self.object_info(refresh=True)
        missing = [n for n in required_nodes if n not in info]
        partial = bool(settings.comfyui_min_nodes and len(info) < settings.comfyui_min_nodes)
        return {"alive": True, "node_count": len(info), "missing_nodes": missing, "partial": partial}

    async def free(self) -> None:
        self._loaded_models = None
        try:
            await self._post("/free", {"unload_models": True, "free_memory": True}, timeout=30.0)
        except httpx.HTTPError as exc:
            logger.warning(f"/free failed: {exc}")

    async def queue_state(self) -> Dict[str, List[Any]]:
        res = await self._get("/queue", timeout=10.0)
        res.raise_for_status()
        return res.json()

    async def wait_idle(self, timeout: float = 6 * 3600, poll: float = 15.0) -> None:
        """Waits until nobody else is rendering (other sessions share the GPU)."""
        start = time.monotonic()
        announced = False
        while time.monotonic() - start < timeout:
            try:
                q = await self.queue_state()
                if not q.get("queue_running") and not q.get("queue_pending"):
                    return
            except httpx.HTTPError as exc:
                raise ComfyUnavailable(f"ComfyUI stopped answering /queue: {exc}") from exc
            if not announced:
                logger.info("ComfyUI queue is busy with other jobs; waiting for it to empty")
                announced = True
            await asyncio.sleep(poll)
        raise JobTimeout("ComfyUI queue never became idle")

    # -------------------------------------------------------------- validation
    async def validate(self, workflow: Workflow) -> None:
        """Checks a workflow against /object_info. Raises WorkflowInvalid with every problem found."""
        info = await self.object_info()
        problems: List[str] = []
        for node_id, node in workflow.items():
            cls = node.get("class_type")
            spec = info.get(cls)
            if spec is None:
                problems.append(f"[{node_id}] node '{cls}' is not installed in ComfyUI")
                continue
            inputs = node.get("inputs", {})
            required = spec.get("input", {}).get("required", {})
            optional = spec.get("input", {}).get("optional", {})
            for name in required:
                if name not in inputs:
                    problems.append(f"[{node_id}] {cls}: missing required input '{name}'")
            for name, value in inputs.items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    if value[0] not in workflow:
                        problems.append(f"[{node_id}] {cls}.{name} links to missing node '{value[0]}'")
                    continue
                field = required.get(name) or optional.get(name)
                if field is None:
                    continue  # dynamic (autogrow / dynamic combo) inputs are not listed flat
                options = self._combo_options(field)
                if options is not None and value not in options:
                    if name.endswith(("_name", "_file")) or name in ("unet_name", "clip_name", "vae_name", "lora_name"):
                        problems.append(f"[{node_id}] model file not installed: '{value}' ({cls}.{name})")
                    else:
                        problems.append(f"[{node_id}] {cls}.{name}='{value}' is not one of {options[:8]}")
        if problems:
            raise WorkflowInvalid("Workflow rejected before queueing:\n  " + "\n  ".join(problems))

    @staticmethod
    def _combo_options(field: List[Any]) -> Optional[List[Any]]:
        if not field:
            return None
        head = field[0]
        if isinstance(head, list):
            return head
        if head == "COMBO" and len(field) > 1 and isinstance(field[1], dict):
            return field[1].get("options")
        return None

    async def available(self, node: str, input_name: str) -> List[str]:
        """Installed choices for a combo input, e.g. ('UNETLoader', 'unet_name')."""
        info = await self.object_info()
        spec = info.get(node, {}).get("input", {})
        field = spec.get("required", {}).get(input_name) or spec.get("optional", {}).get(input_name)
        return list(self._combo_options(field) or []) if field else []

    # ----------------------------------------------------------------- uploads
    async def upload(self, path: Path, subfolder: str = "") -> str:
        """Uploads an input file and returns the name LoadImage/LoadAudio expect."""
        path = Path(path)
        data = path.read_bytes()
        digest = hashlib.sha1(data).hexdigest()[:10]
        remote_name = f"opencine_{digest}_{path.name}"
        files = {"image": (remote_name, data)}
        form = {"type": "input", "subfolder": subfolder, "overwrite": "true"}
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(f"{self.base_url}/upload/image", files=files, data=form)
                res.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComfyUnavailable(f"Upload of {path.name} failed: {exc}") from exc
        body = res.json()
        sub = body.get("subfolder", subfolder)
        name = body.get("name", remote_name)
        full = f"{sub}/{name}" if sub else name
        if self._object_info is not None:
            self._register_input(full)
        return full

    def _register_input(self, name: str) -> None:
        """Adds a freshly uploaded file to cached combo lists so validation accepts it."""
        for node in ("LoadImage", "LoadAudio", "LoadVideo"):
            spec = (self._object_info or {}).get(node, {}).get("input", {}).get("required", {})
            for field in spec.values():
                options = self._combo_options(field)
                if options is not None and name not in options:
                    options.append(name)

    # -------------------------------------------------------------- execution
    async def queue(self, workflow: Workflow) -> str:
        try:
            res = await self._post("/prompt", {"prompt": workflow, "client_id": self.client_id})
        except httpx.HTTPError as exc:
            raise ComfyUnavailable(f"Could not queue prompt: {exc}") from exc
        if res.status_code != 200:
            raise WorkflowInvalid(f"ComfyUI rejected the prompt (HTTP {res.status_code}): {res.text[:2000]}")
        return res.json()["prompt_id"]

    async def wait(self, prompt_id: str, timeout: float) -> Dict[str, Any]:
        start = time.monotonic()
        unreachable_since: Optional[float] = None
        while time.monotonic() - start < timeout:
            await asyncio.sleep(3.0)
            try:
                res = await self._get(f"/history/{prompt_id}", timeout=30.0)
                unreachable_since = None
            except httpx.HTTPError:
                unreachable_since = unreachable_since or time.monotonic()
                if time.monotonic() - unreachable_since > 120:
                    raise ComfyUnavailable("ComfyUI stopped responding while rendering (crashed?)")
                continue
            entry = res.json().get(prompt_id) if res.status_code == 200 else None
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise self._job_error(status)
            if status.get("completed") or status.get("status_str") == "success":
                return entry
        await self._cancel(prompt_id)
        raise JobTimeout(f"Job {prompt_id} exceeded {timeout:.0f}s and was cancelled")

    @staticmethod
    def _job_error(status: Dict[str, Any]) -> JobFailed:
        for kind, msg in status.get("messages", []):
            if kind == "execution_error":
                text = f"{msg.get('node_type')} (node {msg.get('node_id')}): {msg.get('exception_message', '').strip()}"
                oom = "out of memory" in text.lower() or "outofmemory" in text.lower()
                return JobFailed(text, oom=oom)
        return JobFailed(json.dumps(status)[-1500:])

    async def _cancel(self, prompt_id: str) -> None:
        try:
            await self._post("/queue", {"delete": [prompt_id]})
            q = await self.queue_state()
            if any(item[1] == prompt_id for item in q.get("queue_running", [])):
                await self._post("/interrupt", {})
        except httpx.HTTPError:
            pass

    async def download_outputs(self, history: Dict[str, Any], dest_dir: Path,
                               kinds: Iterable[str] = OUTPUT_KINDS) -> List[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        async with httpx.AsyncClient(timeout=600.0) as client:
            for node_out in history.get("outputs", {}).values():
                for kind in kinds:
                    for item in node_out.get(kind, []):
                        if item.get("type", "output") != "output":
                            continue
                        params = {"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                  "type": "output"}
                        res = await client.get(f"{self.base_url}/view", params=params)
                        if res.status_code != 200 or not res.content:
                            raise JobFailed(f"Output {item['filename']} could not be downloaded")
                        target = dest_dir / item["filename"]
                        target.write_bytes(res.content)
                        paths.append(target)
        return paths

    async def run(self, workflow: Workflow, dest_dir: Path, label: str,
                  timeout: Optional[float] = None, suffixes: Iterable[str] = (),
                  expect_files: bool = True) -> List[Path]:
        """Validate -> wait for idle GPU -> queue -> wait -> download -> verify artifacts."""
        await self.validate(workflow)
        async with self._lock:
            await self.wait_idle()
            models = self._model_set(workflow)
            if settings.comfyui_free_between_jobs and models != self._loaded_models:
                await self.free()  # switching model families: release VRAM; same family stays loaded
            self._loaded_models = models
            t0 = time.time()
            prompt_id = await self.queue(workflow)
            logger.info(f"[{label}] queued prompt {prompt_id}")
            try:
                history = await self.wait(prompt_id, timeout or settings.comfyui_timeout)
                outputs = await self.download_outputs(history, dest_dir)
            except ComfyError as exc:
                self._journal(label, prompt_id, t0, "error", error=str(exc))
                raise
        wanted = tuple(suffixes)
        if not expect_files:
            self._journal(label, prompt_id, t0, "ok")
            return outputs
        if wanted:
            outputs = [p for p in outputs if p.suffix.lower() in wanted]
        if not outputs or any(p.stat().st_size == 0 for p in outputs):
            self._journal(label, prompt_id, t0, "error", error="no artifact")
            raise JobFailed(f"[{label}] finished without writing a {'/'.join(wanted) or 'file'} output")
        self._journal(label, prompt_id, t0, "ok", outputs=[str(p) for p in outputs])
        logger.info(f"[{label}] done in {(time.time() - t0) / 60:.1f} min -> {outputs[0].name}")
        return outputs

    @staticmethod
    def _model_set(workflow: Workflow) -> tuple:
        names = []
        for node in workflow.values():
            for key in ("unet_name", "clip_name", "vae_name", "lora_name"):
                if isinstance(node.get("inputs", {}).get(key), str):
                    names.append(node["inputs"][key])
        return tuple(sorted(names))

    def _journal(self, label: str, prompt_id: str, t0: float, status: str, **extra: Any) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "prompt_id": prompt_id,
               "status": status, "seconds": round(time.time() - t0, 1), **extra}
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


comfy_client = ComfyUIClient()
