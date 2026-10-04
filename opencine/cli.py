"""OpenCineAgent Command-Line Interface."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import httpx
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from opencine.agents.critic_vlm import CriticVlmAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.catalog import CATALOG, CUSTOM_NODES
from opencine.core.comfy_client import ComfyError, comfy_client
from opencine.core.config import settings
from opencine.core.models import CheckStatus, ProductionScript
from opencine.orchestrator import OpenCineOrchestrator

app = typer.Typer(name="opencine", add_completion=False,
                  help="OpenCineAgent: 14-agent film pipeline on local open models (MiniMax H3 + Krea 2 via ComfyUI)")
console = Console()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s",
                        handlers=[RichHandler(console=console, show_path=False, markup=False)])
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _ok(flag: bool, yes: str = "OK", no: str = "MISSING") -> str:
    return f"[green]{yes}[/green]" if flag else f"[red]{no}[/red]"


def _fail(exc: Exception) -> None:
    console.print(Panel(Text(str(exc)), title=type(exc).__name__, border_style="red"))
    raise typer.Exit(code=1)


@app.command("doctor")
def doctor() -> None:
    """Checks ComfyUI (nodes and models), Ollama, FFmpeg and optional QC tools."""
    orchestrator = OpenCineOrchestrator()

    async def gather():
        report = await comfy_client.check_system(orchestrator.required_nodes(ProductionScript()))
        installed = {}
        if report["alive"]:
            for node, field, value in orchestrator.required_models():
                installed[value] = value in await comfy_client.available(node, field)
        return report, installed

    report, installed = asyncio.run(gather())
    table = Table(title="OpenCineAgent environment")
    table.add_column("Component", style="cyan")
    table.add_column("Detail")
    table.add_column("Status")
    table.add_row("ComfyUI", f"{settings.comfyui_url} ({report['node_count']} nodes)",
                  _ok(report["alive"] and not report["partial"], "ONLINE", "OFFLINE/PARTIAL"))
    for node in report["missing_nodes"]:
        table.add_row("  node", node, _ok(False))
    for value, ok in installed.items():
        table.add_row("  model", value, _ok(ok))
    try:
        tags = [m["name"] for m in httpx.get(f"{settings.ollama_url}/api/tags", timeout=2).json()["models"]]
    except (httpx.HTTPError, KeyError, ValueError):
        tags = None
    table.add_row("Ollama", settings.ollama_url, _ok(tags is not None, "ONLINE", "OFFLINE"))
    for model, role in ((settings.llm_scriptwriter_model, "scriptwriter"), (settings.vlm_reviewer_model, "QC reviewer")):
        present = bool(tags) and any(t == model or t.startswith(model + ":") for t in tags)
        table.add_row(f"  {role}", model, _ok(present, "OK", f"ollama pull {model}"))
    table.add_row("FFmpeg", str(shutil.which("ffmpeg") or "-"), _ok(bool(shutil.which("ffmpeg"))))
    table.add_row("faster-whisper", "dialogue QC", _ok(importlib.util.find_spec("faster_whisper") is not None,
                                                       "OK", "pip install -e '.\\[qc]'"))
    console.print(table)


app.command("healthcheck", hidden=True)(doctor)


@app.command("models")
def models() -> None:
    """Lists the open model files, their ComfyUI folder and download link."""
    async def present() -> set:
        if not await comfy_client.is_alive():
            return set()
        names = set()
        for node, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"), ("VAELoader", "vae_name"),
                            ("LoraLoaderModelOnly", "lora_name")):
            names.update(await comfy_client.available(node, field))
        return names

    have = asyncio.run(present())
    table = Table(title="Models (place each file in ComfyUI/models/<folder>/)")
    for col in ("Model", "File", "Folder", "Installed", "Download"):
        table.add_column(col)
    for m in CATALOG:
        state = "-" if not have else ("yes" if m.filename in have else ("no" if m.required else "optional"))
        table.add_row(m.group, m.filename, m.folder, state, m.url)
    console.print(table)
    nodes = Table(title="Custom nodes")
    for col in ("Package", "Used for", "Required", "Repository"):
        nodes.add_column(col)
    for name, url, purpose, required in CUSTOM_NODES:
        nodes.add_row(name, purpose, "yes" if required else "optional", url)
    console.print(nodes)


@app.command("characters")
def list_characters() -> None:
    """Shows the cast registry and the state of each identity anchor."""
    registry = AssetIdentityRegistryAgent()
    table = Table(title=f"Cast ({registry.registry_file})")
    for col in ("ID", "Name", "Master portrait", "LoRA", "RefMod"):
        table.add_column(col)
    for c in registry.list_characters():
        portrait = registry.reference_path(c)
        status = ("approved" if c.approved else "pending review") if portrait and portrait.exists() else "none"
        table.add_row(c.id, c.name, status, c.lora_name or "-", c.refmod_name or "-")
    console.print(table)


@app.command("cast")
def cast(character_id: str, candidates: int = typer.Option(2, help="Seeds to render")) -> None:
    """Renders master portrait candidates for a character (Krea 2)."""
    registry = AssetIdentityRegistryAgent()
    try:
        paths = asyncio.run(registry.generate_portraits(character_id, candidates))
    except (ComfyError, KeyError) as exc:
        _fail(exc)
    for p in paths:
        console.print(f"  {p}")
    console.print(f"Look at them, then: [bold]python main.py approve {character_id} <image>[/bold]")


@app.command("approve")
def approve(character_id: str, image: Path) -> None:
    """Approves a master portrait after a human checked it is the right person."""
    if not image.exists():
        _fail(FileNotFoundError(image))
    profile = AssetIdentityRegistryAgent().approve(character_id, image)
    console.print(f"[green]{profile.name}[/green] master portrait -> {profile.reference_image}")


@app.command("refmod")
def refmod(character_id: str, images: Optional[List[Path]] = typer.Argument(
        None, help="Face images (up to 8 varied angles). Default: the approved master portrait")) -> None:
    """Extracts a MiniMax H3 RefMod identity anchor (needs ComfyUI-MiniMaxH3Mod)."""
    registry = AssetIdentityRegistryAgent()
    profile = registry.cast([character_id])[0]
    sources = images or [p for p in [registry.reference_path(profile)] if p and p.exists()]
    try:
        name = asyncio.run(registry.extract_refmod(character_id, sources))
    except (ComfyError, ValueError) as exc:
        _fail(exc)
    console.print(f"[green]RefMod ready:[/green] {name} (used only in shots where {profile.name} is alone)")


def _source(premise: Optional[str], script: Optional[Path]) -> None:
    if not premise and not script:
        _fail(ValueError("Pass --premise or --script"))


@app.command("plan")
def plan(premise: Optional[str] = typer.Option(None, "--premise", "-p"),
         script: Optional[Path] = typer.Option(None, "--script", "-s", help="Hand-written script JSON"),
         characters: Optional[List[str]] = typer.Option(None, "--character", "-c"),
         language: Optional[str] = typer.Option(None, help="Dialogue language, e.g. Spanish"),
         shots: int = typer.Option(4, help="Target number of shots when writing from a premise")) -> None:
    """Writes and compiles the shooting script without rendering (review prompts first)."""
    _source(premise, script)
    try:
        result, prod_dir = asyncio.run(OpenCineOrchestrator().plan(premise, script, characters, language, shots))
    except Exception as exc:  # noqa: BLE001 - surface any planning error to the user
        _fail(exc)
    table = Table(title=f"{result.title} - {len(result.all_shots())} shots")
    for col in ("Shot", "Type", "Cast", "Seconds", "Frames", "Dialogue"):
        table.add_column(col)
    for s in result.all_shots():
        table.add_row(s.shot_id, s.shot_type.value, ", ".join(s.character_ids) or "-", f"{s.duration_sec:g}",
                      str(s.length_frames), " / ".join(d.text for d in s.dialogue)[:70] or "-")
    console.print(table)
    console.print(f"Plan with every prompt: [bold]{prod_dir / 'plan.json'}[/bold]\n"
                  f"Render it with: python main.py run --script {prod_dir / 'plan.json'}")


@app.command("run")
def run_production(
    premise: Optional[str] = typer.Option(None, "--premise", "-p"),
    script: Optional[Path] = typer.Option(None, "--script", "-s", help="Script or plan.json"),
    characters: Optional[List[str]] = typer.Option(None, "--character", "-c"),
    language: Optional[str] = typer.Option(None),
    shots: int = typer.Option(4),
    only: Optional[List[str]] = typer.Option(None, "--only", help="Render only these shot ids (e.g. the hardest one)"),
    music: Optional[Path] = typer.Option(None, help="Score file you own or licensed"),
    burn_subtitles: bool = typer.Option(False, "--burn-subtitles"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Accept master portraits without review"),
    force: bool = typer.Option(False, "--force", help="Re-render shots even if state.json has them"),
) -> None:
    """Runs the full pipeline: script -> identity -> keyframes -> H3 -> QC -> cut -> mix -> master."""
    _source(premise, script)
    orchestrator = OpenCineOrchestrator()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Starting...", total=100)
        try:
            result = asyncio.run(orchestrator.run_pipeline(
                premise=premise, script_path=script, character_ids=characters, language=language, shots=shots,
                only=only, auto_approve=auto_approve, music_file=music, burn_subtitles=burn_subtitles, force=force,
                progress_callback=lambda msg, pct: progress.update(task, description=msg, completed=int(pct * 100)),
            ))
        except Exception as exc:  # noqa: BLE001
            progress.stop()
            _fail(exc)
    flagged = [r.shot_id for r in result.shot_results if not r.approved]
    console.print(Panel(
        f"Master: [bold]{result.final_master_path}[/bold]\nSubtitles: {result.subtitles_srt_path}\n"
        f"Duration: {result.total_duration_sec:.1f}s  Shots: {len(result.shot_results)}  "
        f"QC pass rate: {result.qc_pass_rate:.0%}\n"
        + (f"[yellow]Flagged shots (best take kept): {', '.join(flagged)}[/yellow]\n" if flagged else "")
        + f"Report: {Path(result.final_master_path).parent / 'REPORT.md'}",
        title=result.title, border_style="green" if not flagged else "yellow"))


@app.command("qc")
def inspect_video(clip: Path, plan_file: Path = typer.Option(..., "--plan", help="plan.json of the production"),
                  shot_id: str = typer.Option(..., "--shot")) -> None:
    """Runs the QC agent on an existing clip against its shot in a plan."""
    script = ProductionScript.model_validate_json(plan_file.read_text(encoding="utf-8"))
    shot = next((s for s in script.all_shots() if s.shot_id == shot_id), None)
    if not shot or not clip.exists():
        _fail(ValueError(f"Shot {shot_id} not in plan, or clip missing"))
    report = asyncio.run(CriticVlmAgent().execute(shot, clip, plan_file.parent, attempt=0))
    table = Table(title=f"QC {shot_id}: {'PASSED' if report.pass_qc else 'REJECTED'}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    colors = {CheckStatus.PASSED: "green", CheckStatus.FAILED: "red", CheckStatus.SKIPPED: "yellow"}
    for c in report.checks:
        table.add_row(c.name, f"[{colors[c.status]}]{c.status.value}[/]", c.detail)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
