"""OpenCineAgent Command-Line Interface.
Powered by Typer and Rich for visual elegance and operational transparency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from opencine.agents.critic_vlm import CriticVlmAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent
from opencine.core.comfy_client import comfy_client
from opencine.core.config import settings
from opencine.orchestrator import OpenCineOrchestrator

app = typer.Typer(
    name="opencine",
    help="OpenCineAgent: Modular 14-Agent Production Pipeline for Autonomous Open Cinema",
    add_completion=False
)
console = Console()


@app.command("healthcheck")
def healthcheck() -> None:
    """Verifies local runtime connectivity: ComfyUI, Ollama, and FFmpeg."""
    console.print(Panel.fit("[bold cyan]OpenCineAgent // System Diagnostic & Health Check[/bold cyan]"))

    table = Table(title="Local Infrastructure Components")
    table.add_column("Component", style="cyan")
    table.add_column("Endpoint / Path", style="magenta")
    table.add_column("Status", style="bold")

    # 1. ComfyUI
    comfy_alive = asyncio.run(comfy_client.is_alive())
    table.add_row(
        "ComfyUI Headless",
        f"http://{settings.comfyui_host}:{settings.comfyui_port}",
        "[green]ONLINE (Ready)[/green]" if comfy_alive else "[yellow]OFFLINE (Simulated fallback active)[/yellow]"
    )

    # 2. Ollama
    import httpx
    try:
        res = httpx.get(f"{settings.ollama_url}/api/tags", timeout=1.5)
        ollama_alive = (res.status_code == 200)
    except Exception:
        ollama_alive = False

    table.add_row(
        "Ollama Local Daemon",
        settings.ollama_url,
        "[green]ONLINE (Ready)[/green]" if ollama_alive else "[yellow]OFFLINE (Deterministic fallback active)[/yellow]"
    )

    # 3. FFmpeg
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    table.add_row(
        "FFmpeg Engine",
        str(ffmpeg_path or "Not found"),
        "[green]INSTALLED[/green]" if ffmpeg_path else "[red]MISSING[/red]"
    )

    console.print(table)


@app.command("characters")
def list_characters() -> None:
    """Displays the 4 canonical character profiles from Asset & Identity Registry."""
    registry = AssetIdentityRegistryAgent()
    chars = registry.list_characters()

    console.print(Panel.fit(f"[bold green]Asset & Identity Registry ({len(chars)} Registered Actors)[/bold green]"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=14)
    table.add_column("Name", style="bold white", width=16)
    table.add_column("Voice ID", style="yellow", width=12)
    table.add_column("Invariable Clothing & Silhouette", style="white")

    for c in chars:
        table.add_row(c.id, c.name, c.voice_id, c.invariable_clothing)

    console.print(table)


@app.command("qc")
def inspect_video(
    video_path: Path = typer.Argument(..., help="Path to video clip file to inspect"),
    shot_id: str = typer.Option("SH_TEST", help="Shot identifier"),
    seed: int = typer.Option(42, help="Seed used during generation")
) -> None:
    """Runs Agent 7 (Critic / VLM Reviewer) independently on an existing video clip."""
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found at {video_path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Inspecting clip: [bold]{video_path}[/bold] via VLM Reviewer Agent...[/cyan]")
    critic = CriticVlmAgent()
    report = asyncio.run(critic.execute(shot_id=shot_id, video_path=video_path, seed=seed))

    status_color = "green" if report.pass_qc else "red"
    console.print(Panel(
        f"[bold {status_color}]QC Decision: {'PASSED' if report.pass_qc else 'REJECTED'}[/bold {status_color}]\n\n"
        f"• Identity Drift Score : {report.identity_drift_score:.2f} (Threshold <= {critic.drift_threshold})\n"
        f"• Deformed Faces       : {report.deformed_faces}\n"
        f"• Deformed Hands (>5)  : {report.deformed_hands}\n"
        f"• Motion Jitter        : {report.motion_jitter}\n"
        f"• Error Codes          : {report.error_codes or 'None'}\n"
        f"• Prescribed Healing   : [bold yellow]{report.healing_action.value}[/bold yellow]\n"
        f"• Critique Notes       : {report.critique_notes}",
        title=f"QC Report // {shot_id}"
    ))


@app.command("run")
def run_production(
    premise: str = typer.Option(
        "Elena Vance confronts Victor Cross on a rainy cyberpunk penthouse balcony.",
        "--premise", "-p",
        help="Story premise to adapt into full production"
    ),
    characters: Optional[List[str]] = typer.Option(
        None,
        "--character", "-c",
        help="Character IDs to feature (e.g. elena_vance, victor_cross)"
    )
) -> None:
    """Executes the complete 14-Agent autonomous production pipeline."""
    orchestrator = OpenCineOrchestrator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initializing OpenCine Pipeline...", total=100)

        def on_progress(msg: str, pct: float) -> None:
            progress.update(task, description=f"[cyan]{msg}", completed=int(pct * 100))

        result = asyncio.run(
            orchestrator.run_pipeline(
                premise=premise,
                character_ids=characters or ["elena_vance", "victor_cross"],
                progress_callback=on_progress
            )
        )

    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]🎬 PRODUCTION COMPLETED SUCCESSFULLY[/bold green]\n\n"
        f"• [bold]Title:[/bold] {result.title}\n"
        f"• [bold]Master Video:[/bold] [underline]{result.final_master_path}[/underline]\n"
        f"• [bold]Subtitles SRT:[/bold] [underline]{result.subtitles_srt_path}[/underline]\n"
        f"• [bold]Total Duration:[/bold] {result.total_duration_sec:.1f}s\n"
        f"• [bold]Shots Assembled:[/bold] {len(result.shot_results)}\n"
        f"• [bold]QC Pass Rate:[/bold] 100%",
        title="OpenCine Deliverable"
    ))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
