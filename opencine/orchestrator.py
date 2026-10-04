"""OpenCineAgent Central Orchestrator.

[Script & Breakdown] -> [Identity & Stages] -> [Keyframe -> H3 render -> QC -> heal] -> [Cut, Score, Mix, Master]

Production order follows what worked on real projects: validate the plan and the identities
before rendering, render in storyboard order (an interrupted run still yields a continuous film),
keep every take on disk, and resume from state.json instead of re-rendering approved shots.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from opencine.agents import (
    AssemblyFinalCutAgent,
    AssetIdentityRegistryAgent,
    AudioMixingAgent,
    CriticVlmAgent,
    EnvironmentStageAgent,
    FoleySoundAgent,
    FrameStagingAgent,
    LoopbackHealingAgent,
    MusicScoringAgent,
    ScriptwriterAgent,
    ShotlistPromptAgent,
    SuperResolutionAgent,
    VideoSynthesisAgent,
    VoiceDialogueAgent,
)
from opencine.core import media
from opencine.core.comfy_client import ComfyUIClient, ComfyUnavailable, JobFailed, comfy_client
from opencine.core.config import settings
from opencine.core.models import (
    CheckStatus,
    ProductionResult,
    ProductionScript,
    QCReport,
    SceneScript,
    ShotRenderResult,
    ShotSpecification,
)

logger = logging.getLogger("OpenCine.Orchestrator")
Progress = Optional[Callable[[str, float], None]]


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "production"


def prompt_hash(shot: ShotSpecification) -> str:
    key = f"{shot.keyframe_prompt}|{shot.video_prompt}|{shot.length_frames}|{shot.voice_track}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


class OpenCineOrchestrator:
    def __init__(self, client: Optional[ComfyUIClient] = None, registry_file: Optional[Path] = None) -> None:
        self.comfy = client or comfy_client
        self.registry_agent = AssetIdentityRegistryAgent(registry_file, client=self.comfy)
        self.scriptwriter_agent = ScriptwriterAgent()
        self.shotlist_agent = ShotlistPromptAgent(self.registry_agent)
        self.stage_agent = EnvironmentStageAgent(self.comfy)
        self.framestaging_agent = FrameStagingAgent(self.registry_agent, self.comfy)
        self.video_agent = VideoSynthesisAgent(self.registry_agent, self.comfy)
        self.critic_agent = CriticVlmAgent(self.registry_agent)
        self.healing_agent = LoopbackHealingAgent()
        self.super_res_agent = SuperResolutionAgent()
        self.dialogue_agent = VoiceDialogueAgent()
        self.foley_agent = FoleySoundAgent()
        self.music_agent = MusicScoringAgent(self.comfy)
        self.mixing_agent = AudioMixingAgent()
        self.assembly_agent = AssemblyFinalCutAgent()

    # ---------------------------------------------------------------- preflight
    def required_nodes(self, script: ProductionScript) -> List[str]:
        nodes = ["UNETLoader", "CLIPLoader", "VAELoader", "KSampler", "SaveImage", "LoadImage",
                 "MiniMaxH3ReferenceToVideo", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
                 "Krea2EditModelPatch", "Krea2EditGroundedEncode", "EmptySD3LatentImage"]
        if settings.h3_sla:
            nodes.append("H3SLAAttention")
        shots = script.all_shots()
        if any(line.audio_path for s in shots for line in s.dialogue):
            nodes += ["LoadAudio", "PixaromaH3AudioSync"]
        cast = {cid for s in shots for cid in s.character_ids}
        if any((c := self.registry_agent.get_character(cid)) and c.refmod_name for cid in cast):
            nodes += ["MiniMaxH3RefModsLoader", "MiniMaxH3RefModApply"]
        if settings.music_mode == "minimax":
            nodes += ["MiniMaxMusic3TextEncode", "EmptyMiniMaxMusic3LatentAudio", "SaveAudio"]
        return nodes

    def required_models(self) -> List[Tuple[str, str, str]]:
        models = [
            ("UNETLoader", "unet_name", settings.krea2_unet),
            ("CLIPLoader", "clip_name", settings.krea2_clip),
            ("VAELoader", "vae_name", settings.krea2_vae),
            ("LoraLoaderModelOnly", "lora_name", settings.krea2_identity_edit_lora),
            ("UNETLoader", "unet_name", settings.h3_unet),
            ("CLIPLoader", "clip_name", settings.h3_clip),
            ("VAELoader", "vae_name", settings.h3_video_vae),
            ("VAELoader", "vae_name", settings.h3_audio_vae),
        ]
        if settings.h3_turbo_lora:
            models.append(("LoraLoaderModelOnly", "lora_name", settings.h3_turbo_lora))
        if settings.music_mode == "minimax":
            models += [("UNETLoader", "unet_name", settings.music3_unet),
                       ("CLIPLoader", "clip_name", settings.music3_clip),
                       ("VAELoader", "vae_name", settings.music3_vae)]
        for c in self.registry_agent.list_characters():
            if c.lora_name and not c.approved:
                models.append(("LoraLoaderModelOnly", "lora_name", c.lora_name))
        return models

    async def preflight(self, script: ProductionScript) -> None:
        report = await self.comfy.check_system(self.required_nodes(script))
        if not report["alive"]:
            raise ComfyUnavailable(f"ComfyUI is not running at {settings.comfyui_url}")
        if report["partial"]:
            raise ComfyUnavailable(f"ComfyUI started partially ({report['node_count']} nodes): restart it cleanly")
        problems = [f"node not installed: {n}" for n in report["missing_nodes"]]
        for node, field, value in self.required_models():
            if value not in await self.comfy.available(node, field):
                problems.append(f"model not installed: {value} ({node}.{field})")
        if problems:
            raise ComfyUnavailable("ComfyUI is missing what this production needs (see README > Models):\n  "
                                   + "\n  ".join(dict.fromkeys(problems)))

    # --------------------------------------------------------------------- plan
    async def plan(self, premise: Optional[str] = None, script_path: Optional[Path] = None,
                   character_ids: Optional[List[str]] = None, language: Optional[str] = None,
                   shots: int = 4) -> Tuple[ProductionScript, Path]:
        """Writes plan.json (script + every compiled prompt) for review before any GPU time is spent."""
        if script_path:
            script = self.scriptwriter_agent.load_script(script_path)
            if language:
                script.language = language
        elif premise:
            cast = self.registry_agent.cast(character_ids or [c.id for c in self.registry_agent.list_characters()])
            script = await self.scriptwriter_agent.execute(premise, cast, language, shots)
        else:
            raise ValueError("Provide a premise or a script file")
        await self.foley_agent.execute(script)
        await self.shotlist_agent.execute(script)
        prod_dir = settings.production_dir(slugify(script.title))
        (prod_dir / "plan.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
        return script, prod_dir

    # ------------------------------------------------------------------ shots
    def _state(self, prod_dir: Path) -> Dict[str, dict]:
        path = prod_dir / "state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _save_state(self, prod_dir: Path, state: Dict[str, dict]) -> None:
        (prod_dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    async def prepare_shot(self, shot: ShotSpecification, scene: SceneScript, script: ProductionScript,
                           prod_dir: Path, force: bool = False) -> Path:
        """Stage plate, voice track and keyframe (all Krea 2 / CPU work for the shot)."""
        location = self.shotlist_agent.location_line(scene)
        stage = await self.stage_agent.execute(shot.stage_id, scene, prod_dir, script.visual_style)
        await self.dialogue_agent.execute(shot, prod_dir)
        return await self.framestaging_agent.execute(shot, stage, prod_dir, location, script.visual_style, force)

    async def render_and_heal_shot(self, shot: ShotSpecification, scene: SceneScript, script: ProductionScript,
                                   prod_dir: Path, force: bool = False) -> ShotRenderResult:
        """Keyframe -> H3 -> QC, retried with new seeds (or fewer frames on OOM); best take wins."""
        keyframe = await self.prepare_shot(shot, scene, script, prod_dir, force)

        takes: List[Tuple[QCReport, Path]] = []
        attempt = 0
        while True:
            attempt += 1
            try:
                clip = await self.video_agent.execute(shot, keyframe, prod_dir, attempt)
            except JobFailed as exc:
                if exc.oom and self.healing_agent.shorten_for_oom(shot):
                    await self.comfy.free()
                    continue
                raise
            report = await self.critic_agent.execute(shot, clip, prod_dir, attempt)
            takes.append((report, clip))
            logger.info(f"{shot.shot_id} take {attempt}: {'PASS' if report.pass_qc else 'FAIL ' + str(report.error_codes)}"
                        + (f" (unverified: {', '.join(report.unverified)})" if report.unverified else ""))
            if report.pass_qc:
                break
            if self.healing_agent.same_failure_twice([t[0] for t in takes]):
                logger.warning(f"{shot.shot_id}: same failure on two seeds; the prompt needs a human look")
                break
            if await self.healing_agent.execute(shot, report) is None:
                break

        best_report, best_clip = self.healing_agent.best_attempt(takes)
        return ShotRenderResult(
            shot_id=shot.shot_id, keyframe_path=str(keyframe), raw_video_path=str(best_clip),
            qc_reports=[t[0] for t in takes], approved=best_report.pass_qc, attempts=len(takes), seed=shot.seed,
        )

    # ---------------------------------------------------------------- pipeline
    async def run_pipeline(self, premise: Optional[str] = None, script_path: Optional[Path] = None,
                           character_ids: Optional[List[str]] = None, language: Optional[str] = None,
                           shots: int = 4, only: Optional[List[str]] = None, auto_approve: bool = False,
                           music_file: Optional[Path] = None, burn_subtitles: bool = False, force: bool = False,
                           progress_callback: Progress = None) -> ProductionResult:
        def progress(msg: str, pct: float) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg, pct)

        progress("Writing and compiling the shooting script...", 0.02)
        script, prod_dir = await self.plan(premise, script_path, character_ids, language, shots)
        progress(f"Plan saved to {prod_dir / 'plan.json'}; checking ComfyUI...", 0.06)
        targets = {s.shot_id for s in script.all_shots() if not only or s.shot_id in only}
        if only and len(targets) != len(set(only)):
            raise ValueError(f"Unknown shot id(s): {sorted(set(only) - targets)}")
        await self.preflight(script)
        cast_ids = list(dict.fromkeys(cid for s in script.all_shots() if s.shot_id in targets
                                      for cid in s.character_ids))
        await self.registry_agent.ensure_ready(cast_ids, auto_approve=auto_approve)

        state = self._state(prod_dir)
        results: List[ShotRenderResult] = []
        approved_clips: List[Tuple[ShotSpecification, Path]] = []
        ordered = [(scene, shot) for scene in script.scenes for shot in scene.shots if shot.shot_id in targets]

        def is_done(shot: ShotSpecification) -> bool:
            cached = state.get(shot.shot_id)
            return bool(cached and not force and cached.get("prompt_hash") == prompt_hash(shot)
                        and Path(cached["result"].get("raw_video_path") or "").exists())

        # Phase 1: every keyframe with Krea 2, so H3 then stays loaded for all the clips.
        pending = [(scene, shot) for scene, shot in ordered if not is_done(shot)]
        for i, (scene, shot) in enumerate(pending, 1):
            progress(f"Keyframe {shot.shot_id} [{i}/{len(pending)}]", 0.08 + 0.12 * (i - 1) / max(1, len(pending)))
            await self.prepare_shot(shot, scene, script, prod_dir, force)

        # Phase 2: H3 clips in storyboard order, with QC and healing (H3 stays loaded between clips).
        for done, (scene, shot) in enumerate(ordered, 1):
            progress(f"Shot {shot.shot_id} [{done}/{len(targets)}]", 0.2 + 0.55 * (done - 1) / len(targets))
            if is_done(shot):
                result = ShotRenderResult(**state[shot.shot_id]["result"])
                logger.info(f"{shot.shot_id}: already rendered, reusing {result.raw_video_path}")
            else:
                result = await self.render_and_heal_shot(shot, scene, script, prod_dir, force)
                state[shot.shot_id] = {"prompt_hash": prompt_hash(shot), "result": result.model_dump(mode="json"),
                                       "rendered_at": time.strftime("%F %T")}
                self._save_state(prod_dir, state)
            results.append(result)

        # Phase 3: upscale the chosen takes (needs the VRAM H3 was holding).
        progress("Upscaling...", 0.78)
        for (scene, shot), result in zip(ordered, results):
            final = Path(result.final_video_path) if result.final_video_path else None
            if force or final is None or not final.is_file():
                final = await self.super_res_agent.execute(Path(result.raw_video_path), prod_dir, shot.shot_id)
                result.final_video_path = str(final)
                state[shot.shot_id]["result"] = result.model_dump(mode="json")
                self._save_state(prod_dir, state)
            approved_clips.append((shot, final))

        progress("Cutting the picture...", 0.82)
        picture, stem, total = await self.assembly_agent.execute(approved_clips, prod_dir)
        progress("Scoring and mixing...", 0.88)
        music = await self.music_agent.execute(script, total, prod_dir, music_file)
        master_audio = await self.mixing_agent.execute(stem, music, total, prod_dir / "audio" / "master.wav")
        progress("Mastering...", 0.95)
        slug = slugify(script.title)
        srt = self.assembly_agent.generate_srt_subtitles([s for s, _ in approved_clips], prod_dir / f"{slug}.srt")
        size = media.probe(picture)
        overlay = self.assembly_agent.generate_text_overlay([s for s, _ in approved_clips], prod_dir / "on_screen_text.ass",
                                                             int(size["width"]), int(size["height"]))
        final = await self.assembly_agent.master(picture, master_audio, srt, total, prod_dir, slug, burn_subtitles,
                                                 overlay)

        passed = sum(r.approved for r in results)
        result = ProductionResult(
            production_id=slug, title=script.title, final_master_path=str(final), subtitles_srt_path=str(srt),
            shot_results=results, total_duration_sec=total, qc_pass_rate=passed / max(1, len(results)),
        )
        self.write_report(result, prod_dir)
        progress("Production complete", 1.0)
        return result

    @staticmethod
    def write_report(result: ProductionResult, prod_dir: Path) -> Path:
        """Human-readable QC report: what passed, what failed, what could not be verified."""
        lines = [f"# {result.title}", "", f"Master: `{result.final_master_path}`",
                 f"Duration: {result.total_duration_sec:.1f}s, QC pass rate {result.qc_pass_rate:.0%}", "",
                 "| Shot | Takes | Status | Failed checks | Not verified |", "|---|---|---|---|---|"]
        for r in result.shot_results:
            best = next((q for q in r.qc_reports if q.pass_qc), r.qc_reports[-1] if r.qc_reports else None)
            failed = ", ".join(c.name for c in best.checks if c.status == CheckStatus.FAILED) if best else "-"
            skipped = ", ".join(best.unverified) if best else "-"
            lines.append(f"| {r.shot_id} | {r.attempts} | {'approved' if r.approved else 'FLAGGED'} | "
                         f"{failed or '-'} | {skipped or '-'} |")
        path = prod_dir / "REPORT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (prod_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path
