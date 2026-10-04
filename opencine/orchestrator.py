"""OpenCineAgent Central Orchestrator.
Coordinates the modular 14-agent pipeline across the 4 functional layers:
[Script & Breakdown] -> [Generation & Consistency] -> [QC Filter & Healing] -> [Audio & Final Cut]
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from rich.console import Console

from opencine.agents import (
    ScriptwriterAgent,
    ShotlistPromptAgent,
    AssetIdentityRegistryAgent,
    EnvironmentStageAgent,
    FrameStagingAgent,
    VideoSynthesisAgent,
    CriticVlmAgent,
    LoopbackHealingAgent,
    SuperResolutionAgent,
    VoiceDialogueAgent,
    FoleySoundAgent,
    MusicScoringAgent,
    AudioMixingAgent,
    AssemblyFinalCutAgent
)
from opencine.core.config import settings
from opencine.core.events import EventBus, PipelineEvent, event_bus
from opencine.core.models import (
    AudioTrack,
    HealingAction,
    ProductionResult,
    ProductionScript,
    QCReport,
    ShotRenderResult,
    ShotSpecification
)

console = Console()
logger = logging.getLogger("OpenCine.Orchestrator")


class OpenCineOrchestrator:
    """Master orchestrator executing the 14-agent cinematic pipeline."""

    def __init__(self, bus: EventBus = event_bus) -> None:
        self.bus = bus
        self.logger = logger

        # Layer A: Preproduction & Breakdown
        self.registry_agent = AssetIdentityRegistryAgent()
        self.scriptwriter_agent = ScriptwriterAgent()
        self.shotlist_agent = ShotlistPromptAgent(self.registry_agent)

        # Layer B: Generation & Spatial Consistency
        self.stage_agent = EnvironmentStageAgent()
        self.framestaging_agent = FrameStagingAgent(self.registry_agent)
        self.video_agent = VideoSynthesisAgent()

        # Layer C: Quality Control & Healing
        self.critic_vlm_agent = CriticVlmAgent()
        self.healing_agent = LoopbackHealingAgent()
        self.super_res_agent = SuperResolutionAgent(target_fps=settings.upscale_target_fps)

        # Layer D: Audio, Subtitles & Final Cut
        self.dialogue_agent = VoiceDialogueAgent(self.registry_agent)
        self.foley_agent = FoleySoundAgent()
        self.music_agent = MusicScoringAgent()
        self.mixing_agent = AudioMixingAgent(ducking_db=settings.ducking_attenuation_db)
        self.assembly_agent = AssemblyFinalCutAgent()

    async def render_and_heal_shot(
        self,
        shot: ShotSpecification,
        max_retries: int = settings.max_qc_retries
    ) -> Tuple[ShotSpecification, Path, List[QCReport]]:
        """Executes Layer B & Layer C for a single shot with an automated QC feedback loop."""
        qc_history: List[QCReport] = []
        attempt = 0
        current_shot = shot

        # 1. Lock environment stage
        stage = await self.stage_agent.get_or_create_stage(
            current_shot.stage_id,
            f"Cinematic location for {current_shot.shot_id}"
        )

        while attempt <= max_retries:
            attempt += 1
            self.logger.info(f"--- [Shot {current_shot.shot_id}] Render Attempt {attempt}/{max_retries + 1} (Seed: {current_shot.seed}) ---")

            # 2. Keyframe Staging
            keyframe_path = await self.framestaging_agent.execute(current_shot, stage)

            # 3. Video Synthesis
            raw_video = await self.video_agent.execute(current_shot, keyframe_path)

            # 4. Critic / VLM Reviewer
            qc_report = await self.critic_vlm_agent.execute(
                current_shot.shot_id, raw_video, seed=current_shot.seed
            )
            qc_history.append(qc_report)

            if qc_report.pass_qc:
                self.logger.info(f"✓ Shot {current_shot.shot_id} PASSED QC on attempt {attempt}")
                # 5. Super-Resolution & 60 fps Frame Rate
                master_clip = await self.super_res_agent.execute(raw_video, current_shot.shot_id)
                return current_shot, master_clip, qc_history

            # QC Failed: Execute Loopback Healing
            self.logger.warning(
                f"✗ Shot {current_shot.shot_id} REJECTED by VLM Reviewer. "
                f"Action: {qc_report.healing_action.value}. Errors: {qc_report.error_codes}"
            )

            if qc_report.healing_action == HealingAction.FACE_DETAILER:
                # Mild issue: FaceDetailer / SAM inpainting
                healed_clip, _ = await self.healing_agent.execute(current_shot, raw_video, qc_report)
                if healed_clip:
                    # Run post-healing QC check
                    post_qc = await self.critic_vlm_agent.execute(current_shot.shot_id, healed_clip, seed=current_shot.seed)
                    qc_history.append(post_qc)
                    if post_qc.pass_qc:
                        self.logger.info(f"✓ Shot {current_shot.shot_id} restored by FaceDetailer!")
                        master_clip = await self.super_res_agent.execute(healed_clip, current_shot.shot_id)
                        return current_shot, master_clip, qc_history

            # Critical error or unhealed: Mutate shot parameters for retry
            _, mutated_shot = await self.healing_agent.execute(current_shot, raw_video, qc_report)
            current_shot = mutated_shot

        # Fallback if max retries exceeded: use best effort
        self.logger.error(f"Shot {current_shot.shot_id} reached max retries. Using latest render with upscaling.")
        master_clip = await self.super_res_agent.execute(raw_video, current_shot.shot_id)
        return current_shot, master_clip, qc_history

    async def run_pipeline(
        self,
        premise: str,
        character_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ProductionResult:
        """Runs the entire end-to-end 14-agent cinematic pipeline."""
        self.logger.info(f"Starting OpenCine production for premise: '{premise}'")
        if progress_callback:
            progress_callback("Initializing Asset Registry & Scriptwriting...", 0.05)

        # --- LAYER A: Preproduction & Breakdown ---
        await self.registry_agent.execute()
        raw_script = await self.scriptwriter_agent.execute(premise, character_ids)
        production_script = await self.shotlist_agent.execute(raw_script)

        total_shots = sum(len(sc.shots) for sc in production_script.scenes)
        self.logger.info(f"Script compiled: '{production_script.title}' with {total_shots} shots across {len(production_script.scenes)} scenes.")

        if progress_callback:
            progress_callback(f"Script compiled ({total_shots} shots). Beginning Generation & QC Loop...", 0.15)

        # --- LAYER B & C: Generation, Spatial Consistency & QC Loop ---
        approved_shots: List[Tuple[ShotSpecification, Path]] = []
        shot_results: List[ShotRenderResult] = []
        cumulative_time = 0.0

        all_dialogue_tracks: List[AudioTrack] = []
        all_foley_tracks: List[AudioTrack] = []
        all_music_tracks: List[AudioTrack] = []

        shot_counter = 0
        for scene in production_script.scenes:
            scene_start_time = cumulative_time
            scene_duration = sum(s.duration_sec for s in scene.shots)

            # Generate music bed for this scene
            music_track = await self.music_agent.execute(scene, scene_start_time, scene_duration)
            all_music_tracks.append(music_track)

            for shot in scene.shots:
                shot_counter += 1
                if progress_callback:
                    pct = 0.15 + (0.60 * (shot_counter / max(1, total_shots)))
                    progress_callback(f"Processing Shot {shot.shot_id} [{shot_counter}/{total_shots}]...", pct)

                # Render with QC and healing loop
                final_shot_spec, master_clip_path, qc_reports = await self.render_and_heal_shot(shot)
                approved_shots.append((final_shot_spec, master_clip_path))

                # --- LAYER D: Dialogue & Foley generation for this shot ---
                diag_tracks = await self.dialogue_agent.execute(final_shot_spec, cumulative_time)
                foley_tracks = await self.foley_agent.execute(final_shot_spec, cumulative_time)

                all_dialogue_tracks.extend(diag_tracks)
                all_foley_tracks.extend(foley_tracks)

                shot_results.append(
                    ShotRenderResult(
                        shot_id=shot.shot_id,
                        raw_video_path=str(master_clip_path),
                        qc_reports=qc_reports,
                        approved=qc_reports[-1].pass_qc if qc_reports else True,
                        attempts=len(qc_reports),
                        final_video_path=str(master_clip_path)
                    )
                )

                cumulative_time += final_shot_spec.duration_sec

        if progress_callback:
            progress_callback("Executing Audio Mixing (Sidechain Ducking -12dB)...", 0.85)

        # --- LAYER D: Multichannel Audio Mixing & Final Cut Assembly ---
        master_audio_path = settings.outputs_dir / f"{production_script.title.lower().replace(' ', '_')}_master_audio.wav"
        await self.mixing_agent.execute(
            dialogue_tracks=all_dialogue_tracks,
            foley_tracks=all_foley_tracks,
            music_tracks=all_music_tracks,
            total_duration=cumulative_time,
            output_master_wav=master_audio_path
        )

        if progress_callback:
            progress_callback("Final Cut Assembly & Subtitle Integration...", 0.95)

        production_result = await self.assembly_agent.execute(
            approved_shots=approved_shots,
            master_audio_path=master_audio_path,
            production_title=production_script.title
        )
        production_result.shot_results = shot_results

        if progress_callback:
            progress_callback("Production complete!", 1.0)

        self.logger.info(f"Production successfully exported to: {production_result.final_master_path}")
        return production_result
