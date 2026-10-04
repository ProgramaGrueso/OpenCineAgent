"""OpenCineAgent Global Configuration Settings.
Loads from environment variables (prefix OPENCINE_), a .env file, or defaults.

Model defaults point at the official open checkpoints published by Comfy-Org
(MiniMax H3 and Krea 2). Fine-tunes can be swapped in through .env without
touching code; see README "Models".
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENCINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    project_root: Path = _ROOT
    workspace_dir: Path = _ROOT / "workspace"
    characters_dir: Path = _ROOT / "config" / "characters"

    # ComfyUI Headless Settings
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    comfyui_timeout: float = 3600.0      # per job; an H3 clip takes 2-8 min, a cold UNET load ~15 min
    comfyui_min_nodes: int = 0           # >0 rejects a partially started ComfyUI (node count below it)
    comfyui_free_between_jobs: bool = True   # /free when the next job uses different models

    # Ollama Settings
    ollama_url: str = "http://127.0.0.1:11434"
    llm_scriptwriter_model: str = "qwen3:8b"
    vlm_reviewer_model: str = "qwen2.5vl:7b"
    ollama_timeout: float = 300.0

    # Production format
    dialogue_language: str = "English"
    fps: int = 24
    video_width: int = 1024
    video_height: int = 576
    keyframe_width: int = 1536
    keyframe_height: int = 864

    # Krea 2 (keyframes, master portraits, location plates)
    krea2_unet: str = "krea2_turbo_fp8_scaled.safetensors"
    krea2_clip: str = "qwen3vl_4b_fp8_scaled.safetensors"
    krea2_vae: str = "qwen_image_vae.safetensors"
    krea2_steps: int = 8
    krea2_sampler: str = "euler"
    krea2_scheduler: str = "simple"
    krea2_hires_pass: bool = False       # two-pass recipe for mixes like moodyKrea2Mix (er_sde 7 -> x2 -> dpmpp_2m_sde 9)
    krea2_identity_edit_lora: str = "krea2_identity_edit_v1_2.safetensors"
    krea2_identity_edit_steps: int = 10

    # MiniMax H3 (reference-to-video with native audio)
    h3_unet: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    h3_clip: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    h3_video_vae: str = "minimax_h3_video_vae_fp16.safetensors"
    h3_audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"
    h3_turbo_lora: Optional[str] = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    h3_steps: int = 4                    # 4 with the turbo LoRA; 20 without it; 8 for distilled fine-tunes
    h3_sampler: str = "res_multistep"
    h3_scheduler: str = "simple"
    h3_ref_image_size: str = "max"       # "max" keeps identity; "match" squashes refs and merges faces
    h3_sla: bool = False                 # H3SLAAttention (plaguekind-nodes): faster, better lipsync
    h3_sla_block_size: str = "32"
    h3_min_frames: int = 124             # trained length range of H3: 124-362 frames
    h3_max_frames: int = 362

    # Quality Control & Healing
    max_qc_retries: int = 2
    dialogue_match_threshold: float = 0.55   # whisper transcript vs script similarity
    whisper_model: str = "medium"

    # Upscaling: "none", "lanczos" (ffmpeg) or "clearreality" (spandrel, needs torch)
    upscale_mode: str = "lanczos"
    upscale_width: int = 1920
    upscale_height: int = 1080
    upscale_model_path: Optional[Path] = None
    upscale_python: Optional[str] = None     # python with torch+spandrel, e.g. ~/ComfyUI/venv/bin/python

    # Music: "none", "file" (--music) or "minimax" (MiniMax Music 3 in ComfyUI)
    music_mode: str = "none"
    music3_unet: str = "minimax_music3_dit_fp16.safetensors"
    music3_clip: str = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
    music3_vae: str = "minimax_music3_dav.safetensors"

    # Audio Mixing
    music_gain_db: float = -11.0
    ducking_attack_ms: int = 150
    ducking_release_ms: int = 400
    target_lufs: float = -14.0

    @property
    def comfyui_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    def production_dir(self, slug: str) -> Path:
        d = self.workspace_dir / "productions" / slug
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
