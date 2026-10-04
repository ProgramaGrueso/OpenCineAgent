"""OpenCineAgent Global Configuration Settings.
Loads from environment variables, .env file, or defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENCINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    workspace_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "workspace")
    characters_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "config" / "characters")
    outputs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "workspace" / "outputs")
    temp_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "workspace" / "temp")

    # ComfyUI Headless Settings
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    comfyui_timeout: float = 300.0  # seconds for video generation
    comfyui_simulated: bool = False  # Set to True or fallback if ComfyUI backend lacks checkpoint

    # Ollama Settings
    ollama_url: str = "http://127.0.0.1:11434"
    llm_scriptwriter_model: str = "qwen3:8b"
    vlm_reviewer_model: str = "qwen2.5-vl:7b"
    vlm_fallback_model: str = "qwen3:8b"
    ollama_timeout: float = 120.0

    # Quality Control Thresholds
    max_qc_retries: int = 3
    identity_drift_threshold: float = 0.28  # Any score > 0.28 triggers healing/retry
    strict_hand_verification: bool = True

    # Video Engine Preset
    video_engine: str = "wan2.1"  # "wan2.1", "hunyuan", "animatediff", "ltx"
    default_fps: int = 24
    upscale_target_fps: int = 60
    enable_rife: bool = True
    enable_upscaler: bool = True

    # Audio Mixing Settings
    ducking_attenuation_db: float = -12.0
    ducking_attack_ms: int = 150
    ducking_release_ms: int = 400
    voice_sample_rate: int = 24000
    target_lufs: float = -14.0  # EBU R128 standard for cinematic streaming

    def setup_directories(self) -> None:
        """Ensure all runtime directories exist."""
        for d in [self.workspace_dir, self.characters_dir, self.outputs_dir, self.temp_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.setup_directories()
