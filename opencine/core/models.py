"""OpenCineAgent Core Data Models.

Defines the domain entities for scriptwriting, shotlist, identity registry,
ComfyUI rendering, VLM quality control, and audio/video assembly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CameraMotion(str, Enum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    CRANE_UP = "crane_up"
    ORBIT_360 = "orbit_360"
    TRACKING = "tracking_shot"


class ShotType(str, Enum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"


class HealingAction(str, Enum):
    NONE = "none"
    FACE_DETAILER = "face_detailer"  # Mild issue: FaceDetailer / SAM inpaint
    RETRY_MUTATION = "retry_mutation"  # Critical issue: mutate seed / motion bucket


class DialogueLine(BaseModel):
    speaker_id: str = Field(..., description="ID of the character speaking")
    speaker_name: str = Field(..., description="Display name of speaker")
    text: str = Field(..., description="Spoken line")
    emotion: str = Field(default="neutral", description="Emotion/tone (e.g. tense, whispering, cheerful)")
    start_offset: float = Field(default=0.0, description="Offset in seconds from shot start")
    duration: float = Field(default=2.5, description="Estimated or actual audio duration")


class CharacterProfile(BaseModel):
    id: str = Field(..., description="Unique slug ID (e.g. 'elena_vance')")
    name: str = Field(..., description="Full character name")
    voice_id: str = Field(default="af_bella", description="Kokoro/F5-TTS voice ID or preset")
    face_embedding_path: Optional[str] = Field(None, description="Path to InsightFace/PuLID embedding or ref image")
    lora_name: Optional[str] = Field(None, description="ComfyUI LoRA model filename")
    lora_strength: float = Field(default=0.85, ge=0.0, le=2.0)
    invariable_clothing: str = Field(..., description="Fixed clothing prompt (e.g. 'navy wool trench coat, white collar')")
    facial_features: str = Field(..., description="Prompts describing facial features for identity reinforcement")
    trigger_words: List[str] = Field(default_factory=list, description="LoRA trigger words")


class EnvironmentStage(BaseModel):
    stage_id: str = Field(..., description="Identifier for stage/set (e.g. 'cyber_alley_rain')")
    name: str = Field(..., description="Human readable name of location")
    base_prompt: str = Field(..., description="Prompt describing empty scene / stage architecture")
    depth_map_path: Optional[str] = Field(None, description="Path to ControlNet Depth or 3D projection reference")
    locked_background_image: Optional[str] = Field(None, description="Path to pre-rendered locked environment frame")
    lighting_conditions: str = Field(default="cinematic neon, rainy reflections, moody shadows")


class ShotSpecification(BaseModel):
    shot_id: str = Field(..., description="Unique shot ID, e.g. 'SH_001'")
    scene_id: str = Field(default="SC_01")
    shot_number: int = Field(default=1)
    shot_type: ShotType = Field(default=ShotType.MEDIUM)
    camera_motion: CameraMotion = Field(default=CameraMotion.STATIC)
    action_description: str = Field(..., description="Detailed visual action taking place")
    stage_id: str = Field(..., description="Reference to EnvironmentStage")
    character_ids: List[str] = Field(default_factory=list, description="IDs of characters present in this shot")
    dialogue: List[DialogueLine] = Field(default_factory=list)
    duration_sec: float = Field(default=4.0, ge=1.0, le=30.0)

    @field_validator("shot_type", mode="before")
    @classmethod
    def normalize_shot_type(cls, v: Any) -> ShotType:
        if isinstance(v, ShotType):
            return v
        s = str(v).lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "close_up": ShotType.CLOSE_UP,
            "closeup": ShotType.CLOSE_UP,
            "extreme_close_up": ShotType.EXTREME_CLOSE_UP,
            "cu": ShotType.CLOSE_UP,
            "ecu": ShotType.EXTREME_CLOSE_UP,
            "wide": ShotType.WIDE,
            "wide_shot": ShotType.WIDE,
            "extreme_wide": ShotType.EXTREME_WIDE,
            "medium": ShotType.MEDIUM,
            "mid_shot": ShotType.MEDIUM,
            "medium_shot": ShotType.MEDIUM,
            "two_shot": ShotType.MEDIUM,
            "over_the_shoulder": ShotType.MEDIUM,
        }
        return aliases.get(s, ShotType.MEDIUM)

    @field_validator("camera_motion", mode="before")
    @classmethod
    def normalize_camera_motion(cls, v: Any) -> CameraMotion:
        if isinstance(v, CameraMotion):
            return v
        s = str(v).lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "static": CameraMotion.STATIC,
            "pan_left": CameraMotion.PAN_LEFT,
            "pan_right": CameraMotion.PAN_RIGHT,
            "tilt_up": CameraMotion.TILT_UP,
            "tilt_down": CameraMotion.TILT_DOWN,
            "dolly_in": CameraMotion.DOLLY_IN,
            "dolly_out": CameraMotion.DOLLY_OUT,
            "crane_up": CameraMotion.CRANE_UP,
            "orbit_360": CameraMotion.ORBIT_360,
            "orbit": CameraMotion.ORBIT_360,
            "tracking": CameraMotion.TRACKING,
            "tracking_shot": CameraMotion.TRACKING,
        }
        return aliases.get(s, CameraMotion.STATIC)

    # Prompt Engineering for Wan2.1 / Hunyuan / AnimateDiff
    positive_prompt: str = Field(default="")
    negative_prompt: str = Field(default="blurry, distorted, deformed hands, bad anatomy, lowres, watermark")
    seed: int = Field(default=42)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    motion_bucket_id: int = Field(default=127, ge=1, le=255)
    fps: int = Field(default=24)
    resolution: tuple[int, int] = Field(default=(1024, 576))


class SceneScript(BaseModel):
    scene_id: str = Field(..., description="e.g. 'SC_01'")
    title: str = Field(...)
    location: str = Field(...)
    time_of_day: str = Field(default="night")
    mood: str = Field(default="suspenseful")
    action_summary: str = Field(...)
    shots: List[ShotSpecification] = Field(default_factory=list)


class ProductionScript(BaseModel):
    title: str = Field(default="Untitled Production")
    premise: str = Field(...)
    genre: str = Field(default="Sci-Fi Thriller")
    target_duration_sec: float = Field(default=30.0)
    scenes: List[SceneScript] = Field(default_factory=list)


class QCReport(BaseModel):
    shot_id: str
    pass_qc: bool = Field(..., description="Whether the clip satisfies cinematic quality criteria")
    identity_drift_score: float = Field(default=0.0, ge=0.0, le=1.0, description="0.0 = perfect match, 1.0 = unrecognizable")
    deformed_faces: bool = Field(default=False)
    deformed_hands: bool = Field(default=False)
    motion_jitter: bool = Field(default=False)
    error_codes: List[str] = Field(default_factory=list, description="e.g. ['ERR_HANDS_GT5', 'ERR_IDENTITY_DRIFT']")
    healing_action: HealingAction = Field(default=HealingAction.NONE)
    suggested_seed: Optional[int] = None
    suggested_denoise: Optional[float] = None
    suggested_motion_bucket: Optional[int] = None
    inspected_frames: List[str] = Field(default_factory=list, description="Paths to inspected frame thumbnails")
    critique_notes: str = Field(default="")


class AudioTrack(BaseModel):
    track_id: str
    track_type: str = Field(..., description="'dialogue', 'foley', or 'music'")
    file_path: str
    start_time: float = Field(default=0.0)
    duration_sec: float = Field(default=0.0)
    gain_db: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShotRenderResult(BaseModel):
    shot_id: str
    raw_video_path: Optional[str] = None
    keyframe_path: Optional[str] = None
    healed_video_path: Optional[str] = None
    upscaled_video_path: Optional[str] = None
    qc_reports: List[QCReport] = Field(default_factory=list)
    approved: bool = False
    attempts: int = 0
    final_video_path: Optional[str] = None


class ProductionResult(BaseModel):
    production_id: str
    title: str
    final_master_path: str
    subtitles_srt_path: str
    subtitles_ass_path: Optional[str] = None
    shot_results: List[ShotRenderResult] = Field(default_factory=list)
    total_duration_sec: float = 0.0
    qc_pass_rate: float = 0.0
