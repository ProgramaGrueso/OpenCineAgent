"""OpenCineAgent Core Data Models.

Domain entities for scriptwriting, identity, ComfyUI rendering (Krea 2 + MiniMax H3),
quality control and final assembly.
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
    HANDHELD = "handheld"


class ShotType(str, Enum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"


class HealingAction(str, Enum):
    NONE = "none"
    RETRY_NEW_SEED = "retry_new_seed"   # anatomy / identity / dialogue failures: same prompt, new seed
    SHORTEN_CLIP = "shorten_clip"       # CUDA OOM: fewer frames fit in VRAM


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"   # the check could not run (tool or model missing): never counted as a pass


_SHOT_ALIASES = {
    "closeup": ShotType.CLOSE_UP, "cu": ShotType.CLOSE_UP, "ecu": ShotType.EXTREME_CLOSE_UP,
    "wide_shot": ShotType.WIDE, "establishing": ShotType.EXTREME_WIDE, "mid_shot": ShotType.MEDIUM,
    "medium_shot": ShotType.MEDIUM, "two_shot": ShotType.MEDIUM, "over_the_shoulder": ShotType.MEDIUM,
    "medium_close_up": ShotType.CLOSE_UP,
}
_MOTION_ALIASES = {"orbit": CameraMotion.ORBIT_360, "tracking": CameraMotion.TRACKING,
                   "push_in": CameraMotion.DOLLY_IN, "pull_out": CameraMotion.DOLLY_OUT}


def _norm(v: Any) -> str:
    return str(v).lower().strip().replace("-", "_").replace(" ", "_")


class DialogueLine(BaseModel):
    speaker_id: str = Field(..., description="ID of the character speaking")
    speaker_name: str = Field(default="", description="Display name of speaker")
    text: str = Field(..., description="Spoken line, in the production's dialogue language")
    emotion: str = Field(default="neutral", description="Delivery direction, e.g. 'whispering, sardonic'")
    start_offset: float = Field(default=0.5, ge=0.0, description="Seconds from shot start")
    duration: float = Field(default=2.5, gt=0.0, description="Estimated spoken duration")
    audio_path: Optional[str] = Field(None, description="Pre-recorded line (actor/TTS). Enables lipsync mode")


class TextCue(BaseModel):
    """On-screen text burned in post (models cannot be trusted to write letters)."""
    text: str
    start: float = Field(default=0.0, ge=0.0, description="Seconds from shot start")
    end: float = Field(..., gt=0.0, description="Seconds from shot start")
    position: str = Field(default="right", description="left | right | center | bottom | title")


class CharacterProfile(BaseModel):
    id: str = Field(..., description="Unique slug ID (e.g. 'elena_vance')")
    name: str = Field(..., description="Full character name")
    description: str = Field(..., description="Who they are, age and build (e.g. 'a woman in her thirties')")
    facial_features: str = Field(..., description="Face and hair; repeated verbatim in every prompt")
    invariable_clothing: str = Field(..., description="Fixed wardrobe; describe material and color separately")
    voice_description: str = Field(default="a natural voice", description="Voice direction for H3 native speech")
    reference_image: Optional[str] = Field(None, description="Approved master portrait (relative to characters dir)")
    approved: bool = Field(default=False, description="A human confirmed the master portrait")
    lora_name: Optional[str] = Field(None, description="Krea 2 character LoRA used for the master portrait")
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    trigger_words: List[str] = Field(default_factory=list, description="LoRA trigger words")
    refmod_name: Optional[str] = Field(None, description="MiniMax H3 RefMod (single-face shots only)")

    def identity_block(self) -> str:
        return f"{self.description}, {self.facial_features}, wearing {self.invariable_clothing}"


class EnvironmentStage(BaseModel):
    stage_id: str = Field(..., description="Identifier for stage/set (e.g. 'penthouse_terrace')")
    name: str = Field(..., description="Human readable name of location")
    base_prompt: str = Field(..., description="The empty location, with time of day and lighting")
    locked_background_image: Optional[str] = Field(None, description="Rendered empty plate, reused by every shot")


class ShotSpecification(BaseModel):
    shot_id: str = Field(..., description="Unique shot ID, e.g. 'SH_001'")
    scene_id: str = Field(default="SC_01")
    shot_number: int = Field(default=1)
    shot_type: ShotType = Field(default=ShotType.MEDIUM)
    camera_motion: CameraMotion = Field(default=CameraMotion.STATIC)
    action_description: str = Field(..., description="What happens, in present tense, in temporal order")
    stage_id: str = Field(..., description="Reference to EnvironmentStage")
    character_ids: List[str] = Field(default_factory=list, description="Characters on screen, main one first")
    dialogue: List[DialogueLine] = Field(default_factory=list)
    sound_description: str = Field(default="", description="Diegetic sound for H3's native audio")
    keyframe_action: Optional[str] = Field(None, description="What the first frame shows, if it differs from the action")
    on_screen_text: List[TextCue] = Field(default_factory=list, description="Labels/titles burned in post")
    duration_sec: float = Field(default=6.0, ge=1.0, le=15.0)

    # Compiled by the Shotlist agent
    keyframe_prompt: str = Field(default="", description="Krea 2 prompt for the first frame")
    video_prompt: str = Field(default="", description="MiniMax H3 prompt (alignment line + [Shot 1] contract)")
    length_frames: int = Field(default=124, description="H3 length: 5 + 17k frames, rendered >= duration")
    seed: int = Field(default=42)
    voice_track: Optional[str] = Field(None, description="Per-clip voice WAV for lipsync mode")

    @field_validator("shot_type", mode="before")
    @classmethod
    def normalize_shot_type(cls, v: Any) -> ShotType:
        if isinstance(v, ShotType):
            return v
        s = _norm(v)
        try:
            return ShotType(s)
        except ValueError:
            return _SHOT_ALIASES.get(s, ShotType.MEDIUM)

    @field_validator("camera_motion", mode="before")
    @classmethod
    def normalize_camera_motion(cls, v: Any) -> CameraMotion:
        if isinstance(v, CameraMotion):
            return v
        s = _norm(v)
        try:
            return CameraMotion(s)
        except ValueError:
            return _MOTION_ALIASES.get(s, CameraMotion.STATIC)


class SceneScript(BaseModel):
    scene_id: str = Field(..., description="e.g. 'SC_01'")
    title: str = Field(...)
    location: str = Field(..., description="Empty location description used for the stage plate")
    location_image: Optional[str] = Field(None, description="Photo of a real location, used instead of a generated plate")
    time_of_day: str = Field(default="night")
    mood: str = Field(default="suspenseful")
    action_summary: str = Field(default="")
    music_description: str = Field(default="", description="Score direction for this scene")
    shots: List[ShotSpecification] = Field(default_factory=list)


class ProductionScript(BaseModel):
    title: str = Field(default="Untitled Production")
    premise: str = Field(default="")
    genre: str = Field(default="Drama")
    language: str = Field(default="English", description="Dialogue language passed to H3")
    visual_style: str = Field(default="photorealistic cinematic film still, natural skin texture, shallow depth of field")
    seed: Optional[int] = Field(None, description="Same seed for every shot (retakes still change it)")
    scenes: List[SceneScript] = Field(default_factory=list)

    def all_shots(self) -> List[ShotSpecification]:
        return [shot for scene in self.scenes for shot in scene.shots]


class QCCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str = ""


class QCReport(BaseModel):
    shot_id: str
    attempt: int = 1
    pass_qc: bool
    checks: List[QCCheck] = Field(default_factory=list)
    error_codes: List[str] = Field(default_factory=list)
    healing_action: HealingAction = Field(default=HealingAction.NONE)
    transcript: str = Field(default="")
    inspected_frames: List[str] = Field(default_factory=list)

    @property
    def unverified(self) -> List[str]:
        return [c.name for c in self.checks if c.status == CheckStatus.SKIPPED]


class AudioTrack(BaseModel):
    track_id: str
    track_type: str = Field(..., description="'dialogue', 'music' or 'clip' (H3 native audio)")
    file_path: str
    start_time: float = Field(default=0.0)
    duration_sec: float = Field(default=0.0)
    gain_db: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShotRenderResult(BaseModel):
    shot_id: str
    keyframe_path: Optional[str] = None
    raw_video_path: Optional[str] = None
    final_video_path: Optional[str] = None
    qc_reports: List[QCReport] = Field(default_factory=list)
    approved: bool = False
    attempts: int = 0
    seed: int = 0


class ProductionResult(BaseModel):
    production_id: str
    title: str
    final_master_path: str
    subtitles_srt_path: str
    shot_results: List[ShotRenderResult] = Field(default_factory=list)
    total_duration_sec: float = 0.0
    qc_pass_rate: float = 0.0
