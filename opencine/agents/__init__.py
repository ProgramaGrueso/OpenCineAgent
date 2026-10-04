"""OpenCineAgent: The 14 Autonomous Agents Registry."""

from opencine.agents.base import BaseAgent

# Layer A: Pre-production & Breakdown
from opencine.agents.scriptwriter import ScriptwriterAgent
from opencine.agents.shotlist import ShotlistPromptAgent
from opencine.agents.identity_registry import AssetIdentityRegistryAgent

# Layer B: Generation & Spatial Consistency
from opencine.agents.environment import EnvironmentStageAgent
from opencine.agents.framestaging import FrameStagingAgent
from opencine.agents.video_synthesis import VideoSynthesisAgent

# Layer C: Quality Control & Healing
from opencine.agents.critic_vlm import CriticVlmAgent
from opencine.agents.healing import LoopbackHealingAgent
from opencine.agents.super_res import SuperResolutionAgent

# Layer D: Audio, Subtitles & Assembly
from opencine.agents.voice_dialogue import VoiceDialogueAgent
from opencine.agents.foley_sound import FoleySoundAgent
from opencine.agents.music_scoring import MusicScoringAgent
from opencine.agents.audio_mixing import AudioMixingAgent
from opencine.agents.assembly import AssemblyFinalCutAgent

__all__ = [
    "BaseAgent",
    # Layer A
    "ScriptwriterAgent",
    "ShotlistPromptAgent",
    "AssetIdentityRegistryAgent",
    # Layer B
    "EnvironmentStageAgent",
    "FrameStagingAgent",
    "VideoSynthesisAgent",
    # Layer C
    "CriticVlmAgent",
    "LoopbackHealingAgent",
    "SuperResolutionAgent",
    # Layer D
    "VoiceDialogueAgent",
    "FoleySoundAgent",
    "MusicScoringAgent",
    "AudioMixingAgent",
    "AssemblyFinalCutAgent",
]
