"""MiniMax H3 graphs: reference-to-video with native audio (or lipsync to a voice track) and RefMod extraction."""

from __future__ import annotations

import math
from typing import List, Optional

from opencine.core.config import settings

Workflow = dict


def h3_length(duration_sec: float, fps: int = 24) -> int:
    """Smallest valid H3 length (5 + 17k frames) covering the shot, clamped to the trained range.
    Clips are rendered a bit longer and trimmed to the exact duration at assembly."""
    frames = max(settings.h3_min_frames, math.ceil(duration_sec * fps))
    k = math.ceil((frames - 5) / 17)
    return min(5 + 17 * k, settings.h3_max_frames)


def h3_reference_to_video(prompt: str, ref_images: List[str], width: int, height: int, length: int,
                          seed: int, prefix: str, refmod: Optional[str] = None,
                          voice_track: Optional[str] = None) -> Workflow:
    """REF2VA: <Picture 1> = keyframe, <Picture 2..> = master portraits / props.

    refmod      only for shots with ONE face: a RefMod pushes identity on the whole conditioning,
                so two faces collapse into the same person.
    voice_track uploaded WAV with the shot's lines at their timestamps -> PixaromaH3AudioSync lipsync.
                Without it H3 speaks the <d>...</d> lines itself (native audio).
    """
    wf: Workflow = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.h3_unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": settings.h3_clip, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": settings.h3_video_vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": settings.h3_audio_vae}},
        "r2v": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0], "prompt": prompt,
            "width": width, "height": height, "length": length,
            "ref_image_size": settings.h3_ref_image_size}},
    }
    for i, name in enumerate(ref_images):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["r2v"]["inputs"][f"ref_images.ref_image_{i}"] = [f"img{i}", 0]

    model = ["unet", 0]
    if settings.h3_turbo_lora:
        wf["turbo"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": settings.h3_turbo_lora, "strength_model": 1.0}}
        model = ["turbo", 0]
    if settings.h3_sla:
        wf["sla"] = {"class_type": "H3SLAAttention", "inputs": {
            "model": model, "sparsity_ratio": 0.8, "block_size": settings.h3_sla_block_size,
            "dense_last_steps": 0, "dense_steps": "1", "protect_audio": True,
            "stabilize_motion": True, "enabled": True}}
        model = ["sla", 0]

    cond = ["r2v", 0]
    if refmod:
        wf["refmods"] = {"class_type": "MiniMaxH3RefModsLoader", "inputs": {
            "show_info": False, "mod_1": refmod, "strength_1": 1.0, "copies_1": 2,
            **{f"mod_{i}": "(none)" for i in range(2, 9)},
            **{f"strength_{i}": 1.0 for i in range(2, 9)},
            **{f"copies_{i}": 1 for i in range(2, 9)}}}
        wf["refmod_apply"] = {"class_type": "MiniMaxH3RefModApply", "inputs": {
            "conditioning": cond, "mods": ["refmods", 0], "override": False, "retention": 1.0,
            "curve_direction": "constant", "curve_shape": "linear", "curve_value": 1.0, "scramble_seed": -1}}
        cond = ["refmod_apply", 0]
    wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": cond}}

    latent = ["r2v", 1]
    if voice_track:
        wf["voice"] = {"class_type": "LoadAudio", "inputs": {"audio": voice_track}}
        wf["r2v"]["inputs"]["ref_audios.ref_audio_0"] = ["voice", 0]
        wf["sync"] = {"class_type": "PixaromaH3AudioSync", "inputs": {
            "model": model, "latent": ["r2v", 1], "audio_vae": ["avae", 0], "track": ["voice", 0]}}
        model, latent, audio = ["sync", 0], ["sync", 1], ["sync", 2]
    else:
        wf["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["ks", 0], "vae": ["avae", 0]}}
        audio = ["adec", 0]

    wf["ks"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": cond, "negative": ["neg", 0], "latent_image": latent, "seed": seed,
        "steps": settings.h3_steps, "cfg": 1.0, "sampler_name": settings.h3_sampler,
        "scheduler": settings.h3_scheduler, "denoise": 1.0}}
    wf["vdec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["vdec", 0], "fps": float(settings.fps),
                                                            "audio": audio}}
    wf["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix,
                                                         "format": "mp4", "codec": "h264"}}
    return wf


def refmod_extract(name: str, images: List[str], description: str) -> Workflow:
    """Encodes a character's identity into a reusable RefMod (saved by ComfyUI under models/refmods).
    mode 'encode', never pooling: pooling destroys facial identity. All fields are explicit because
    the API rejects the node when a preset is used without them."""
    wf: Workflow = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.h3_unet, "weight_dtype": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": settings.h3_video_vae}},
        "extract": {"class_type": "MiniMaxH3RefModExtract", "inputs": {
            "name": name, "concept_type": "identity", "mode": "encode", "latent_frames": 1, "identity": 0,
            "merge": False, "motion_only": False, "pool_h": 16, "pool_w": 16, "multiplier": 1,
            "description": description, "save": True, "subfolder": "", "vae": ["vae", 0],
            "background_retention": 0.0, "ref_resolution": 1024, "max_tokens": 5120,
            "budget_policy": "truncate"}},
        "done": {"class_type": "PreviewAny", "inputs": {"source": ["extract", 0]}},
    }
    for i, image in enumerate(images):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        wf["extract"]["inputs"][f"refs_image.ref_image_{i}"] = [f"img{i}", 0]
    return wf
