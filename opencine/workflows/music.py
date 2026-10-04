"""MiniMax Music 3 graph (official template values: 30 steps, cfg 1.7, top_k 50, euler/simple).
The sampler is a regular KSampler; the core SaveAudio node is used because SaveAudioAdvanced's
dynamic 'format' combo breaks through the API."""

from __future__ import annotations

from opencine.core.config import settings

Workflow = dict


def minimax_music3(caption: str, lyrics: str, seconds: float, seed: int, prefix: str) -> Workflow:
    seconds = round(min(max(seconds, 5.0), 360.0), 2)
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": settings.music3_unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": settings.music3_clip, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": settings.music3_vae}},
        "enc": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {
            "clip": ["clip", 0], "caption": caption, "lyrics": lyrics, "seed": seed,
            "max_duration": seconds, "cfg_scale": 1.7, "top_k": 50}},
        "latent": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {"seconds": seconds, "batch_size": 1}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["enc", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "positive": ["enc", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": 30, "cfg": 1.7, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveAudio", "inputs": {"audio": ["decode", 0], "filename_prefix": prefix}},
    }
