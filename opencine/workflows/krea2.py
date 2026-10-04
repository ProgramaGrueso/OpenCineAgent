"""Krea 2 graphs: text-to-image (master portraits, location plates) and Identity Edit (keyframes)."""

from __future__ import annotations

from typing import Optional, Tuple

from opencine.core.config import settings

Workflow = dict


def _loaders(wf: Workflow) -> None:
    wf["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": settings.krea2_unet, "weight_dtype": "default"}}
    wf["clip"] = {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": settings.krea2_clip, "type": "krea2", "device": "default"}}
    wf["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": settings.krea2_vae}}


def krea2_text_to_image(prompt: str, width: int, height: int, seed: int, prefix: str,
                        lora: Optional[Tuple[str, float]] = None) -> Workflow:
    """Krea 2 t2i. With `krea2_hires_pass` it uses the two-pass recipe validated on Krea 2 mixes:
    er_sde 7 steps at half size (leftover noise) -> latent x2 bislerp -> dpmpp_2m_sde from step 4."""
    wf: Workflow = {}
    _loaders(wf)
    model = ["unet", 0]
    if lora:
        wf["lora"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model, "lora_name": lora[0], "strength_model": lora[1]}}
        model = ["lora", 0]
    wf["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
    wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}

    if settings.krea2_hires_pass:
        wf["latent"] = {"class_type": "EmptyLatentImage",
                        "inputs": {"width": width // 2, "height": height // 2, "batch_size": 1}}
        wf["pass1"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": model, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "add_noise": "enable", "noise_seed": seed, "steps": 7, "cfg": 1.0, "sampler_name": "er_sde",
            "scheduler": "simple", "start_at_step": 0, "end_at_step": 7, "return_with_leftover_noise": "enable"}}
        wf["up"] = {"class_type": "LatentUpscaleBy",
                    "inputs": {"samples": ["pass1", 0], "upscale_method": "bislerp", "scale_by": 2.0}}
        wf["pass2"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": model, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["up", 0],
            "add_noise": "enable", "noise_seed": seed, "steps": 9, "cfg": 1.0, "sampler_name": "dpmpp_2m_sde",
            "scheduler": "sgm_uniform", "start_at_step": 4, "end_at_step": 999,
            "return_with_leftover_noise": "disable"}}
        samples = ["pass2", 0]
    else:
        wf["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        wf["ks"] = {"class_type": "KSampler", "inputs": {
            "model": model, "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": settings.krea2_steps, "cfg": 1.0, "sampler_name": settings.krea2_sampler,
            "scheduler": settings.krea2_scheduler, "denoise": 1.0}}
        samples = ["ks", 0]

    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": samples, "vae": ["vae", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return wf


def krea2_identity_edit(prompt: str, scene_image: str, subject_image: str, width: int, height: int,
                        seed: int, prefix: str) -> Workflow:
    """Krea 2 Identity Edit: places the subject (image B) into the scene (image A) keeping face and set.
    Order matters: scene first, subject second. ref_boost 4 holds identity from a single photo."""
    wf: Workflow = {}
    _loaders(wf)
    wf["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["unet", 0], "lora_name": settings.krea2_identity_edit_lora, "strength_model": 1.0}}
    wf["img_scene"] = {"class_type": "LoadImage", "inputs": {"image": scene_image}}
    wf["img_subject"] = {"class_type": "LoadImage", "inputs": {"image": subject_image}}
    wf["enc_scene"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["img_scene", 0], "vae": ["vae", 0]}}
    wf["enc_subject"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["img_subject", 0], "vae": ["vae", 0]}}
    wf["patch"] = {"class_type": "Krea2EditModelPatch", "inputs": {
        "model": ["lora", 0], "source_latent": ["enc_scene", 0], "source_image": ["img_scene", 0],
        "source_latent_b": ["enc_subject", 0], "source_image_b": ["img_subject", 0],
        "ref_boost": 4.0, "ref_boost_a": 1.0, "fit_mode": "fit", "vae": ["vae", 0]}}
    for key, text in (("pos", prompt), ("neg", "")):
        wf[key] = {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": ["clip", 0], "prompt": text, "image": ["img_scene", 0], "image_b": ["img_subject", 0],
            "grounding_px": 768, "system_prompt": ""}}
    wf["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    wf["ks"] = {"class_type": "KSampler", "inputs": {
        "model": ["patch", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
        "seed": seed, "steps": settings.krea2_identity_edit_steps, "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "simple", "denoise": 1.0}}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return wf
