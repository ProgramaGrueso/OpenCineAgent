"""Open model files used by OpenCineAgent, where ComfyUI expects them, and where to download them."""

from __future__ import annotations

from typing import List, NamedTuple

H3_REPO = "https://huggingface.co/Comfy-Org/MiniMax-H3"
KREA2_REPO = "https://huggingface.co/Comfy-Org/Krea-2"
MUSIC3_REPO = "https://huggingface.co/Comfy-Org/MiniMax-Music-3"


class ModelFile(NamedTuple):
    group: str
    filename: str
    folder: str
    url: str
    required: bool = True


CATALOG: List[ModelFile] = [
    ModelFile("MiniMax H3", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models",
              f"{H3_REPO}/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    ModelFile("MiniMax H3", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders",
              f"{H3_REPO}/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ModelFile("MiniMax H3", "minimax_h3_video_vae_fp16.safetensors", "vae",
              f"{H3_REPO}/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors"),
    ModelFile("MiniMax H3", "minimax_h3_audio_vae_fp32.safetensors", "vae",
              f"{H3_REPO}/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"),
    ModelFile("MiniMax H3", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras",
              f"{H3_REPO}/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"),
    ModelFile("Krea 2", "krea2_turbo_fp8_scaled.safetensors", "diffusion_models",
              f"{KREA2_REPO}/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
    ModelFile("Krea 2", "qwen3vl_4b_fp8_scaled.safetensors", "text_encoders",
              f"{KREA2_REPO}/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
    ModelFile("Krea 2", "qwen_image_vae.safetensors", "vae",
              f"{KREA2_REPO}/resolve/main/vae/qwen_image_vae.safetensors"),
    ModelFile("Krea 2", "krea2_identity_edit_v1_2.safetensors", "loras",
              "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors"),
    ModelFile("MiniMax Music 3", "minimax_music3_dit_fp16.safetensors", "diffusion_models",
              f"{MUSIC3_REPO}/resolve/main/diffusion_models/minimax_music3_dit_fp16.safetensors", False),
    ModelFile("MiniMax Music 3", "minimax_music3_text_encoder_pruned_int8_convrot.safetensors", "text_encoders",
              f"{MUSIC3_REPO}/resolve/main/text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
              False),
    ModelFile("MiniMax Music 3", "minimax_music3_dav.safetensors", "vae",
              f"{MUSIC3_REPO}/resolve/main/vae/minimax_music3_dav.safetensors", False),
]

CUSTOM_NODES = [
    ("comfyui-krea2edit", "https://github.com/lbouaraba/comfyui-krea2edit", "Krea 2 Identity Edit (keyframes)", True),
    ("ComfyUI-MiniMaxH3Mod", "https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod", "RefMod identity anchors", False),
    ("ComfyUI-Pixaroma", "https://github.com/pixaroma/ComfyUI-Pixaroma", "H3 lipsync to a recorded voice", False),
    ("plaguekind-nodes", "https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes", "H3SLAAttention (OPENCINE_H3_SLA=true)", False),
]
