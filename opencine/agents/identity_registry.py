"""Agent 3: Asset & Identity Registry Agent (Layer A).
Maintains centralized registry with fixed embeddings, LoRAs, and invariable clothing/silhouette.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from opencine.agents.base import BaseAgent
from opencine.core.config import settings
from opencine.core.models import CharacterProfile


class AssetIdentityRegistryAgent(BaseAgent):
    """Central registry for character face embeddings, LoRA weights, and invariable prompts."""

    def __init__(self, registry_file: Optional[Path] = None) -> None:
        super().__init__(name="AssetIdentityRegistryAgent", layer="Layer A")
        self.registry_file = registry_file or (settings.characters_dir / "registry.json")
        self._characters: Dict[str, CharacterProfile] = {}
        self._load_default_characters()

    def _load_default_characters(self) -> None:
        """Seed default 4 characters if registry doesn't exist yet."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        profile = CharacterProfile(**item)
                        self._characters[profile.id] = profile
                return
            except Exception as e:
                self.logger.warning(f"Could not load existing registry: {e}. Seeding defaults.")

        # Default standard 4 characters for cinematic production
        defaults = [
            CharacterProfile(
                id="elena_vance",
                name="Elena Vance",
                voice_id="af_bella",
                face_embedding_path=str(settings.characters_dir / "elena_vance_face.safetensors"),
                lora_name="elena_vance_v2.safetensors",
                lora_strength=0.85,
                invariable_clothing="navy blue high-collar wool trench coat, dark turtleneck, titanium wristwatch",
                facial_features="sharp jawline, piercing emerald eyes, raven hair tied in a sleek low ponytail, slight scar on left brow",
                trigger_words=["elena_vance_style", "cinematic lighting", "8k portrait"]
            ),
            CharacterProfile(
                id="marcus_reid",
                name="Marcus Reid",
                voice_id="am_adam",
                face_embedding_path=str(settings.characters_dir / "marcus_reid_face.safetensors"),
                lora_name="marcus_reid_v1.safetensors",
                lora_strength=0.80,
                invariable_clothing="tactical charcoal bomber jacket, faded grey henley, rugged leather holster harness",
                facial_features="weathered rugged face, trimmed salt-and-pepper beard, intense hazel eyes, prominent nose bridge",
                trigger_words=["marcus_reid_soldier", "photorealistic", "raw texture"]
            ),
            CharacterProfile(
                id="maya_lin",
                name="Dr. Maya Lin",
                voice_id="af_sarah",
                face_embedding_path=str(settings.characters_dir / "maya_lin_face.safetensors"),
                lora_name="maya_lin_cyber.safetensors",
                lora_strength=0.88,
                invariable_clothing="minimalist white lab tunic with luminescent cyan stitching, graphite trousers",
                facial_features="delicate features, warm amber eyes, asymmetrical chin-length obsidian bob, augmented optic temple piece",
                trigger_words=["maya_lin_scientist", "subsurface scattering", "masterpiece"]
            ),
            CharacterProfile(
                id="victor_cross",
                name="Victor Cross",
                voice_id="am_michael",
                face_embedding_path=str(settings.characters_dir / "victor_cross_face.safetensors"),
                lora_name="victor_cross_noir.safetensors",
                lora_strength=0.82,
                invariable_clothing="bespoke pinstripe charcoal three-piece suit, silver signet ring, silk obsidian tie",
                facial_features="stoic aristocratic expression, slicked-back platinum hair, steel-blue eyes, subtle cybernetic cheek seam",
                trigger_words=["victor_cross_exec", "noir lighting", "dramatic rim light"]
            )
        ]

        for p in defaults:
            self._characters[p.id] = p
        self.save_registry()

    def save_registry(self) -> None:
        """Persist character registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump([p.model_dump() for p in self._characters.values()], f, indent=2)
        self.logger.info(f"Saved {len(self._characters)} character profiles to {self.registry_file}")

    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        return self._characters.get(character_id)

    def list_characters(self) -> List[CharacterProfile]:
        return list(self._characters.values())

    def register_character(self, profile: CharacterProfile) -> None:
        self._characters[profile.id] = profile
        self.save_registry()

    def compile_character_prompt_enhancement(self, character_ids: List[str]) -> str:
        """Generates deterministic identity reinforcement strings for prompt injection."""
        enhancements = []
        for cid in character_ids:
            char = self.get_character(cid)
            if char:
                enhancements.append(
                    f"Character ({char.name}): {char.facial_features}, wearing {char.invariable_clothing}."
                )
        return " ".join(enhancements)

    async def execute(self, *args: any, **kwargs: any) -> Dict[str, CharacterProfile]:
        """Returns the loaded character registry."""
        await self.emit("identity_registry.loaded", {"count": len(self._characters)})
        return self._characters
