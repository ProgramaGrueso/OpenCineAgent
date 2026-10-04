"""Agent 3: Asset & Identity Registry Agent (Layer A).

Owns each character's identity anchors:
  * a master portrait (Krea 2 t2i, optional character LoRA, grey studio background) that a human
    approves once; every keyframe is derived from it with Identity Edit, so all shots share one face;
  * an optional MiniMax H3 RefMod extracted from approved images (used only in single-face shots);
  * the identity block (description, face, wardrobe) repeated verbatim in every prompt.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from opencine.agents.base import BaseAgent
from opencine.core.comfy_client import ComfyUIClient, JobFailed, comfy_client
from opencine.core.config import settings
from opencine.core.models import CharacterProfile
from opencine.core.prompts import master_portrait_prompt
from opencine.core.seeds import stable_seed
from opencine.workflows import krea2_text_to_image, refmod_extract


class IdentityNotReady(RuntimeError):
    """A character has no approved master portrait yet."""


class AssetIdentityRegistryAgent(BaseAgent):
    def __init__(self, registry_file: Optional[Path] = None, client: Optional[ComfyUIClient] = None) -> None:
        super().__init__(name="AssetIdentityRegistryAgent", layer="Layer A")
        self.registry_file = registry_file or (settings.characters_dir / "registry.json")
        self.comfy = client or comfy_client
        self._characters: Dict[str, CharacterProfile] = {}
        self.load()

    # ------------------------------------------------------------- persistence
    def load(self) -> None:
        self._characters.clear()
        if not self.registry_file.exists():
            self.logger.warning(f"No character registry at {self.registry_file}")
            return
        for item in json.loads(self.registry_file.read_text(encoding="utf-8")):
            profile = CharacterProfile(**item)
            self._characters[profile.id] = profile

    def save_registry(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = [p.model_dump(exclude_none=True) for p in self._characters.values()]
        self.registry_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def get_character(self, character_id: str) -> Optional[CharacterProfile]:
        return self._characters.get(character_id)

    def list_characters(self) -> List[CharacterProfile]:
        return list(self._characters.values())

    def register_character(self, profile: CharacterProfile) -> None:
        self._characters[profile.id] = profile
        self.save_registry()

    def character_dir(self, character_id: str) -> Path:
        return self.registry_file.parent / character_id

    def reference_path(self, profile: CharacterProfile) -> Optional[Path]:
        if not profile.reference_image:
            return None
        path = Path(profile.reference_image)
        return path if path.is_absolute() else self.registry_file.parent / path

    def cast(self, character_ids: List[str]) -> List[CharacterProfile]:
        missing = [cid for cid in character_ids if cid not in self._characters]
        if missing:
            raise KeyError(f"Unknown character id(s): {missing}. Registered: {sorted(self._characters)}")
        return [self._characters[cid] for cid in character_ids]

    # ------------------------------------------------------- master portraits
    async def generate_portraits(self, character_id: str, candidates: int = 2,
                                 style: str = "photorealistic") -> List[Path]:
        """Renders candidate master portraits; a human picks one with approve()."""
        profile = self.cast([character_id])[0]
        out_dir = self.character_dir(character_id) / "candidates"
        lora = (profile.lora_name, profile.lora_strength) if profile.lora_name else None
        paths = []
        for i in range(candidates):
            seed = stable_seed(character_id, "portrait", i)
            wf = krea2_text_to_image(master_portrait_prompt(profile, style), 832, 1216, seed,
                                     f"opencine/cast/{character_id}", lora=lora)
            out = await self.comfy.run(wf, out_dir, f"portrait:{character_id}:{i}", suffixes=(".png",))
            target = out_dir / f"seed_{seed}.png"
            out[0].replace(target)
            paths.append(target)
        await self.emit("identity.portraits", {"character_id": character_id, "candidates": [str(p) for p in paths]})
        return paths

    def approve(self, character_id: str, image: Path) -> CharacterProfile:
        profile = self.cast([character_id])[0]
        target = self.character_dir(character_id) / "master.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(image).resolve() != target.resolve():
            shutil.copy2(image, target)
        profile.reference_image = str(target.relative_to(self.registry_file.parent))
        profile.approved = True
        self.save_registry()
        return profile

    async def ensure_ready(self, character_ids: List[str], auto_approve: bool = False) -> List[CharacterProfile]:
        """Every cast member needs an approved master portrait before any shot is rendered."""
        cast = self.cast(character_ids)
        for profile in cast:
            ref = self.reference_path(profile)
            if ref and ref.exists() and profile.approved:
                continue
            if not auto_approve:
                raise IdentityNotReady(
                    f"'{profile.id}' has no approved master portrait. Run:\n"
                    f"  python main.py cast {profile.id}            # renders candidates\n"
                    f"  python main.py approve {profile.id} <image>  # after checking it is the right person\n"
                    f"or pass --auto-approve to accept the first candidate unseen."
                )
            if ref and ref.exists():
                self.approve(profile.id, ref)
            else:
                candidate = (await self.generate_portraits(profile.id, candidates=1))[0]
                self.approve(profile.id, candidate)
            self.logger.warning(f"Auto-approved master portrait for '{profile.id}' without human review")
        return cast

    # ------------------------------------------------------------------ RefMod
    async def extract_refmod(self, character_id: str, images: List[Path]) -> str:
        """Builds a MiniMax H3 RefMod from approved face images (8 varied angles work best)."""
        profile = self.cast([character_id])[0]
        if not images:
            raise ValueError("RefMod extraction needs at least one face image")
        name = f"opencine_{character_id}_refmod"
        uploaded = [await self.comfy.upload(p) for p in images[:8]]
        wf = refmod_extract(name, uploaded, f"{profile.name}: {profile.facial_features}")
        await self.comfy.run(wf, settings.workspace_dir / "refmods", f"refmod:{character_id}", expect_files=False)
        await self.comfy.object_info(refresh=True)
        if name not in await self.comfy.available("MiniMaxH3RefModsLoader", "mod_1"):
            raise JobFailed(f"RefMod '{name}' was not found in ComfyUI after extraction")
        profile.refmod_name = name
        self.save_registry()
        await self.emit("identity.refmod", {"character_id": character_id, "refmod": name})
        return name

    async def execute(self, *args: object, **kwargs: object) -> Dict[str, CharacterProfile]:
        await self.emit("identity_registry.loaded", {"count": len(self._characters)})
        return self._characters
