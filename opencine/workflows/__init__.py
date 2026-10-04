"""ComfyUI API graph builders.

Graphs are built in code (not edited JSON templates) so every parameter has one source of truth,
and they are validated against the live server's /object_info before queueing.
Recipes come from production runs: see docs/PRODUCTION_NOTES.md.
"""

from opencine.workflows.h3 import h3_length, h3_reference_to_video, refmod_extract
from opencine.workflows.krea2 import krea2_identity_edit, krea2_text_to_image
from opencine.workflows.music import minimax_music3

__all__ = [
    "h3_length",
    "h3_reference_to_video",
    "refmod_extract",
    "krea2_identity_edit",
    "krea2_text_to_image",
    "minimax_music3",
]
