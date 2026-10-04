"""Prompt contracts for Krea 2 and MiniMax H3.

The H3 structure below is the one that held identity and lipsync across real productions:
  alignment line -> subject identity block -> [Shot 1] action with <d>[Language] ...</d> lines
  -> overall_soundscape -> non_diegetic_music
Rules: a single [Shot 1] (internal cuts are written as prose), one speaker per clip when possible,
short lines, the identity block repeated verbatim in every shot, music added in post ("No music.").
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from opencine.core.models import CameraMotion, CharacterProfile, ShotSpecification, ShotType

ALIGNMENT_LINE = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
)
MAX_LINE_CHARS = 130
MAX_SHOT_DIALOGUE_CHARS = 200
NO_TEXT = "No text, no captions, no subtitles appear anywhere."

FRAMING: Dict[ShotType, str] = {
    ShotType.EXTREME_WIDE: "Extreme wide establishing shot, the figures small in the vast location",
    ShotType.WIDE: "Wide shot, full body visible in the environment",
    ShotType.MEDIUM: "Medium shot from the waist up, natural conversational framing",
    ShotType.CLOSE_UP: "Intimate close-up from the chest up, sharp focus on the eyes, shallow depth of field",
    ShotType.EXTREME_CLOSE_UP: "Extreme close-up on the face, detailed skin texture",
}
CAMERA: Dict[CameraMotion, str] = {
    CameraMotion.STATIC: "Locked-off camera with subtle natural breathing.",
    CameraMotion.PAN_LEFT: "The camera pans smoothly from right to left.",
    CameraMotion.PAN_RIGHT: "The camera pans smoothly from left to right.",
    CameraMotion.TILT_UP: "The camera tilts up slowly.",
    CameraMotion.TILT_DOWN: "The camera tilts down slowly.",
    CameraMotion.DOLLY_IN: "Slow controlled push-in toward the subject.",
    CameraMotion.DOLLY_OUT: "Slow pull-out revealing the surroundings.",
    CameraMotion.CRANE_UP: "The camera cranes upward in a sweeping move.",
    CameraMotion.ORBIT_360: "The camera orbits slowly around the subject.",
    CameraMotion.TRACKING: "Steady tracking shot moving with the subject.",
    CameraMotion.HANDHELD: "Handheld subtle camera motion.",
}


def _speaker_order(shot: ShotSpecification) -> List[str]:
    order: List[str] = []
    for line in shot.dialogue:
        if line.speaker_id not in order:
            order.append(line.speaker_id)
    return order


def video_prompt(shot: ShotSpecification, cast: Sequence[CharacterProfile], language: str,
                 location: str, visual_style: str) -> str:
    """MiniMax H3 prompt. <Picture 1> is the keyframe; <Picture k+1> is the master portrait of cast[k-1]."""
    subject_of = {c.id: f"<Subject {i}>" for i, c in enumerate(cast, 1)}
    blocks = []
    for i, c in enumerate(cast, 1):
        blocks.append(
            f"<Subject {i}> is {c.name}, {c.description}, shown in <Picture 1> and in <Picture {i + 1}>. "
            f"The face, facial structure, eyes, nose, lips and identity of <Subject {i}> are strictly preserved "
            f"from <Picture {i + 1}> with zero alteration: {c.facial_features}. "
            f"<Subject {i}> wears {c.invariable_clothing}."
        )

    speakers = _speaker_order(shot)
    voice_tag = {cid: f"S{n}" for n, cid in enumerate(speakers, 1)}
    by_id = {c.id: c for c in cast}
    lines = []
    for line in shot.dialogue:
        who = subject_of.get(line.speaker_id, "A voice off-screen")
        voice = by_id[line.speaker_id].voice_description if line.speaker_id in by_id else "a clear voice"
        lines.append(
            f"{who}, {line.emotion}, in {voice} ({voice_tag[line.speaker_id]}), says: "
            f"<d>[{language}] {line.text.strip()}</d> The lips of {who} articulate every word."
        )
    silent = [subject_of[c.id] for c in cast if c.id not in speakers]
    if shot.dialogue and silent:
        lines.append(f"{' and '.join(silent)} {'stays' if len(silent) == 1 else 'stay'} silent with closed mouth.")

    action = shot.action_description.strip().rstrip(".")
    description = (
        f"[Shot 1] {FRAMING[shot.shot_type]}. {action}. {' '.join(lines)} {CAMERA[shot.camera_motion]} "
        f"{location.strip().rstrip('.')}. {visual_style.strip().rstrip('.')}. {NO_TEXT}"
    )
    description = re.sub(r"\s+", " ", description).strip()

    if speakers:
        names = " and ".join(subject_of.get(s, "the off-screen voice") for s in speakers)
        who_speaks = f"Only {names} {'speaks' if len(speakers) == 1 else 'speak'}."
    else:
        who_speaks = "No one speaks."
    sound = (shot.sound_description.strip().rstrip(".") + ". ") if shot.sound_description.strip() else ""

    parts = [ALIGNMENT_LINE, ""]
    if blocks:
        parts += [" ".join(blocks), ""]
    parts += [
        f"integrated_multimodal_description: {description}",
        f"overall_soundscape: {sound}{who_speaks}",
        "non_diegetic_music: No music.",
    ]
    return "\n".join(parts)


def validate_video_prompt(prompt: str) -> List[str]:
    """Returns contract violations (empty list = valid)."""
    issues = []
    if not prompt.startswith(ALIGNMENT_LINE):
        issues.append("must start with the alignment line")
    if "[Shot 1]" not in prompt:
        issues.append("missing [Shot 1]")
    if re.search(r"\[Shot [2-9]\]", prompt):
        issues.append("only [Shot 1] is allowed; write internal cuts as prose ('Quick cut to ...')")
    if prompt.count("<d>") != prompt.count("</d>"):
        issues.append("unbalanced <d> ... </d> dialogue tags")
    spoken = re.findall(r"<d>\[(\w+)\]\s*(.*?)</d>", prompt, re.S)
    if prompt.count("<d>") != len(spoken):
        issues.append("every <d> must start with a [Language] tag, e.g. <d>[Spanish] ...</d>")
    for _, text in spoken:
        if len(text) > MAX_LINE_CHARS:
            issues.append(f"line longer than {MAX_LINE_CHARS} chars (H3 rushes or cuts it): '{text[:40]}...'")
    if sum(len(t) for _, t in spoken) > MAX_SHOT_DIALOGUE_CHARS:
        issues.append(f"more than {MAX_SHOT_DIALOGUE_CHARS} chars of dialogue in one clip; split the shot")
    for field in ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"):
        if field not in prompt:
            issues.append(f"missing field '{field}'")
    if len(set(re.findall(r"\((S\d)\)", prompt))) > 2:
        issues.append("more than two speakers in one clip")
    return issues


def master_portrait_prompt(c: CharacterProfile, style: str) -> str:
    trigger = ", ".join(c.trigger_words)
    return (
        f"vertical medium full shot portrait, {trigger + ', ' if trigger else ''}{c.description}, "
        f"{c.facial_features}, wearing {c.invariable_clothing}, standing facing the camera with a calm neutral "
        f"expression, plain light grey seamless studio background, soft even light, {style}, no text anywhere."
    )


def stage_plate_prompt(location: str, time_of_day: str, mood: str, style: str) -> str:
    # Krea 2 (and its mixes) lean toward portraits: say "empty" first and repeat it, or a person appears.
    return (
        f"An empty, unoccupied location with nobody in it: {location.strip().rstrip('.')}, {time_of_day}, "
        f"{mood} atmosphere, wide establishing view. The scene is completely empty: no people, no person, "
        f"no figure, no face, no body. {style}, no text anywhere."
    )


def keyframe_prompt(shot: ShotSpecification, cast: Sequence[CharacterProfile], location: str, style: str,
                    step: int = 0) -> str:
    """Krea 2 Identity Edit prompt for placing cast[step] into the current image.
    Step 0 places the first person into the empty plate; later steps add people while
    keeping everything already in the frame (chained edits avoid merged faces)."""
    framing = FRAMING[shot.shot_type].lower()
    action = (shot.keyframe_action or shot.action_description).strip().rstrip(".")
    if not cast:
        return f"{FRAMING[shot.shot_type]}. {action}. {location}. {style}, no text anywhere."
    c = cast[step]
    keep = ("Keep the location exactly as it is." if step == 0 else
            "Keep everyone already in the image, their faces, their clothes and the location exactly as they are.")
    verb = "Place" if step == 0 else "Add"
    return (
        f"{verb} this person, {c.name}, into this scene: {action}; {framing}. {keep} "
        f"Keep this person's exact face, facial structure, eyes, nose, lips and skin tone: {c.facial_features}. "
        f"This person wears {c.invariable_clothing}. {location.strip().rstrip('.')}. {style}, "
        f"no new text, no captions, no watermark."
    )
