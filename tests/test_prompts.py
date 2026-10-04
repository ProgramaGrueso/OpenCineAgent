"""H3 prompt contract, frame lengths and seeds."""

from opencine.core.models import CharacterProfile, DialogueLine, ShotSpecification, ShotType
from opencine.core.prompts import ALIGNMENT_LINE, validate_video_prompt, video_prompt
from opencine.core.seeds import stable_seed
from opencine.workflows import h3_length

ELENA = CharacterProfile(id="elena", name="Elena", description="a woman in her thirties",
                         facial_features="green eyes, black ponytail", invariable_clothing="a navy wool coat",
                         voice_description="a low voice")
VICTOR = CharacterProfile(id="victor", name="Victor", description="a man in his fifties",
                          facial_features="platinum hair", invariable_clothing="a charcoal suit")


def shot(**kw):
    base = dict(shot_id="SH_1", action_description="She looks at the camera", stage_id="s",
                character_ids=["elena"], shot_type=ShotType.CLOSE_UP)
    base.update(kw)
    return ShotSpecification(**base)


def test_video_prompt_follows_contract():
    s = shot(dialogue=[DialogueLine(speaker_id="elena", text="You knew it.")], sound_description="Rain")
    prompt = video_prompt(s, [ELENA], "Spanish", "A rooftop", "photorealistic")
    assert prompt.startswith(ALIGNMENT_LINE)
    assert "<Subject 1> is Elena" in prompt and "<Picture 2>" in prompt
    assert "<d>[Spanish] You knew it.</d>" in prompt and "(S1)" in prompt
    assert "overall_soundscape: Rain. Only <Subject 1> speaks." in prompt
    assert prompt.endswith("non_diegetic_music: No music.")
    assert validate_video_prompt(prompt) == []


def test_silent_partner_and_two_subjects():
    s = shot(character_ids=["elena", "victor"], dialogue=[DialogueLine(speaker_id="victor", text="No.")])
    prompt = video_prompt(s, [ELENA, VICTOR], "English", "A rooftop", "film")
    assert "<Subject 2> is Victor" in prompt and "<Picture 3>" in prompt
    assert "<Subject 1> stays silent with closed mouth." in prompt
    assert "Only <Subject 2> speaks." in prompt


def test_validator_catches_contract_violations():
    good = video_prompt(shot(), [ELENA], "English", "A rooftop", "film")
    assert validate_video_prompt(good) == []
    assert any("[Shot 1]" in i for i in validate_video_prompt(good + " [Shot 2] cut"))
    assert any("unbalanced" in i for i in validate_video_prompt(good + " <d>[English] hi"))
    assert any("[Language]" in i for i in validate_video_prompt(good + " <d>hi</d>"))
    long_line = video_prompt(shot(dialogue=[DialogueLine(speaker_id="elena", text="x" * 140)]),
                             [ELENA], "English", "A rooftop", "film")
    assert any("130" in i for i in validate_video_prompt(long_line))
    assert any("alignment" in i for i in validate_video_prompt("hello"))


def test_h3_length_uses_trained_grid():
    for seconds in (1, 5, 5.2, 6, 8, 12.5, 15):
        frames = h3_length(seconds)
        assert (frames - 5) % 17 == 0
        assert 124 <= frames <= 362
        assert frames >= min(seconds * 24, 362)
    assert h3_length(5) == 124 and h3_length(6) == 158


def test_stable_seed_is_deterministic():
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
    assert 0 <= stable_seed("x") < 2**31 - 1
