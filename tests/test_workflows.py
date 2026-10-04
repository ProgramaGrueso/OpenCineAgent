"""Graph builders and client-side validation against /object_info."""

import pytest

from opencine.core.comfy_client import ComfyUIClient, WorkflowInvalid
from opencine.core.config import settings
from opencine.workflows import (
    h3_reference_to_video,
    krea2_identity_edit,
    krea2_text_to_image,
    minimax_music3,
    refmod_extract,
)


def classes(wf):
    return {n["class_type"] for n in wf.values()}


def assert_links_resolve(wf):
    for node_id, node in wf.items():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in wf, f"{node_id} links to missing {value[0]}"


def test_h3_native_audio_graph():
    wf = h3_reference_to_video("p", ["kf.png", "elena.png"], 1024, 576, 158, 7, "x")
    assert_links_resolve(wf)
    assert {"VAEDecodeAudio", "CreateVideo", "SaveVideo", "LoraLoaderModelOnly"} <= classes(wf)
    assert "PixaromaH3AudioSync" not in classes(wf) and "MiniMaxH3RefModsLoader" not in classes(wf)
    assert wf["r2v"]["inputs"]["ref_images.ref_image_1"] == ["img1", 0]
    assert wf["r2v"]["inputs"]["ref_image_size"] == "max"
    assert wf["ks"]["inputs"]["steps"] == settings.h3_steps


def test_h3_refmod_and_lipsync_graph():
    wf = h3_reference_to_video("p", ["kf.png"], 1024, 576, 124, 7, "x", refmod="m_refmod", voice_track="v.wav")
    assert_links_resolve(wf)
    assert {"MiniMaxH3RefModsLoader", "MiniMaxH3RefModApply", "PixaromaH3AudioSync", "LoadAudio"} <= classes(wf)
    assert wf["ks"]["inputs"]["positive"] == ["refmod_apply", 0]
    assert wf["video"]["inputs"]["audio"] == ["sync", 2]
    loader = wf["refmods"]["inputs"]
    assert all(f"strength_{i}" in loader and f"copies_{i}" in loader for i in range(1, 9))


def test_h3_without_turbo_lora_and_with_sla(monkeypatch):
    monkeypatch.setattr(settings, "h3_turbo_lora", None)
    monkeypatch.setattr(settings, "h3_sla", True)
    wf = h3_reference_to_video("p", ["kf.png"], 1024, 576, 124, 7, "x")
    assert "LoraLoaderModelOnly" not in classes(wf)
    assert wf["sla"]["inputs"]["protect_audio"] is True
    assert wf["ks"]["inputs"]["model"] == ["sla", 0]


def test_krea2_graphs(monkeypatch):
    edit = krea2_identity_edit("p", "scene.png", "subject.png", 1536, 864, 3, "x")
    assert_links_resolve(edit)
    assert edit["patch"]["inputs"]["ref_boost"] == 4.0
    assert edit["patch"]["inputs"]["source_image"] == ["img_scene", 0]
    single = krea2_text_to_image("p", 832, 1216, 1, "x", lora=("char.safetensors", 1.0))
    assert_links_resolve(single)
    assert single["ks"]["inputs"]["model"] == ["lora", 0]
    monkeypatch.setattr(settings, "krea2_hires_pass", True)
    hires = krea2_text_to_image("p", 832, 1216, 1, "x")
    assert_links_resolve(hires)
    assert hires["latent"]["inputs"]["width"] == 416 and "LatentUpscaleBy" in classes(hires)


def test_refmod_and_music_graphs():
    wf = refmod_extract("name", ["a.png", "b.png"], "desc")
    assert_links_resolve(wf)
    assert wf["extract"]["inputs"]["mode"] == "encode" and "PreviewAny" in classes(wf)
    music = minimax_music3("cap", "", 30, 1, "x")
    assert_links_resolve(music)
    assert "SaveAudio" in classes(music)


OBJECT_INFO = {
    "UNETLoader": {"input": {"required": {"unet_name": [["model_a.safetensors"]], "weight_dtype": [["default"]]}}},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {}]}}},
}


async def test_validate_reports_every_problem():
    client = ComfyUIClient("http://unused")
    client._object_info = OBJECT_INFO
    ok = {"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "model_a.safetensors", "weight_dtype": "default"}},
          "k": {"class_type": "KSampler", "inputs": {"model": ["u", 0], "seed": 1}}}
    await client.validate(ok)
    bad = {"u": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing.safetensors", "weight_dtype": "default"}},
           "k": {"class_type": "KSampler", "inputs": {"model": ["nope", 0]}},
           "x": {"class_type": "NotANode", "inputs": {}}}
    with pytest.raises(WorkflowInvalid) as err:
        await client.validate(bad)
    text = str(err.value)
    assert "model file not installed: 'missing.safetensors'" in text
    assert "missing required input 'seed'" in text
    assert "links to missing node 'nope'" in text
    assert "'NotANode' is not installed" in text


def test_job_error_flags_oom():
    status = {"status_str": "error", "messages": [["execution_error", {
        "node_type": "KSampler", "node_id": "ks", "exception_message": "torch.OutOfMemoryError: Allocation ..."}]]}
    err = ComfyUIClient._job_error(status)
    assert err.oom and "KSampler" in str(err)


async def test_free_only_when_model_set_changes(monkeypatch, tmp_path):
    client = ComfyUIClient("http://unused")
    freed = []

    async def noop(*_a, **_k):
        return None

    async def fake_free():
        freed.append(True)
        client._loaded_models = None

    out = tmp_path / "x.mp4"
    out.write_bytes(b"data")

    async def fake_download(history, dest, kinds=()):
        return [out]

    async def fake_queue(wf):
        return "pid"

    async def fake_wait(pid, timeout):
        return {}

    for name, fn in (("validate", noop), ("wait_idle", noop), ("free", fake_free), ("queue", fake_queue),
                     ("wait", fake_wait), ("download_outputs", fake_download)):
        monkeypatch.setattr(client, name, fn)
    monkeypatch.setattr(client, "journal_path", tmp_path / "journal.jsonl")
    h3 = h3_reference_to_video("p", ["kf.png"], 64, 64, 124, 1, "x")
    krea = krea2_text_to_image("p", 64, 64, 1, "x")
    for wf in (krea, h3, h3, h3, krea):
        await client.run(wf, tmp_path, "job")
    assert len(freed) == 3   # krea (first), switch to h3, switch back to krea
