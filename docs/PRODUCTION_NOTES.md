# Production notes: MiniMax H3 + Krea 2 in ComfyUI

These rules come from producing music videos, commercials and a 40+ episode short-form series
with this stack on a single 16 GB GPU (RTX 4070 Ti SUPER). Each one is something that failed
first. OpenCineAgent implements them; this page explains why.

## Order of work

1. **Write and review the plan before rendering.** `python main.py plan ...` writes `plan.json`
   with every compiled prompt. Reading it is free; a wrong graph discovered after 16 renders is not.
2. **Approve each master portrait by eye.** If the identity anchor is the wrong face, every later
   check validates the model against its own output. A human "yes, that's her" costs ten seconds.
3. **Render the hardest shot first** (`run --only SH_00X`). It validates the whole graph.
4. **Render the rest in storyboard order.** If a run stops, you still have a continuous film from second 0.
5. **Keep every take.** Corrections are new takes (`_t2`, `_t3`), approved files are never overwritten.

## Identity

| Rule | Why |
|---|---|
| One master portrait per character; every keyframe derives from it with Krea 2 Identity Edit | Two separate text-to-image generations of "the same" person drift apart |
| Identity block repeated verbatim in every prompt (description, face, wardrobe) | Wardrobe that is not pinned changes between shots |
| Describe material and color separately ("a wool trench coat in navy blue") | Mixed descriptions drift (champagne turned pink across a whole video) |
| RefMod only in shots with one face | A RefMod pushes identity on the whole conditioning: two faces become clones |
| `ref_image_size: max` | `match` scales references to the render size and averages two faces into one |
| Two or three people: add them one Identity Edit at a time | One prompt with several people merges faces or adds a third one |
| Identity anchors are the same files in every clip | Regenerating anchors per clip degrades identity by clip 6 |

## H3 prompt contract

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

<Subject 1> is ... shown in <Picture 1> and in <Picture 2>. The face ... strictly preserved ... with zero alteration.

integrated_multimodal_description: [Shot 1] <framing>. <action>. <Subject 1>, <emotion>, in <voice> (S1), says:
<d>[Spanish] ...</d> The lips of <Subject 1> articulate every word. <camera>. <location>. <style>. No text ...
overall_soundscape: <ambience and action sounds>. Only <Subject 1> speaks.
non_diegetic_music: No music.
```

- Only `[Shot 1]`. Internal cuts are prose ("Quick cut to a close-up of ...").
- One speaker per clip; a second voice gets `(S2)`, never the same id.
- Lines up to ~130 characters, ~200 per clip. Longer lines are rushed or cut.
- To make someone talk, have them stop what they are doing and look at the camera: if they eat while
  talking, H3 prioritizes eating.
- Music is added in post. H3 "re-sings" or restarts music per clip.
- No text in renders. Models invent letters; all titles and subtitles go in post.

## Lengths and memory

- H3 is trained on **124-362 frames** at 24 fps (`5 + 17k`). Render the next valid length and trim
  at assembly; trimming per clip keeps cuts from drifting.
- VRAM is filled by **frames x reference images**, not references alone. On OOM, shorten the clip
  (OpenCineAgent retries at 124 frames).
- One heavy job at a time on 16 GB. The ComfyUI queue serializes jobs but does not release VRAM
  between clients: call `/free` between jobs (done automatically). A video player open on the GPU
  can be enough to cause an OOM.
- A cold load of the H3 UNET takes several minutes the first time; it is not hung.
- Measured on a 4070 Ti SUPER: a 243-frame clip at 1024x576 with `max` references and a RefMod takes
  11.5-12.6 min. Downscaling the references to 768 px did not make it faster (the frames dominate the
  cost), so references stay at full size. See CHANGELOG.md for the other measurements.
- ComfyUI keeps loaded models in RAM (31-36 GB with H3 + Krea 2). On a 64 GB machine that leaves
  little room: `/free` releases it between productions.

## ComfyUI API pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `Required input is missing` with a preset | The API needs every required field even when a preset overrides them | Builders write every field explicitly |
| `Node X not found` although it is installed | ComfyUI started partially (port fight, DB lock) | Compare `/object_info` node count with normal; set `OPENCINE_COMFYUI_MIN_NODES` |
| Job "succeeded" in seconds, no file | Exit code / queue state trusted instead of the artifact | Success = downloaded, non-empty output |
| Killing your script leaves the GPU busy | The ComfyUI queue outlives the client | Timeouts delete the job by id and `/interrupt` |
| `SaveAudioAdvanced` / `SaveVideo` TypeError on `format` | Dynamic combos | Core `SaveAudio`; `SaveVideo` with flat `format: mp4, codec: h264` |
| Output not found | Image nodes return `images`, audio nodes `audio` | The client reads every output kind |
| `CUDA error: illegal memory access` kills ComfyUI | Triton kernel of `H3SLAAttention` at block 32 on some builds | `OPENCINE_H3_SLA=false` or block 64 |
| Extract node runs nothing | `MiniMaxH3RefModExtract` is not an output node | A `PreviewAny` terminal node triggers it |

## Audio and finishing

- The master carries H3's native dialogue and sound, trimmed with each clip, plus the score ducked
  under it. Loudness: fixed levels, then two-pass linear `loudnorm` to -14 LUFS.
- Filters that blend layers (glow, screen) must run in RGB (`format=gbrp`); in YUV they tint magenta.
  Measure color, not only brightness, before calling a fade "white".
- Upscaling with 4x-ClearRealityV1 + area downsampling is sharper than lanczos; the upscaler writes
  video only, so the original audio is muxed back.

## Quality control

"Verified" means measured. The QC report separates **passed**, **failed** and **not verified**
(tool or model missing). Whisper catches cut or invented lines; the VLM compares frames with the
approved portraits. A bad number is a reason to look at the frames, not to discard a take blindly.
