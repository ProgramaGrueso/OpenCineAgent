# Changelog

## 2.0.0 (2026-10-04): real pipeline on MiniMax H3 + Krea 2

The 1.0 agents simulated every model (procedural images, FFmpeg zooms instead of video, sine-wave
voices, a seed-based fake QC). 2.0 renders for real through ComfyUI and was tested on an RTX 4070 Ti
SUPER (16 GB): an 11 s two-shot scene and a 15-shot, 150 s expression showcase with one character.

### Engine
- ComfyUI client: validates each graph against `/object_info` before queueing (missing nodes, missing
  inputs, model files not installed), uploads inputs over HTTP, waits for an idle queue, detects CUDA
  OOM, cancels jobs on timeout, and counts a job as done only when its output file was downloaded.
- `/free` only when the next job uses a different set of models, so consecutive H3 clips keep the
  14 GB UNET loaded.
- Graph builders: H3 reference-to-video (native audio, RefMod, lipsync, SLA), Krea 2 text-to-image
  and Identity Edit, RefMod extraction, MiniMax Music 3. Core `SaveVideo`/`SaveAudio` nodes.

### Agents
- Identity: approved master portrait per character (human gate), RefMod in single-face shots only.
- Shotlist: H3 prompt contract plus a validator that rejects bad shots before any GPU time.
- Keyframes: Krea 2 Identity Edit into a locked location plate, people added one at a time.
- QC: spec, motion, audio, Whisper dialogue and VLM identity checks; unavailable checks are reported
  as "not verified", never as passed. Healing retries with a new seed, shortens clips on OOM and
  keeps the best take.
- Post: frame-exact cut, ducked score, two-pass loudnorm, ClearReality upscale with audio kept.

### Added while testing the 15-shot showcase
- `keyframe_action`: the first frame can show a different state than the shot's full action.
- `on_screen_text`: timed labels and titles burned in post (ASS), since models cannot write text.
- `seed` at script level: the same seed for every shot.
- Phased rendering: all keyframes first, then all H3 clips, then upscaling.
- Keyframe cache: identical keyframe inputs are rendered once (15 shots, 1 render).
- Stage plate prompt says "empty" first and repeats it; Krea 2 mixes otherwise draw a person.

### Measured (RTX 4070 Ti SUPER 16 GB, h3ErosMax 8 steps, 1024x576)
| Test | Result |
|---|---|
| H3 clip, 243 frames, refs at `max`, RefMod | 11.5-12.6 min per clip (cold UNET load adds a few minutes) |
| Same clip with references downscaled to 768 px | 11.8 min, faces indistinguishable: no speed gain, **not adopted** |
| Same clip with `H3SLAAttention` block 64 | stopped after more than 20 min without a result: no conclusion |
| Generating at 864x480 | not tested yet |
| Krea 2 portrait / plate / Identity Edit keyframe | 20 s / 25 s / 70 s |
| ComfyUI RAM with H3 + Krea 2 cached | 31-36 GB; `/free` drops it to ~1.5 GB |

### Known limits
- With ~62 GB RAM, ComfyUI's model cache can push the system into swap; long runs should be started
  from a normal terminal, and `/free` between productions releases the RAM.
- Identity QC needs a vision model in Ollama (`qwen2.5vl:7b`); without it identity is "not verified".
