---
name: sound-design-skill
description: Music beds (BGM) and sound effects for a video, as WAV files with full provenance, handed to audio-production-skill MIX - generated locally (ACE-Step 1.5 for BGM, EzAudio for SFX, both optional installs, never downloaded) or chosen, trimmed and looped from the user's own library with a declared licence (no engine); seamless loop beds and transition stingers timed to a cut. Use when asked to make background music or a music bed of a given length and mood, a sound effect (a door, footsteps, a whoosh), a loopable bed or a stinger that lands on a cut, to build or search a folder of licensed BGM / SFX, or to fit a library track to a scene. EzAudio runs only after the user has been told about its training data. All ops are EXPERIMENTAL; nothing is mixed here.
---

# sound-design-skill

Run `sound-design <command> ... --json` (installed by `pip install -e .`, Python >= 3.9, stdlib only). With `--json`, stdout holds exactly one JSON document, ASCII only. Probing, cutting, joining and loudness go through ffmpeg-skill, found through `--ffmpeg-skill DIR`, `$SOUND_DESIGN_FFMPEG_SKILL_DIR`, `$VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill` or `../ffmpeg-skill`. `sound-design doctor --json` says which engines are `OK`, `MISSING` or `INCOMPATIBLE`, the compute device and GPU memory, and how to install each engine; it never downloads anything.

| Task | Command |
|---|---|
| A 60 s music bed | `sound-design generate-bgm --prompt "calm piano bed" --tags "ambient,70 bpm" --duration 60 -o bed.wav --seed 42 --json` (10-600 s) |
| A 120 BPM bed for a 15 s lyric video, with its beat grid | `sound-design generate-bgm --prompt "upbeat pop" --duration 15 --bpm 120 --time-signature 4 -o pv.wav --json` (becomes 16 s = 8 bars; `pv.beats.json` beside it) |
| Same, with lyrics / on the CPU | add `--lyrics lyrics.txt` / `--device cpu` (default instrumental, `--device auto`) |
| A bed from the larger 4B XL model (opt-in, a GPU with >= 20 GB) | `sound-design generate-bgm --prompt "cinematic strings" --duration 60 -o bed.wav --model acestep-v15-xl-turbo --json` (see "Choosing the BGM model") |
| A sound effect (after the user agreed, see below) | `sound-design generate-sfx --prompt "a wooden door closes" --tags "indoor" --duration 3 -o door.wav --seed 7 --acknowledge-ezaudio-training-data --json` (whole seconds, 1-10) |
| Add the user's own file to a library | `sound-design library-import track.wav --library ./sounds --licence CC-BY-4.0 --credit "Music: name" --kind bgm --tags piano --mood calm --loopable --json` |
| Add a downloaded stock track with its licence records | `sound-design library-import track.mp3 --library ./sounds --licence CC-BY-4.0 --title "Morning Light" --author "Aiko Tanaka" --licence-url https://creativecommons.org/licenses/by/4.0/ --licence-file LICENSE.txt --provider Freesound --kind bgm --json` (add `--attribution "<line>"` when the source gives one, `--content-id registered` only when the source says so) |
| What is in a library, with licences and lengths | `sound-design library-index ./sounds --loudness --json` |
| A library track fitted to 60 s | `sound-design library-pick ./sounds --kind bgm --tags piano --duration 60 -o bed.wav --json` (add `--file <path>` to name the track, `--crossfade 0.5` for loops) |
| A seamless 8 s loop of a bed (for loops in an edit) | `sound-design loop-bed bed.wav -o loop.wav --duration 8 --search 1 --json` (source at least duration + crossfade long; the loop point is searched within `--search` s) |
| A 1 s whoosh whose peak lands on the cut at 0.5 s | `sound-design stinger whoosh.wav -o stinger.wav --duration 1 --peak-at 0.5 --type whoosh --json` |
| Check the seams of a looped file | `sound-design seam-check loop-x10.wav --period 8 --json` (or `--seams 8,16`; exit 1 when a seam fails) |
| Effects on the cuts and emphasis captions (a cue sheet for MIX) | `sound-design sfx-cues --cuts 1.0,2.5,4.75 --emphasis 3.2 --library ./sounds --min-spacing 0.25 -o cues.json --json` (cut = whoosh, emphasis = pop unless `--effect cut=<name>`; `--file ding=ding.wav` for a generated effect) |
| Same, no file replayed within 30 s, and bright pops | `sound-design sfx-cues --cuts 1.0,12.0,25.0 --emphasis 3.2 --library ./sounds --min-reuse-spacing 30 --character pop=bright -o cues.json --json` (both opt-in; see below) |
| What is installed | `sound-design doctor --json`, `sound-design engines --json` |
| What this skill declares | `sound-design contract --json` (alias `skill --json`) |

Every op takes `--timeout S` and `--ffmpeg-skill DIR`; those that write a file take `--overwrite` (an existing output is refused without it); the generators also take `--seed` (drawn and recorded when omitted) and `--offline`. The output is a new `.wav`; the input is never modified, and an output appears whole or not at all, after ffmpeg-skill `probe` has checked it.

## When to use it

- **The user has music or effects with a known licence** (bought packs, commissioned tracks, CC0 / CC BY): library mode first. `library-import` each file with its licence, then `library-pick` for each scene. No engine runs and nothing is AI-generated (`ai_generated: false`).
- **Nothing suitable in the library:** `generate-bgm` (ACE-Step) or `generate-sfx` (EzAudio), if `doctor` shows the engine `OK`.
- **Then always** hand the WAV and its `provenance` to audio-production-skill MIX, which places it, ducks it under speech and sets the loudness. This skill does not mix, duck or normalise.

## Loops and stingers

- `loop-bed` and `stinger` run no engine. They take a WAV you already have (a `generate-bgm`, `generate-sfx` or `library-pick` result, or the user's file), edit it through ffmpeg-skill only, and measure the result; its licence and engine stay in the source's own provenance, so pass both on.
- A loop's end runs into its start with no click: repeat it end to end, with no crossfade. The result reports the loop length it found (`loop.length`, within `--search` of `--duration`), how alike the crossfaded material is (`loop.similarity`; a warning below 0.5) and the measured wrap (`seam`). A whole number of bars loops best.
- A stinger is exactly `--duration` long and its loudest 10 ms is within 10 ms of `--peak-at` (at least 20 ms from either end), else nothing is written. Tell MIX to start it `--peak-at` seconds before the cut.
- `seam-check` is this skill's own loop check: click (±1 ms), level and octave-band spectrum of 1 s on each side. qc-skill has no loop seam check yet.

## Sound-effect cue sheets

- `sfx-cues` places nothing and decides nothing: the times are the caller's (from the edit's cut list and the emphasis captions), one cue per time. A time closer than `--min-spacing` to the last kept cue is dropped (listed under `dropped` with the reason), never moved. Never invent emphasis times for it.
- Each cue says which file (`file`, `from`: library or a named WAV, `source`: its entry under `sources` with the licence), where it starts (`start`; `file_offset` when it would start before 0), its `gain_db` to reach `--level` dBFS, and whether its peak (whoosh, rise, ...) or its onset (pop, ding, ...) sits on the time. Library effects need a declared licence, chosen like `library-pick` by the effect name as a tag. Hand the sheet to audio-production-skill MIX with the sources' licences.
- **The same file again (opt-in, off by default).** `--min-reuse-spacing S`: an event whose effect would play the same file (same SHA-256) as a cue kept less than S s earlier takes the next library file for that effect in rank order (the first-ranked one whenever it is free; never random). With no other eligible file, or for a `--file` effect, the event is dropped (`dropped[].by: min_reuse_spacing`), never moved and never given another effect. `substitutions` and each cue's `substituted` say which file replaced which. The `--min-spacing` check comes first; a dropped event counts for neither rule. Suggested: 30 s for a talking-head edit; shorter for fast-cut edits, since with N files an effect has at most N cues in any S s; leave it off when one signature sound is wanted. Tell the user when cues were dropped.
- **Character (opt-in).** `--character pop=bright` (or an event's `"character": ["bright"]`, which replaces it) keeps only the library files for that effect whose tags or mood carry every word. It matches what the library already says; it never infers or adds a tag, and no matching file is `INPUT_NOT_FOUND`: import or tag the file, never guess. Words to use in `library-import --tags` / `--mood` and here: `bright`, `dark`, `warm`, `metallic`, `soft`, `hard`, `punchy`, `subtle`, `short`, `long`, `high`, `low`, `rising`, `falling`, `airy`, `playful` (other words match too).

## Ask the user before EzAudio

EzAudio (`generate-sfx`) was trained on YouTube audio (AudioSet, VGGSound, AudioCaps) not licensed for training by its owners. By the owner's decision the user is told this before it runs. Without `--acknowledge-ezaudio-training-data`, `generate-sfx` stops with `CONFIRMATION_REQUIRED` (exit 8) before doing anything; the message gives the caveat in English and Japanese. Show the user that message and pass the flag only after they agree; never add it on your own. `contract --json` (the `generate-sfx` tool) and `doctor --json` (`checks.operations.sfx`) carry `confirmation_required: true` so an orchestrator can ask first. The acknowledgement is recorded in the provenance, and every SFX result repeats the training-data warning: pass it on.

## Licences in library mode

- `library-import` refuses a missing, empty or placeholder licence (`unknown`, `n/a`, ...) with `LICENCE_REQUIRED` (exit 2). Ask the user for the licence; never fill one in.
- A licence is recorded exactly as declared, with `verified: false`. This skill does not check or clear licences: say "declared as CC-BY-4.0", never "cleared".
- `library-index` reports a file without a declared licence as `licence: "unknown"`, and `library-pick` never uses it (it is listed under `excluded` with the reason).
- Pass on the warnings: a non-commercial term, a no-derivatives term on a trimmed or looped file, a CC BY licence without a credit, a file changed since its import. Keep the credit with anything published.
- **Stock tracks.** Import what the user downloaded with what the source states: `--licence`, `--licence-url` and / or `--licence-file` (copied into `licences/`), `--title`, `--author`, `--provider`, `--attribution`, `--content-id`. The skill downloads nothing; `doctor` lists stock sources (`MISSING` without an API key, `MANUAL` without an API) but never calls them. The receipt (`licence.attribution` in the `library-pick` result and provenance, `handoff.credit_line`) carries the credit line: the declared one, else the credit, else for CC BY one composed from declared parts only (warned: check it). Put it in the video's description or credits. A track declared `--content-id registered` is warned about: the upload may get a claim, so tell the user before publishing.
- Fit: within max(0.1 s, 1 %) the file is used whole; a longer file is trimmed (no fade: MIX fades); a shorter file is looped with a crossfade only when it was imported with `--loopable`, else it is excluded. The same library and query always give the same file.

## Limits to tell the user

- **EXPERIMENTAL.** `audio.bgm_generate`, `audio.sfx_generate` and `audio.library_import` have lifecycle `EXPERIMENTAL`. Seed determinism and duration accuracy of the real ACE-Step and EzAudio are not measured yet (the tests use fake engines with the same API). Say the result is generated and should be listened to.
- **Generated music is not checked** for similarity to existing recordings; music recognition and copyright checks are not this skill's job. ACE-Step's commercial-use statement is the vendor's claim.
- **Tempo.** With `--bpm` the length becomes the nearest whole number of bars (a warning says so), so the bed ends on a bar line, and the beat grid (every beat, downbeats, bar lines) is written as `<name>.beats.json` and returned as `tempo_map`. The grid is the requested one with the first beat at 0 s (`measured: false`): ACE-Step was asked for the tempo, but how well it keeps it is not measured. Pass the grid on; do not call it measured.
- ACE-Step runs DiT only (its language model is off until the Qwen3 base licences are verified), so prompts steer style, not structure.
- Read and pass on every entry of `warnings`.

## Engines (optional)

| Op | Engine | Install |
|---|---|---|
| `generate-bgm` | ACE-Step 1.5 (MIT code and weights; 2B turbo by default, 4B XL turbo with `--model`), Python 3.11-3.12 in its own checkout and `.venv` | `doctor` prints the exact line; set `SOUND_DESIGN_ACESTEP_PYTHON` (and `SOUND_DESIGN_ACESTEP_CHECKPOINTS`, else the Hugging Face cache) |
| `generate-sfx` | EzAudio-XL (MIT code and weights, Flan-T5-XL Apache-2.0) in its own checkout and `.venv` | `doctor` prints the exact line; set `SOUND_DESIGN_EZAUDIO_PYTHON` and `SOUND_DESIGN_EZAUDIO_DIR` (checkout with `ckpts/`); Flan-T5-XL in the Hugging Face cache |
| library mode | none | nothing to install beyond ffmpeg-skill |

### Choosing the BGM model

`generate-bgm` runs ACE-Step's 2B turbo model, `acestep-v15-turbo`, unless `--model` says otherwise (request key `model`). It needs less than 4 GB of GPU memory and also runs on the CPU (slowly). `--model acestep-v15-xl-turbo` selects the XL turbo model: a 4B DiT, MIT like the default, with commercial use of the output stated on its own model card. Choose it from `doctor --json`, `checks.operations.bgm.models`, where each model has `ready`, its files, `vram` and `install`:

- **GPU memory.** XL needs a GPU with at least 20 GB, because this skill runs it without CPU offload or quantization. Its model card lists 12 GB (with offload and INT8), 16 GB (with offload), 20 GB (no offload) and 24 GB (with the 4B LM). Below 20 GB, or with no GPU, `generate-bgm --model acestep-v15-xl-turbo` stops before anything runs with `INSUFFICIENT_VRAM` (exit 3). Tell the user, and offer the default model (`error.details.default_model`): drop `--model`. Apple Silicon counts its unified memory, which the CPU shares. When the GPU memory cannot be measured (ROCm, Intel XPU), the bed is made and a warning says it was not checked. An Intel GPU is only noticed through `xpu-smi`; on one without it, pass `--device xpu`.
- **Download.** About 20 GB more on disk, into the same checkpoints directory as the main model (`SOUND_DESIGN_ACESTEP_CHECKPOINTS`; a Hugging Face cache snapshot does not work for XL). Without it, the call stops with `MODEL_NOT_FOUND` and `error.details.install` gives the line, for example `uv run acestep-download --model acestep-v15-xl-turbo --dir "$SOUND_DESIGN_ACESTEP_CHECKPOINTS"`. If the main model is what is missing, the line is the main model's. Ask the user before downloading.
- **The record.** The result's `engine.model`, `engine.weights_license` and `engine.output_use`, and `provenance.parameters.model` (`selected_by`, `repository`, `vram`), name the model that ran. Pass them on as you do for the default.
- **Not measured.** XL has not been run on a real GPU by this skill's tests; the 20 GB figure is its model card's. Say so if the user asks how well it works.

When an op fails with `ENGINE_UNAVAILABLE` or `MODEL_NOT_FOUND`, `error.details.install` holds the line to pass on. Do not download or install an engine without the user's go-ahead: the skill never downloads weights, and its worker refuses every non-loopback connection.

## Errors

With `--json`, a failure is `{"ok": false, "error": {"code", "message", "retryable", "details"}}`, and the code fixes the exit code.

| Code | Exit | Retry? | Meaning, and what to do |
|---|---|---|---|
| `INVALID_ARGUMENT` | 2 | no | A flag value is malformed (duration out of the engine's range, seed, prompt too long); fix the call. |
| `INPUT_NOT_FOUND` | 2 | no | An input file does not exist or is not a file. |
| `UNSUPPORTED_MEDIA` | 2 | no | ffmpeg-skill cannot read the file as audio only. |
| `OUTPUT_EXISTS` | 2 | no | The output exists; pick another path or pass `--overwrite` deliberately. |
| `OUTPUT_IS_INPUT` | 2 | no | The output would replace an input; never allowed. |
| `LICENCE_REQUIRED` | 2 | no | A library file has no declared licence; ask the user for it. |
| `FFMPEG_SKILL_UNAVAILABLE` | 3 | no | No usable ffmpeg-skill, or it lacks a tool or flag this op needs; the message names the gap. |
| `ENGINE_UNAVAILABLE` | 3 | no | Unknown engine, or not installed; pass on `details.install`. |
| `MODEL_NOT_FOUND` | 3 | no | The engine is installed but its weights are not on disk; pass on `details.install`. |
| `TOOL_ERROR` | 4 | yes | An ffmpeg-skill tool failed or timed out. |
| `ENGINE_FAILED` | 4 | no | The engine ran and failed; `details` says why. |
| `ENGINE_TIMEOUT` | 4 | yes | The engine exceeded `--timeout` and was killed; retry with a longer one. |
| `INVALID_RESULT` | 4 | no | The output failed validation (duration, rate, streams); nothing was written. |
| `NOT_IMPLEMENTED` | 5 | no | Declared in the contract but not built yet. |
| `CANCELLED` | 6 | yes | Interrupted by a signal. |
| `INTERNAL_ERROR` | 7 | no | A bug in this skill; report it. |
| `CONFIRMATION_REQUIRED` | 8 | no | EzAudio needs the user's acknowledgement first; ask, then add `--acknowledge-ezaudio-training-data`. |
| `INSUFFICIENT_VRAM` | 3 | no | The `--model` chosen needs more GPU memory than this machine has (XL: 20 GB); drop `--model` to use `details.default_model`. |

## Not this skill's job

Mixing, ducking and loudness targets (audio-production-skill MIX), voice, narration or dubbing (dubbing-skill), noise reduction and voice isolation (voice-enhance-skill), music recognition, licence clearance or copyright checks, downloading engines or weights (a one-time manual step, `doctor` says how), direct ffmpeg calls (every media step is an ffmpeg-skill tool).
