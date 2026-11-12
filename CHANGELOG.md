# Changelog

## 0.3.2 - unreleased

Roadmap step 8, the measurement half (ADR-018). No CLI, contract or op change.

- An integration test runs ffmpeg-skill `scenes --beats` on `generate-bgm --bpm` output and compares it with the written `<name>.beats.json`. The cases are 120 BPM 4/4 for 15 s, 90 BPM 3/4 for 20 s and 137 BPM 4/4 for 12 s. Every written beat has a measured one within 20 ms (measured: at most 1.07 ms), and the measured tempo is the requested one. On ffmpeg-skill before 3.2.2, which refuses a WAV, the refusal is asserted instead.
- The test engine's stereo was in antiphase (right = -left), so a mono downmix was silent and `scenes` found no beats. It is now in phase.
- Step 8 stays open: the bed measured is the test engine's, which puts its clicks on the grid. Real ACE-Step output still needs an engine install.

## 0.3.1 - unreleased

Roadmap step 7 closed (ADR-017). No CLI, contract or op change.

- A `loop-bed` result repeated ten times now passes qc-skill's own loop seam check, as the Done-when asks. This is qc-skill 1.3.0's `AudioRule.loop_seam`, run through `qc run`, at the wrap and all nine seams. It is given this skill's `seam-check` limits: window 1 s, click 3, level 1.5 dB, band 6 dB. Two beds pass: the periodic fixture and a new drifting bed that never repeats exactly. A naive cut fails on the click. SKILL.md and the README give the rule to hand to qc-skill.
- CI's integration job checks out qc-skill main (`SOUND_DESIGN_QC_SKILL_DIR`).

## 0.3.0 - unreleased

GENERATE_BGM model selection (ADR-016). Opt-in; without `--model` a bed is made as before.

- `generate-bgm --model acestep-v15-turbo|acestep-v15-xl-turbo` (request key `model`). The default stays `acestep-v15-turbo` (2B). `acestep-v15-xl-turbo` is ACE-Step 1.5 XL turbo (`ACE-Step/acestep-v15-xl-turbo`, about 4B parameters, MIT; its model card states the output may be used commercially), loaded through the same `initialize_service` / `generate_music` call with `config_path` set to it.
- Checked before anything runs: the XL files in the checkpoints directory (`SOUND_DESIGN_ACESTEP_CHECKPOINTS`; about 20 GB more), else `MODEL_NOT_FOUND` with the download line; and a GPU with at least 20 GB (the model card's figure without CPU offload; this skill passes no offload or quantization), else the new error `INSUFFICIENT_VRAM` (exit 3) naming the default model to fall back to. GPU memory that cannot be measured (ROCm; Intel XPU, now noticed through `xpu-smi`, or `--device xpu` where no GPU was found) is allowed with a warning.
- The result and provenance carry the model that ran with its own licence and output-use statement (`engine.model`, `engine.weights_license`, `engine.output_use`, `engine.training_data`), `request.model`, and `provenance.parameters.model` (`selected_by`, repository, revision, the memory check).
- `doctor` lists each BGM model under `checks.operations.bgm.models` (files, `vram`, `ready`, `install`); a missing XL model is not a problem, an XL model this GPU cannot run is a warning. The contract publishes the models (`engines.registered.acestep.models`, `default_model`), the flag in the generate-bgm `argv`, `device.model_fit`, `policy.models` and `INSUFFICIENT_VRAM`.

## 0.2.0 - unreleased

SFX cue variety (ADR-015). Both opt-in; without them a cue sheet is as before.

- `sfx-cues --min-reuse-spacing <s>` (request key `min_reuse_spacing`, default off): an event whose effect would play the same sound file (same SHA-256) as a cue kept less than that many seconds earlier takes the first file in that effect's library ranking that no kept cue played within that time (rank order, never random). With none, or for a named `--file`, the event is dropped with its reason, never moved. `substitutions` and each cue's `substituted` say which file replaced which; every drop carries `by` (`min_spacing` or `min_reuse_spacing`). The `--min-spacing` check comes first, and a dropped event counts for neither rule.
- `sfx-cues --character <effect>=<word,...>` (request key `characters`; per event `character`): only library files whose tags or mood carry every word, e.g. a bright pop. No library field is added and no tag is inferred; no match is `INPUT_NOT_FOUND`. A suggested vocabulary is in SKILL.md, the README and the contract.
- Each cue names its entry under `sources` (`source`); sources carry `effect`, `character` and `rank`. The schema stays `sound-design/sfx-cues@1`.

## 0.1.0 - unreleased

Stock music and SFX with licence records (roadmap step 10, ADR-014).

- `library-import` gains `--licence-url`, `--licence-file` (the downloaded licence text, copied into `<library>/licences/<sha256>.txt`), `--title`, `--provider`, `--attribution` and `--content-id registered|not-registered` (only as the source states it). `--licence` stays required (`LICENCE_REQUIRED` as before).
- The licence block (import, index, pick, cue sheets, provenance) carries `attribution`: the declared line, else the credit, else for CC BY one composed from declared parts in TASL order (warned). `library-pick`'s handoff adds `credit_line` and `content_id`.
- A track declared registered for Content ID is warned about on import, index and pick.
- `doctor` reports `checks.stock_sources`: Freesound and Jamendo `MISSING` until their API key is set (`KEY_SET` after), YouTube Audio Library `MANUAL`. Nothing is ever fetched; doctor's status is unchanged.

Sound-effect cue sheet (roadmap step 9, ADR-013).

- `sound-design sfx-cues [--cuts <t,...>] [--emphasis <t,...>] [--events <json>] [--library <dir>] [--effect cut|emphasis=<name>] [--file <name>=<wav>] [--min-spacing 0.25] [--level -12] [-o cues.json] --json` (schema `sound-design/sfx-cues@1`): one cue per caller time (none added, none moved; one closer than the spacing to the last kept cue is dropped with the reason), effects from the library (declared licence) or named WAVs, the effect's peak or onset on the time, `gain_db` to the level. For audio-production-skill MIX to place; nothing is mixed.

Tempo-targeted GENERATE_BGM (roadmap step 8, ADR-012).

- `generate-bgm --bpm <30-300> [--time-signature 2|3|4|6]`: the tempo and time signature go to ACE-Step as `GenerationParams.bpm` / `timesignature` (read at ca1e85fe), the length becomes the nearest whole number of bars inside the engine's range (120 BPM and 15 s become 16 s), and the beat grid is written beside the WAV as `<name>.beats.json` (schema `sound-design/beat-grid@1`) and returned as `tempo_map`. The grid is the requested one (`measured: false`); the engine's adherence is not measured.
- Not done: ffmpeg-skill `scenes --beats` refuses audio-only input and, on a video wrapper, resolves a synthetic click track's phase 40-60 ms off, so the roadmap's 20 ms agreement cannot be shown; no real ACE-Step output exists here.

Loopable beds, timed stingers and a seam check (roadmap step 7, ADR-011). No engine; edits through ffmpeg-skill only.

- `sound-design loop-bed <in.wav> -o loop.wav --duration <s> [--search 1] [--crossfade 0.25] --json` (schema `sound-design/loop-bed@1`): the loop point is searched within `--search` s of `--duration`; the source's continuation is crossfaded into its start (ffmpeg-skill `cut`, `join`, `cut`), so the wrap is the source's own continuity; the wrap is measured and a failing one refused (`INVALID_RESULT`).
- `sound-design stinger <in.wav> -o out.wav --duration <s> --peak-at <s> [--type whoosh|hit|rise|other] --json` (schema `sound-design/stinger@1`): exactly `--duration` long with its loudest 10 ms within 10 ms of `--peak-at`; cut, faded for 5 / 10 ms where cut short (ffmpeg-skill `audio`) and padded with digital silence; re-measured, refused otherwise.
- `sound-design seam-check <file.wav> --period <s> | --seams <t,...> --json` (schema `sound-design/seam-check@1`, read-only, exit 1 on a failing seam): click within ±1 ms, level and octave-band spectrum of 1 s on each side. qc-skill has no loop seam check.
- New `pcm.py`: a stdlib PCM WAV reader (8 / 16 / 24 / 32-bit, float, extensible) and the peak / seam / loop-point measurements. ffmpeg-skill `audio` (`--fade-in`, `--fade-out`) joins the tools used. Doctor lists `loop` and `stinger`; the contract lists the three tools, `policy.shaping` and `provenance.shaping_fields`.
- Tests: `tests/test_shaping.py`, and an integration test in which a loop repeated ten times passes the seam check at all nine seams and a naive loop fails at every seam.

ffmpeg-skill 3.0.0 (ADR-010).

- ffmpeg-skill 3.x declares `contract_version` 2.0; it is read like 1.0, with no warning. Every tool and flag this skill uses is unchanged, and its `cut` calls always pass `--accurate`, so 3.0's new `cut.py` defaults do not reach it. The CI integration job also runs against 3.0.0, pinned by commit.

`SKILL.md` (roadmap step 5, skill side, ADR-009): what an agent reads, at the repository root.

- Task -> command table with an example for every tool, when to use library mode or the generators, the EzAudio confirmation as an instruction (ask the user, then add `--acknowledge-ezaudio-training-data`), the library licence rules, limits, engines, the error table and what is not this skill's job. No CLI, contract or op change.
- `tests/test_unit.py` `SkillMdTests` keep it true: frontmatter name = package name, description with "Use when" under 1024 characters, under 30 KB, an example for every contract tool, every example flag declared in that tool's `argv`, the error table equal to `ERROR_TABLE`, every capability named and `EXPERIMENTAL`, the EzAudio flag still required.
- CI: the read-only OS checkout names `kajisho5/Velisa` (renamed from `kajisho5/AI-video-production-OS`).

Library mode (roadmap step 4, ADR-008): your own BGM / SFX files with a declared licence, no engine.

- `sound-design library-import <file> --library <dir> --licence <SPDX|text> --kind bgm|sfx [--tags] [--mood] [--loopable] [--source] [--author] [--credit] [--name] [--no-loudness] [--overwrite] --json`. A missing, empty or placeholder (`unknown`, `n/a`, ...) licence is refused first, with the new error `LICENCE_REQUIRED` (exit 2, not retryable; message in English and Japanese). The file must be audio only by ffmpeg-skill `probe`; its loudness is measured by `loudness --measure-only`. It is copied into the library (verified by SHA-256, one rename) or registered in place, and its entry is written to `sound-library.json`. Result schema `sound-design/library-import@1`; `audio.library_import` is `EXPERIMENTAL`.
- `sound-library.json` (schema `sound-design/library@1`) at the library root: per file, keyed by its library-relative path in Unicode NFC, the declared `licence`, `kind`, `tags`, `mood`, `loopable`, `source`, `author`, `credit`, plus what the import measured. Read as UTF-8 with an optional BOM, written as UTF-8.
- `sound-design library-index <dir> [--loudness] --json` (read-only, schema `sound-design/library-index@1`): every audio file with its licence (`declared`, or `unknown` for a file without a declared licence), kind, tags, mood and the duration from ffmpeg-skill `probe`, optionally the EBU R128 loudness. It also reports unreadable files, files changed since import, missing files and a summary.
- `sound-design library-pick <dir> -o out.wav --kind bgm|sfx [--tags] [--mood] [--duration] [--file] [--crossfade 0.5] [--overwrite] --json` (schema `sound-design/library-pick@1`):
  - deterministic choice: declared licence, kind, >= 1 matched tag / mood, audio only, fittable; ranked by matches, fit (none < trim < loop), length difference, path, SHA-256; every exclusion listed with its reason;
  - fit through ffmpeg-skill: used whole within max(0.1 s, 1 %) (PCM WAV copied as is, other formats converted by `cut --accurate`); trimmed by `cut --accurate`; looped (files marked `loopable` only) by `join` with a crossfade of at most a quarter of the file, then `cut`; at most 64 copies;
  - the WAV is validated by `probe` and measured by `loudness --measure-only` (never normalised), then placed with one rename;
  - provenance: source file and SHA-256, the licence as declared (`verified: false`), selection rule and query, fit, `engine: null`, `ai_generated: false`.
- Licences are never inferred or cleared: unknown stays `unknown`. Warnings flag a non-commercial term, a no-derivatives term on a fitted file, and a CC BY licence without a credit.
- No engine in library mode: `library.py` imports none, and the tests patch every engine entry point (EzAudio and ACE-Step) to fail. The EzAudio confirmation (ADR-007) is unchanged. Contract (`policy.library_licence`, `policy.library_engine`, the `library-index` and `library-pick` tools, `provenance.library_fields`, `handoff.library`) and doctor (`checks.operations.library`) say so.
- ffmpeg-skill tools used: `probe`, `cut`, `join`, `loudness` (each checked flag by flag against the live contract). ffmpeg-skill is not changed. Gaps found (ADR-008): `loop` refuses audio-only input, and `join --transition fade --duration 0` does not join end to end.
- Tests: `tests/test_library.py` covers licence reporting, fitting maths, deterministic selection, indexing (Japanese, NFD / NFC), import, pick, the provenance, engine isolation, and a cp1252 console. One integration test runs import, index, trim, loop, FLAC conversion and refusal through the real ffmpeg-skill with WAVs written by the test.

EzAudio needs the user's acknowledgement first (ADR-007, the owner's decision: "whenever EzAudio is used, tell me first").

- `generate-sfx` on the `ezaudio` engine refuses to run without `--acknowledge-ezaudio-training-data` (`SfxRequest.acknowledge_ezaudio_training_data`). The refusal comes after the pure argument and output checks and before any engine inspection, ffmpeg-skill call or worker.
- New error `CONFIRMATION_REQUIRED`, exit code 8, not retryable. The message, in English and then Japanese, states the training-data caveat (AudioSet / VGGSound / AudioCaps, YouTube audio not licensed for training) and the exact flag to add. `details` also carries `message_en`, `message_ja`, `flag`, `training_data` and `approval: "CONFIRM"`.
- With the flag, `provenance.confirmation` and the result's `confirmation` record `acknowledged: true`, the flag and `acknowledged_at` (UTC). `engine.confirmation_required` joins the provenance engine fields; BGM records `confirmation: null`.
- `EngineSpec` gains `confirmation_flag`, `confirmation_reason` and `confirmation_reason_ja`; `confirmation_required` is published in `to_dict()`.
- The contract (each generation tool, and `policy.confirmation`) and doctor (`checks.operations.<op>`) expose `confirmation_required` and `confirmation` (flag, reason in English and Japanese, `CONFIRMATION_REQUIRED`, `approval: "CONFIRM"`), so video-production-agent can surface a CONFIRM step before running the op. The SFX `argv` lists the flag as required.
- Tests: refusal before any engine or ffmpeg-skill call, the bilingual message, the CLI exit code, success with the flag, the provenance record, contract and doctor fields, and the integration refusal against the real ffmpeg-skill. Existing SFX tests pass the flag.

SFX engine and GENERATE_SFX (roadmap step 3, ADR-005, ADR-006).

- ADR-005: the SFX engine is EzAudio (`ezaudio`), checked from primary sources:
  - code MIT (haidog-yaqub/EzAudio `LICENSE`), weights MIT (HF `OpenSound/EzAudio`), text encoder `google/flan-t5-xl` Apache-2.0;
  - no statement about the output anywhere; the licences do not restrict it;
  - trained on AudioSet / VGGSound / AudioCaps (YouTube audio, not licensed for training by its owners). This is recorded as `training_data` and `output_warning` in the `EngineSpec`, every result's `warnings` and every provenance.
- Also checked for SFX. Rejected:
  - Stable Audio Open 1.0 / Small: the Stability AI Community License, now read, ends commercial use above USD 1M revenue;
  - AudioLDM / AudioLDM2: CC BY-NC-SA code and weights;
  - AudioGen, Tango / Tango 2, TangoFlux, MMAudio, AudioX and Auffusion: non-commercial;
  - ThinkSound: the README forbids commercial use;
  - SoundCTM: it needs AudioLDM's non-commercial VAE at inference.

  Unverified: HunyuanVideo-Foley (licence text not read), and GenAU, AudioLCM, Make-An-Audio 2 and Kling-Foley (no public model found).
- `sound-design generate-sfx --prompt/--tags --duration <whole s> -o out.wav [--seed] [--device auto|cuda|cpu] --json`. Result schema `sound-design/generate-sfx@1`; `audio.sfx_generate` is `EXPERIMENTAL`; default timeout 600 s.
- EzAudio is called through its Python API as read from haidog-yaqub/EzAudio@86f18268: `EzAudio(model_name="s3_xl", ckpt_path=<checkout>/ckpts/s3/ezaudio_s3_xl.pt, vae_path=<checkout>/ckpts/vae/1m.pt, device)`, then `generate_audio(prompt, length=<s>, guidance_scale=5, guidance_rescale=0.75, ddim_steps=100, eta=1, random_seed=seed, randomize_seed=False)`. The result is written once as 16-bit PCM with soundfile. This runs in the worker only, under `SOUND_DESIGN_EZAUDIO_PYTHON`, from the checkout directory.
- No download is possible:
  - doctor and the worker require the checkpoints and the Flan-T5-XL cache snapshot first;
  - both checkpoint paths are passed;
  - `EzAudio.download_ckpt` is replaced with a refusal;
  - Hugging Face stays offline;
  - netguard blocks every non-loopback socket.
- The duration is whole seconds: EzAudio feeds `length * 50` to `torch.randn` as a size.
- Doctor lists `ezaudio` (`OK` / `MISSING` / `INCOMPATIBLE`; the checkout's API is detected from `api/ezaudio.py`, its commit read from `.git`), and the SFX op as implemented. The CI doctor checks expect both engines.
- `EngineSpec.output_warning` is new. The provenance engine fields gain `training_data` and `output_warning`.
- Tests run a fake EzAudio checkout with the verified API (deterministic audio for seed and length) through a stub ffmpeg-skill and, in the integration job, the real ffmpeg-skill probe. Real-engine determinism, duration accuracy and memory are not measured yet.

GENERATE_BGM (roadmap step 2, ADR-004).

- `sound-design generate-bgm --prompt/--tags --duration -o out.wav [--seed] [--lyrics] [--device] --json`. Result schema `sound-design/generate-bgm@1`; `audio.bgm_generate` is `EXPERIMENTAL`.
- ACE-Step 1.5 is called through its Python API as read from ace-step/ACE-Step-1.5@ca1e85fe: `AceStepHandler.initialize_service(config_path="acestep-v15-turbo", vae_checkpoint="official")`, then `generate_music(dit, None, GenerationParams(...), GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav"))`. This runs in the worker only, under `SOUND_DESIGN_ACESTEP_PYTHON`.
- An omitted seed is drawn by the skill, so the engine always runs with a fixed, recorded seed.
- No download is possible:
  - doctor and the worker check ACE-Step's own component rule; it requires the 1.7B LM on disk, so doctor now reports MISSING without it;
  - `ACESTEP_CHECKPOINTS_DIR` points ACE-Step at the local directory;
  - ACE-Step's Hugging Face / ModelScope download functions are replaced with refusals;
  - the worker blocks every non-loopback socket and name lookup (`engines/netguard.py`).
- The 5 Hz LM is off (`llm_handler=None`, no chain-of-thought) until the Qwen3 base licences are verified; `EngineSpec.lm_status` records this in the contract, doctor and every provenance.
- The output is validated through ffmpeg-skill `probe` (one audio stream, the engine's sample rate, duration within max(0.1 s, 1 %)), then renamed into place from a private run directory.
- The provenance carries the engine, licences, LM status, caption, seed, durations, output SHA-256 and the exact ACE-Step parameters.
- Tests run a fake `acestep` package with the verified API and a deterministic WAV (seed, duration). Real-engine determinism and duration accuracy are not measured yet.

Package skeleton (roadmap step 1).

- ADR-001: engines. ACE-Step 1.5 checked from primary sources, code and weights separately:
  - Code MIT (bundled nano-vllm MIT); weights MIT on the ACE-Step model cards; the bundled text encoder Qwen3-Embedding-0.6B is Apache-2.0.
  - Commercial use of the output is stated by the model card; the training-data description is the vendor's claim (unverified). No credit, no known watermark.
  - It downloads models on first run by itself; the skill prevents that (doctor reports MISSING, the worker runs offline).
  - Also checked: ACE-Step v1 and YuE (Apache-2.0, not now), DiffRhythm (no licence found, unverified), Stable Audio Open (gated licence text, unverified; step 3). Rejected: MusicGen (CC BY-NC weights), AudioLDM2-music (CC BY-NC-SA weights).
  - No SFX engine is chosen; that ADR is step 3.
- ADR-002: skeleton.
  - Commands: `sound-design contract` / `skill`, `doctor` (`--offline`, `--ffmpeg-skill`), `engines`.
  - Engine registry with an `EngineSpec` (licences, output-use terms, training-data claim, duration range, devices, minimum VRAM, isolation, sources) and `OK` / `MISSING` / `INCOMPATIBLE` checks that never import the engine (package metadata, weights files in a checkpoints directory or the Hugging Face cache).
  - Doctor reports the compute device and GPU memory (nvidia-smi, Apple Silicon unified memory via sysctl; ROCm noticed) and ACE-Step's own model tier for it, without importing torch.
  - Worker subprocess in its own process group, under ACE-Step's own interpreter when configured; it validates the request (op, prompt, 10-600 s duration, seed) and then raises `NOT_IMPLEMENTED` (step 2).
  - Typed errors; provenance block with prompt, seed, duration and the engine's licence and output-use fields; ffmpeg-skill adapter by capability detection (`probe`).
- ADR-003: JSON output is ASCII (`ensure_ascii`), human output uses `backslashreplace`, so Windows code pages neither crash the CLI nor its consumers (as audio-production-skill ADR-20 and dubbing-skill ADR-004).
- The three capabilities (`audio.bgm_generate`, `audio.sfx_generate`, `audio.library_import`) are declared with lifecycle `PROPOSED` and are not implemented yet.
