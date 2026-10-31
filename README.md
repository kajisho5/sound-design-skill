<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/app-icon-dark.png">
    <img src="brand/app-icon.png" width="128" alt="sound-design-skill app icon">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/logo/sound-design-skill-wordmark-white.svg">
    <img src="brand/logo/sound-design-skill-wordmark-black.svg" height="48" alt="sound-design-skill">
  </picture>
</p>

<p align="center"><b>Music beds and effects, placed in the mix</b></p>


Music beds (BGM) and sound effects for the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS) (department 音 / Audio). Local engines only. They are optional and never downloaded by the skill. Generated files are handed to [audio-production-skill](https://github.com/kajisho5/audio-production-skill) MIX, which places them in the programme; nothing is mixed here. Probing goes through [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill).

**Status: 0.1.0, GENERATE_BGM and GENERATE_SFX experimental.** `contract`, `doctor`, `engines`, `generate-bgm` ([ADR-004](docs/decisions.md)) and `generate-sfx` ([ADR-006](docs/decisions.md); EzAudio needs `--acknowledge-ezaudio-training-data`, [ADR-007](docs/decisions.md)) work. Library mode is declared (`PROPOSED`) and arrives in [roadmap](ECOSYSTEM_ROADMAP.md) step 4.

| Op | Capability | Engine | Step |
|---|---|---|---|
| GENERATE_BGM | `audio.bgm_generate` | ACE-Step 1.5 | 2 |
| GENERATE_SFX | `audio.sfx_generate` | EzAudio ([ADR-005](docs/decisions.md)) | 3 |
| Library mode | `audio.library_import` | none (user files with a licence field) | 4 |

```bash
pip install -e .                     # stdlib only, Python >= 3.9
sound-design doctor                  # engine OK / MISSING / INCOMPATIBLE, device and GPU memory, how to install
sound-design engines --json          # status, version, licences and output-use terms per engine
sound-design contract --json         # machine-readable contract (alias: skill --json)
sound-design generate-bgm --prompt "calm piano bed" --tags "ambient,70 bpm" --duration 60 -o bed.wav --seed 42 --json
sound-design generate-sfx --prompt "a wooden door closes" --tags "indoor" --duration 3 -o door.wav --seed 7 \
  --acknowledge-ezaudio-training-data --json   # required: see "Confirmation before EzAudio" below
```

## GENERATE_BGM ([ADR-004](docs/decisions.md))

- **Inputs.** `--prompt` and / or `--tags` (joined into one caption, < 512 characters), `--duration` (ACE-Step: 10-600 s), `--seed` (drawn and recorded when omitted). Optional: `--lyrics <file>` (default instrumental) and `--device auto|cuda|mps|xpu|cpu`.
- **Checks first.** Arguments, the output path (a new `.wav` unless `--overwrite`), the engine (`OK`) and ffmpeg-skill are all checked before anything runs.
- **Engine.** ACE-Step runs in the worker under `SOUND_DESIGN_ACESTEP_PYTHON`, through its Python API (`AceStepHandler.initialize_service` + `generate_music`, read from its source at a fixed commit). It is pointed at the local checkpoints and is DiT only: the 5 Hz language model is off until the Qwen3 base licences are verified.
- **Validation.** ffmpeg-skill `probe` checks the WAV: one audio stream, the engine's sample rate, the requested duration ± max(0.1 s, 1 %). Then one rename puts it in place; nothing is written otherwise.
- **Result.** The JSON carries the provenance: engine, version and model, code / weights licences, output-use statement, LM status, caption, seed, requested and measured duration, output SHA-256, the exact ACE-Step parameters and the ffmpeg-skill version. The WAV goes to audio-production-skill MIX.

Seed determinism and duration accuracy of the real ACE-Step are not measured yet (no install here); the tests use a fake engine with the same API.

## GENERATE_SFX ([ADR-006](docs/decisions.md))

- **Inputs.** `--prompt` and / or `--tags` (< 512 characters; EzAudio's text encoder reads 100 tokens), `--duration` in whole seconds (EzAudio: 1-10 s), `--seed` (drawn and recorded when omitted), `--device auto|cuda|cpu`.
- **Same frame as BGM.** Every check passes first, then EzAudio runs in the worker, ffmpeg-skill `probe` checks the WAV (one audio stream, 24 kHz, duration ± max(0.1 s, 1 %)), and one rename puts it in place.
- **Engine.** EzAudio runs under `SOUND_DESIGN_EZAUDIO_PYTHON`, through its Python API (`EzAudio(model_name="s3_xl", ckpt_path, vae_path, device)` + `generate_audio`, read from its source at a fixed commit). Both checkpoint paths are passed and its `download_ckpt` is replaced with a refusal. Flan-T5-XL comes from the Hugging Face cache only.
- **Confirmation before EzAudio ([ADR-007](docs/decisions.md)).** By the owner's decision, the user is told about EzAudio's training data before it runs. Without `--acknowledge-ezaudio-training-data`, `generate-sfx` stops with `CONFIRMATION_REQUIRED` (exit 8). This happens before any engine inspection, ffmpeg-skill call or worker. The message, in English and Japanese, gives the caveat and the flag to add. With the flag, `provenance.confirmation` records `acknowledged: true` and `acknowledged_at` (UTC). `contract --json` (the `generate-sfx` tool) and `doctor --json` (`checks.operations.sfx`) carry `confirmation_required: true` with the reason, so an orchestrator such as video-production-agent can ask the user as a CONFIRM step first.
- **Warning in every result.** EzAudio's licences (MIT, Apache-2.0) allow commercial use and say nothing about the output. It was trained on AudioSet / VGGSound / AudioCaps, which are YouTube audio not licensed for training by its owners. Every result's `warnings` and every provenance (`engine.training_data`, `engine.output_warning`) say so ([ADR-005](docs/decisions.md)).

Seed determinism, duration accuracy and memory use of the real EzAudio are not measured yet; the tests use a fake checkout with the same API.

## Engines (optional, [ADR-001](docs/decisions.md), [ADR-005](docs/decisions.md))

| Engine | Code / weights licence | Output use | Hardware | How the skill reaches it |
|---|---|---|---|---|
| `acestep` ACE-Step 1.5 (BGM) | MIT / MIT (text encoder Qwen3-Embedding-0.6B Apache-2.0) | commercial use stated by the model card; training data is the vendor's claim (unverified) | < 4 GB VRAM for the 2B turbo model; CUDA, MPS, ROCm, XPU, CPU documented | worker under its own environment: `SOUND_DESIGN_ACESTEP_PYTHON`; weights in `SOUND_DESIGN_ACESTEP_CHECKPOINTS` or the Hugging Face cache |
| `ezaudio` EzAudio-XL (SFX) | MIT / MIT (text encoder flan-t5-xl Apache-2.0) | no statement on the output; the licences do not restrict it. Trained on YouTube audio (AudioSet, VGGSound, AudioCaps): recorded as a risk in every result | not documented (unverified); CUDA or CPU | worker under its own environment: `SOUND_DESIGN_EZAUDIO_PYTHON`; the checkout and its `ckpts/` in `SOUND_DESIGN_EZAUDIO_DIR`; Flan-T5-XL in the Hugging Face cache |

Rejected for SFX ([ADR-005](docs/decisions.md)):
- Stable Audio Open 1.0 / Small: Stability AI Community License, whose commercial use ends above USD 1M annual revenue.
- AudioLDM / AudioLDM2, AudioGen, Tango / Tango 2, TangoFlux, MMAudio, AudioX and Auffusion: non-commercial weights or code.
- ThinkSound: the README forbids commercial use.
- SoundCTM: it needs AudioLDM's non-commercial VAE at inference.

ACE-Step needs Python 3.11-3.12 and pins its own torch, so it lives in its own checkout and `.venv` (`uv sync`); `doctor` prints the exact install line. Left alone, ACE-Step downloads its models on first run, from Hugging Face or ModelScope. The skill never lets it:
- doctor reports `MISSING` until every component ACE-Step checks is on disk, including the 1.7B LM, which `acestep-download` fetches with the main model and this skill never loads;
- in the worker, ACE-Step's download functions are replaced with ones that refuse;
- the worker refuses every non-loopback connection and name lookup, so the Hugging Face offline variables are not the only guard.

EzAudio is a git checkout with an unpinned `requirements.txt`, so give it its own `.venv` too. Left alone, it downloads a missing checkpoint itself. Doctor reports `MISSING` until the DiT and VAE checkpoints and the Flan-T5-XL snapshot are on disk. In the worker, both checkpoint paths are passed explicitly and `EzAudio.download_ckpt` refuses; the same socket guard applies.

`doctor` also reports the compute device: NVIDIA GPUs and their total / free memory (`nvidia-smi`), Apple Silicon unified memory, or CPU only, with ACE-Step's own recommended model tier for that memory. It never imports torch.

## ffmpeg-skill

The skill looks for ffmpeg-skill in this order: `--ffmpeg-skill`, `SOUND_DESIGN_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill`, `../ffmpeg-skill`. Compatibility is checked tool by tool and flag by flag against its contract, not by version number. It needs `probe`.

## JSON and errors

With `--json`, stdout holds exactly one JSON document, ASCII only ([ADR-003](docs/decisions.md)). A failure is `{"ok": false, "error": {"code", "message", "retryable", "details"}}`, and the code fixes the exit code (`contract --json` lists them under `errors`). `TOOL_ERROR`, `ENGINE_TIMEOUT` and `CANCELLED` are retryable. A declared op that is not built yet is `NOT_IMPLEMENTED` (exit 5). An engine that needs the user's acknowledgement first (EzAudio) is `CONFIRMATION_REQUIRED` (exit 8) until its flag is passed ([ADR-007](docs/decisions.md)).

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests need no engine, no weights, no GPU, no FFmpeg and no network. Package lookups are mocked or pointed at a fake install, the checkpoints and the Hugging Face cache are temporary directories, and `nvidia-smi` / `sysctl` are mocked. GENERATE_BGM runs the real worker against a fake `acestep` package with ACE-Step's verified API, which writes a deterministic WAV for a seed and a duration. GENERATE_SFX does the same against a fake EzAudio checkout (`api/ezaudio.py` with the verified API, its `ckpts/` layout and a Flan-T5-XL cache snapshot). The unit tests validate it through a stub ffmpeg-skill, the integration tests through the real one. The integration tests need a real ffmpeg-skill checkout (`SOUND_DESIGN_FFMPEG_SKILL_DIR` or `../ffmpeg-skill`) and FFmpeg. The OS registry conformance checks run when `AI_VIDEO_PRODUCTION_OS_DIR` (or `../AI-video-production-OS`) is present.

## Licence

MIT. Engine licences and output-use terms are listed by `sound-design engines --json` and go into the provenance of every generated file.
