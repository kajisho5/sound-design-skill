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

**Status: 0.1.0, GENERATE_BGM experimental.** `contract`, `doctor`, `engines` and `generate-bgm` work ([ADR-004](docs/decisions.md)). GENERATE_SFX and library mode are declared (`PROPOSED`) and arrive in [roadmap](ECOSYSTEM_ROADMAP.md) steps 3 and 4.

| Op | Capability | Engine | Step |
|---|---|---|---|
| GENERATE_BGM | `audio.bgm_generate` | ACE-Step 1.5 | 2 |
| GENERATE_SFX | `audio.sfx_generate` | to be chosen (ADR in step 3) | 3 |
| Library mode | `audio.library_import` | none (user files with a licence field) | 4 |

```bash
pip install -e .                     # stdlib only, Python >= 3.9
sound-design doctor                  # engine OK / MISSING / INCOMPATIBLE, device and GPU memory, how to install
sound-design engines --json          # status, version, licences and output-use terms per engine
sound-design contract --json         # machine-readable contract (alias: skill --json)
sound-design generate-bgm --prompt "calm piano bed" --tags "ambient,70 bpm" --duration 60 -o bed.wav --seed 42 --json
```

## GENERATE_BGM ([ADR-004](docs/decisions.md))

- **Inputs.** `--prompt` and / or `--tags` (joined into one caption, < 512 characters), `--duration` (ACE-Step: 10-600 s), `--seed` (drawn and recorded when omitted). Optional: `--lyrics <file>` (default instrumental) and `--device auto|cuda|mps|xpu|cpu`.
- **Checks first.** Arguments, the output path (a new `.wav` unless `--overwrite`), the engine (`OK`) and ffmpeg-skill are all checked before anything runs.
- **Engine.** ACE-Step runs in the worker under `SOUND_DESIGN_ACESTEP_PYTHON`, through its Python API (`AceStepHandler.initialize_service` + `generate_music`, read from its source at a fixed commit). It is pointed at the local checkpoints and is DiT only: the 5 Hz language model is off until the Qwen3 base licences are verified.
- **Validation.** ffmpeg-skill `probe` checks the WAV: one audio stream, the engine's sample rate, the requested duration ± max(0.1 s, 1 %). Then one rename puts it in place; nothing is written otherwise.
- **Result.** The JSON carries the provenance: engine, version and model, code / weights licences, output-use statement, LM status, caption, seed, requested and measured duration, output SHA-256, the exact ACE-Step parameters and the ffmpeg-skill version. The WAV goes to audio-production-skill MIX.

Seed determinism and duration accuracy of the real ACE-Step are not measured yet (no install here); the tests use a fake engine with the same API.

## Engine (optional, [ADR-001](docs/decisions.md))

| Engine | Code / weights licence | Output use | Hardware | How the skill reaches it |
|---|---|---|---|---|
| `acestep` ACE-Step 1.5 (BGM) | MIT / MIT (text encoder Qwen3-Embedding-0.6B Apache-2.0) | commercial use stated by the model card; training data is the vendor's claim (unverified) | < 4 GB VRAM for the 2B turbo model; CUDA, MPS, ROCm, XPU, CPU documented | worker under its own environment: `SOUND_DESIGN_ACESTEP_PYTHON`; weights in `SOUND_DESIGN_ACESTEP_CHECKPOINTS` or the Hugging Face cache |

ACE-Step needs Python 3.11-3.12 and pins its own torch, so it lives in its own checkout and `.venv` (`uv sync`); `doctor` prints the exact install line. Left alone, ACE-Step downloads its models on first run, from Hugging Face or ModelScope. The skill never lets it:
- doctor reports `MISSING` until every component ACE-Step checks is on disk, including the 1.7B LM, which `acestep-download` fetches with the main model and this skill never loads;
- in the worker, ACE-Step's download functions are replaced with ones that refuse;
- the worker refuses every non-loopback connection and name lookup, so the Hugging Face offline variables are not the only guard.

`doctor` also reports the compute device: NVIDIA GPUs and their total / free memory (`nvidia-smi`), Apple Silicon unified memory, or CPU only, with ACE-Step's own recommended model tier for that memory. It never imports torch.

## ffmpeg-skill

The skill looks for ffmpeg-skill in this order: `--ffmpeg-skill`, `SOUND_DESIGN_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill`, `../ffmpeg-skill`. Compatibility is checked tool by tool and flag by flag against its contract, not by version number. It needs `probe`.

## JSON and errors

With `--json`, stdout holds exactly one JSON document, ASCII only ([ADR-003](docs/decisions.md)). A failure is `{"ok": false, "error": {"code", "message", "retryable", "details"}}`, and the code fixes the exit code (`contract --json` lists them under `errors`). `TOOL_ERROR`, `ENGINE_TIMEOUT` and `CANCELLED` are retryable. A declared op that is not built yet is `NOT_IMPLEMENTED` (exit 5).

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests need no engine, no weights, no GPU, no FFmpeg and no network. Package lookups are mocked or pointed at a fake install, the checkpoints and the Hugging Face cache are temporary directories, and `nvidia-smi` / `sysctl` are mocked. GENERATE_BGM runs the real worker against a fake `acestep` package with ACE-Step's verified API, which writes a deterministic WAV for a seed and a duration. The unit tests validate it through a stub ffmpeg-skill, the integration tests through the real one. The integration tests need a real ffmpeg-skill checkout (`SOUND_DESIGN_FFMPEG_SKILL_DIR` or `../ffmpeg-skill`) and FFmpeg. The OS registry conformance checks run when `AI_VIDEO_PRODUCTION_OS_DIR` (or `../AI-video-production-OS`) is present.

## Licence

MIT. Engine licences and output-use terms are listed by `sound-design engines --json` and go into the provenance of every generated file.
