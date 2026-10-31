# Architecture Decision Records

## ADR-001 Engines: ACE-Step 1.5 for BGM; the SFX engine is decided in step 3

Checked 2026-10-31 against the repositories, model hosts and package indexes listed under Sources. Code and weights are judged separately: a permissive code licence says nothing about the checkpoint, the text encoder it ships with, or the use of what it generates. AI-video-production-OS ADR-012 named ACE-Step 1.5 for this skill ("MIT, commercial use of output stated by the model card"). Its ecosystem roadmap says candidate engines stay unverified until the owning repository's ADR checks the licences of both code and weights. This ADR is that check for BGM. It also records the other music engines someone would propose instead. Sound effects are out of scope here: roadmap step 3 is their ADR, and every SFX candidate is unverified until then.

### Requirements

The skill makes music beds and effects for conference and event video (openings, transitions, background under talks). The engine has to meet these:

- **Licence.** Code *and* weights allow commercial use of the generated audio. Attribution or a watermark is acceptable if it is recorded. Non-commercial (NC) licences, revenue-capped licences, gated licences we could not read, and weights with no stated licence are rejected or left unverified.
- **Local and offline.** It runs on the user's machine with no network at run time. Weights are fetched once by the user, never by the skill.
- **Consumer hardware.** A default model that runs on a single consumer GPU (or Apple Silicon), with a documented CPU fallback.
- **Isolation.** This package stays MIT and stdlib-only. The engine is never imported into it; it runs in a worker process under its own environment.
- **Obligations are data.** The engine's licences, output-use statement, training-data claim, credit and watermark go into an `EngineSpec`, then into the provenance of every file, then into the ProductionReceipt (OS ADR-012).

### ACE-Step 1.5

| Item | Finding (primary source) |
|---|---|
| Code | **MIT**, "Copyright (c) 2026 ACEStep" (ace-step/ACE-Step-1.5 `LICENSE`; `pyproject.toml` `license = {text = "MIT"}`, version `1.5.0`). The bundled `acestep/third_parts/nano-vllm` is MIT ("Copyright (c) 2025 Xingkai Yu") |
| Weights | **MIT** on the Hugging Face model cards checked: `ACE-Step/Ace-Step1.5` (turbo DiT, VAE, 1.7B LM, text encoder), `ACE-Step/acestep-v15-base`, `ACE-Step/acestep-v15-xl-turbo`, `ACE-Step/acestep-5Hz-lm-0.6B` (card metadata `license: mit`) |
| Text encoder | `Qwen3-Embedding-0.6B`, shipped inside `ACE-Step/Ace-Step1.5`, is **Apache-2.0** (HF `Qwen/Qwen3-Embedding-0.6B`). The LMs are fine-tuned from Qwen3-0.6B / 1.7B / 4B; the Qwen3 base licences were not re-checked here: **unverified** (the ACE-Step cards state MIT for the fine-tuned weights) |
| Use of the output | The `Ace-Step1.5` model card: "Commercial-Ready ... You can strictly use the generated music for **commercial purposes**". The README disclaimer asks users to verify originality, disclose AI involvement and get permission when adapting protected styles; it is guidance, not a licence condition |
| Training data | The model card says: licensed tracks, royalty-free / public-domain music, and MIDI-to-audio synthetic data. The dataset is not published, so this is the vendor's claim: **unverified** |
| Credit / watermark | No credit required by the licence. No watermark is documented: **none known** (not tested on output) |
| Hardware | "Runs locally with less than 4GB of VRAM" (card). README tiers: <= 6 GB DiT-only with INT8 + CPU offload, up to >= 24 GB for XL + 4B LM. CUDA recommended; MPS, ROCm, Intel XPU and CPU documented. XL (4B DiT) needs >= 12 GB with offload |
| Duration | 10 s to 600 s (README "Flexible Duration"; `GenerationParams.duration` 10-600) |
| Seed | `GenerationParams.seed` and `use_random_seed=False` for reproducible results (docs/en/INFERENCE.md). Bit-exact determinism across devices is **unverified**; step 2 measures it |
| Install | Python 3.11-3.12, `uv sync` from the git checkout. It pins torch per platform, gradio 6.2.0 and transformers < 4.58. The PyPI project `ace-step` is a different, older release (0.1.0, Apache 2.0), so it is not an install path |
| Downloads | "Models are downloaded automatically on first run" (INSTALL.md), from Hugging Face or ModelScope |

**Finding: ACE-Step downloads on its own.** Left alone, ACE-Step fetches its models on first use. The skill never lets that happen: doctor reports `MISSING` until the user has fetched the weights, the worker forces the Hugging Face libraries offline (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`), and the engine refuses to run before its weights are found. ModelScope downloads are not covered by those variables; the step 2 implementation must pass explicit local checkpoint paths so no download path is reached.

**Finding: the output-use statement is the model card's.** The MIT licence covers the weights. The permission to use generated music commercially is stated on the model card, not in a licence file. Both are recorded in the `EngineSpec` (`weights_license`, `output_use`) and copied into every file's provenance, so a later change to the card can be traced.

### Other music engines checked

| Candidate | Code | Weights | Verdict |
|---|---|---|---|
| ACE-Step v1 (ace-step/ACE-Step, `ACE-Step-v1-3.5B`) | Apache-2.0 (repository LICENSE) | Apache-2.0 (HF card metadata) | **Not now**: superseded by 1.5 from the same team; a permissive fallback if 1.5 regresses |
| YuE (multimodal-art-projection/YuE, `YuE-s1-7B-anneal-en-cot`) | Apache-2.0 (repository LICENSE) | Apache-2.0 (HF card metadata) | **Not now**: a 7B song (lyrics) model; heavier than needed for instrumental beds. Output-use terms and training data not checked: **unverified** |
| DiffRhythm (ASLP-lab/DiffRhythm, `DiffRhythm-1_2`) | `main/LICENSE` returned 404; other file names and branches were not searched | No licence in the HF card metadata | **Unverified**: no licence found where it was looked for |
| MusicGen (facebookresearch/audiocraft) | MIT (`LICENSE`) | **CC BY-NC 4.0** (`LICENSE_weights`; HF `facebook/musicgen-small`, `-medium`) | **Rejected**: non-commercial weights |
| AudioLDM2 music (`cvssp/audioldm2-music`) | not checked | **CC BY-NC-SA 4.0** (HF card metadata) | **Rejected** for BGM: non-commercial weights. Other AudioLDM2 checkpoints belong to the step 3 SFX ADR |
| Stable Audio Open 1.0 / Small (stabilityai) | not checked | "Stability AI Community License" (HF card metadata `license: other`); the licence text is behind a gated download and could not be read | **Unverified**; candidate for the step 3 SFX ADR, not for BGM here |

### Decision

- **Registry.** The engine registry holds one engine, `acestep` (ACE-Step 1.5), the default for GENERATE_BGM. No SFX engine is registered: `DEFAULT_ENGINE["sfx"]` is `null` until the step 3 ADR. (Step 3: [ADR-005](#adr-005-sfx-engine-ezaudio-mit-code-and-weights-the-other-candidates-are-non-commercial-or-revenue-capped) registers `ezaudio` for SFX.)
- **Default model.** The 2B turbo DiT (`acestep-v15-turbo`) with its VAE and text encoder, which fits the documented < 4 GB path. The 5 Hz LM planner is optional; doctor reports which LMs are present and falls back to ACE-Step's own DiT-only mode when none is. XL models are allowed later only with a GPU that meets their documented minimum.
- **EngineSpec.** Code licence, weights licence (with the Apache-2.0 text encoder), output-use statement, training-data claim, credit, watermark, duration range, devices, minimum VRAM, isolation, supported version range (`>=1.5.0 <1.6.0`), install line, sources and the date of this check. `contract --json` and `engines --json` publish it; every file's provenance copies the licence and output-use fields.
- **Isolation.** ACE-Step runs in the worker subprocess under its own interpreter, `SOUND_DESIGN_ACESTEP_PYTHON` (normally `<checkout>/.venv/bin/python`). It is never a pip extra of this package, and this package never imports it, torch or transformers.
- **Availability.** `doctor` reports `OK`, `MISSING` or `INCOMPATIBLE`, with the install line. The package is found by distribution metadata (`ace-step` 1.5.x and a findable `acestep` module), the weights by the required files in `SOUND_DESIGN_ACESTEP_CHECKPOINTS` or a Hugging Face cache snapshot of `ACE-Step/Ace-Step1.5`. An explicit checkpoints directory is never replaced by the cache.

### ecosystem.json

`ecosystem.json` in AI-video-production-OS names ACE-Step 1.5 already. The value below adds the findings; it is to be applied in AI-video-production-OS, which this repository does not edit:

```json
"engines": [
  {"name": "ACE-Step 1.5", "note": "BGM / MIT(コード・重み) / テキストエンコーダ Qwen3-Embedding-0.6B は Apache-2.0 / 生成物の商用利用はモデルカードに明記 / 学習データはベンダー申告(未検証) / VRAM 4GB未満から / 別venv"},
  {"name": "(SFX: step 3 の ADR で選定)", "note": "候補 Stable Audio Open・AudioLDM2 は未検証"}
]
```

### Follow-ups

- Step 2: measure seed determinism and the duration tolerance on a real install; pass local checkpoint paths so neither Hugging Face nor ModelScope is contacted.
- Step 3: the SFX engine ADR (Stable Audio Open, AudioLDM2 and others), with code and weights licences read from their primary sources.
- Re-check the Qwen3 base-model licences of the 5 Hz LMs before an LM is enabled by default.

### Sources

- ACE-Step 1.5: [ace-step/ACE-Step-1.5 (README: commercial use, VRAM tiers, duration, disclaimer)](https://github.com/ace-step/ACE-Step-1.5), [LICENSE (MIT)](https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE), [pyproject.toml (1.5.0, MIT, Python 3.11-3.12, pinned dependencies)](https://github.com/ace-step/ACE-Step-1.5/blob/main/pyproject.toml), [nano-vllm LICENSE (MIT)](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/third_parts/nano-vllm/LICENSE), [docs/en/INSTALL.md (auto-download, devices, environment variables)](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md), [docs/en/INFERENCE.md (GenerationParams: duration, seed)](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md), [HF ACE-Step/Ace-Step1.5 (license: mit; commercial use; training data; files)](https://huggingface.co/ACE-Step/Ace-Step1.5), [HF ACE-Step/acestep-v15-base](https://huggingface.co/ACE-Step/acestep-v15-base), [HF ACE-Step/acestep-v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [HF ACE-Step/acestep-5Hz-lm-0.6B](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B), [HF Qwen/Qwen3-Embedding-0.6B (apache-2.0)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [PyPI ace-step (0.1.0, not this release)](https://pypi.org/project/ace-step/)
- Others: [ace-step/ACE-Step LICENSE (Apache-2.0)](https://github.com/ace-step/ACE-Step/blob/main/LICENSE), [HF ACE-Step/ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B), [multimodal-art-projection/YuE LICENSE (Apache-2.0)](https://github.com/multimodal-art-projection/YuE/blob/main/LICENSE), [HF m-a-p/YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot), [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm), [HF ASLP-lab/DiffRhythm-1_2](https://huggingface.co/ASLP-lab/DiffRhythm-1_2), [facebookresearch/audiocraft LICENSE (MIT)](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE), [LICENSE_weights (CC BY-NC 4.0)](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights), [HF facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small), [HF cvssp/audioldm2-music (cc-by-nc-sa-4.0)](https://huggingface.co/cvssp/audioldm2-music), [HF stabilityai/stable-audio-open-1.0 (gated)](https://huggingface.co/stabilityai/stable-audio-open-1.0), [HF stabilityai/stable-audio-open-small (gated)](https://huggingface.co/stabilityai/stable-audio-open-small)

## ADR-002 Package skeleton: contract, doctor with device and GPU memory, engine registry and worker before any op

### Context

Roadmap step 1 asks for the skeleton, the contract and doctor, and the ACE-Step worker; doctor reports the device and VRAM. It is done when the OS static checks pass (`registry/conformance.py`: `publishes_contract`, `lifecycle_declared`, `dependency_version_ranges`, plus `no_unsafe_shell_out` and `doctor_status`). The pattern is dubbing-skill ADR-002 and voice-enhance-skill ADR-002, which follow transcription-skill.

### Decision

- **Stdlib-only core, Python >= 3.9.** `pip install sound-design-skill` installs no third-party package. ACE-Step is a separate environment (ADR-001), not an extra.
- **CLI.** `sound-design contract --json` (alias `skill --json`), `sound-design doctor --json [--offline] [--ffmpeg-skill DIR]`, `sound-design engines --json [--offline]`. With `--json`, stdout holds exactly one JSON document, on success and on failure.
- **Contract in the OS registry shape.**
  - Registry fields: `skill_id`, `version`, `contract_version`, `provides`, `dependencies` with `version_range`, `not_provided`.
  - Also published: `tools`, `engines`, `device`, `policy`, `provenance`, `execution`, `errors` and `handoff` (BGM goes to audio-production-skill MIX, which owns ミックス; nothing is mixed here).
  - Three capabilities are published with lifecycle `PROPOSED`, their tools as `implemented: false`: `audio.bgm_generate` (GENERATE_BGM, step 2), `audio.sfx_generate` (GENERATE_SFX, step 3) and `audio.library_import` (library mode, step 4). Each moves to `EXPERIMENTAL` in the PR that implements it.
- **Typed errors.** One `SoundDesignError` with a code from a fixed table. The code decides the exit code and the `retryable` flag. Retryable codes: `TOOL_ERROR`, `ENGINE_TIMEOUT`, `CANCELLED`. A declared but unbuilt path raises `NOT_IMPLEMENTED` (exit 5).
- **Engine registry and worker.**
  - Engines are classes with an `EngineSpec`, registered in `engines/__init__.py`. Only registered ids can run.
  - `python -m sound_design_skill.engines.worker <request.json> <result.json>` runs one engine in a child process in its own process group, with a fixed argv, a minimal environment (`SOUND_DESIGN_*`, `HF_HOME` / `HF_HUB_CACHE`, `CUDA_VISIBLE_DEVICES`, no tokens) and the Hugging Face libraries forced offline. Errors come back as structured JSON. `run_engine` kills the whole group on timeout.
  - The worker starts under ACE-Step's own interpreter when `SOUND_DESIGN_ACESTEP_PYTHON` is set; this package's import root goes first on `PYTHONPATH`.
  - `Engine.process` validates the request without the engine (op, non-empty prompt, duration inside the engine's 10-600 s, seed in `[0, 2**32)`), checks availability, then raises `NOT_IMPLEMENTED` with roadmap step 2. No weights are downloaded or loaded in this release.
- **Lookups never import an engine.** The package is found by distribution metadata (`engines/pkgprobe.py`, a standalone script run with `-I` under the engine's interpreter), the weights by files on disk.
- **Device and GPU memory without torch.** `device.py` reads NVIDIA GPUs with `nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version` (list argv, own process group, 10 s timeout), recognises Apple Silicon and reads its unified memory with `sysctl -n hw.memsize`, and only notices ROCm (`rocm-smi` on PATH). Doctor publishes `checks.device` with the accelerator, GPUs, total and free VRAM, and ACE-Step's own recommendation tier for that memory. What torch would pick inside ACE-Step's environment is not checked; ACE-Step detects its GPU itself. The device never fails doctor: a CPU-only machine or a GPU under 4 GB is a warning.
- **`--offline`.** No registered engine needs the network. `--offline` changes nothing today; it is accepted and reported so the agent can pass it unconditionally.
- **ffmpeg-skill by capability detection.** The adapter locates a checkout (`--ffmpeg-skill`, `SOUND_DESIGN_FFMPEG_SKILL_DIR`, `VIDEO_AGENT_FFMPEG_SKILL_DIR`, `~/.claude/skills/ffmpeg-skill`, `./vendor/ffmpeg-skill`, `../ffmpeg-skill`) and checks the one tool this skill needs, `probe` (duration of generated and imported files), never the version number (audio-production-skill ADR-15).
- **Doctor status.**
  - `ok`: ffmpeg-skill is usable and every engine is `OK`.
  - `degraded`: an engine is `MISSING` or `INCOMPATIBLE`. Exit 0, because the skill itself works.
  - `fail`: Python or ffmpeg-skill is not usable. Exit 1.
  - Doctor always lists the engine with licences, output-use terms and install line, the device, and each op with its roadmap step; SFX says its engine is still to be chosen.

## ADR-003 The CLI's JSON is ASCII out, independent of the console code page

### Context

The contract carries Japanese text (the ownership names from `ecosystem.json`, such as ミックス), and later the provenance carries the user's prompt in any language. Windows consoles and pipes encode stdio through a legacy code page (cp1252). audio-production-skill (ADR-20) and dubbing-skill (ADR-004) both hit the crash this causes, first in the CLI (`UnicodeEncodeError`) and then in a consumer reading the output with the platform default (`UnicodeDecodeError`).

### Decision

- Every JSON document is written with `ensure_ascii=True`: non-ASCII text appears as `\u` escapes. Any consumer on any code page parses the same document, and the decoded strings are unchanged. The worker's request and result files are ASCII too.
- Human-readable output (no `--json`) uses `errors="backslashreplace"` on stdout and stderr, so a console that cannot show Japanese prints escapes instead of crashing.
- A unit test runs the CLI with `PYTHONIOENCODING=cp1252` and checks that the JSON is pure ASCII and still decodes to the Japanese text. The CI steps open `doctor.json` explicitly as UTF-8.

## ADR-004 GENERATE_BGM op: ACE-Step's Python API in the worker, DiT only, no download path reachable, validated through ffmpeg-skill

### Context

Roadmap step 2: prompt / tags, duration and seed, with provenance that records the engine, the seed and the licence. It is done when the duration is within tolerance and the seed is deterministic, verified on the engine. ADR-001 left three things for this step: pass local checkpoint paths so that neither Hugging Face nor ModelScope is contacted (the Hugging Face offline variables do not cover ModelScope), measure determinism and the duration tolerance, and keep the 5 Hz LM off until the Qwen3 base licences are checked. The frame follows voice-enhance-skill ADR-003 (`ops.py` on main): checks before any work, the engine in the worker, the result validated through ffmpeg-skill, then one rename.

### ACE-Step 1.5's API, as read from its source

Read only, from ace-step/ACE-Step-1.5 at commit `ca1e85fe9430179831e6bc6be790c332190a3866` (2026-08-29; `pyproject.toml` still `version = "1.5.0"`). Nothing was installed or run, and no weights were downloaded.

| What | Where (at that commit) | What this skill relies on |
|---|---|---|
| Entry points | `docs/en/INFERENCE.md` Quick Start; `acestep/inference.py` `generate_music(dit_handler, llm_handler, params, config, save_dir=None, progress=None) -> GenerationResult` | `AceStepHandler()` from `acestep.handler`, `GenerationParams`, `GenerationConfig` and `generate_music` from `acestep.inference` |
| DiT start-up | `acestep/core/generation/handler/init_service_orchestrator.py` `initialize_service(project_root, config_path, device="auto", ..., prefer_source=None, use_mlx_dit=True, vae_checkpoint=None) -> (status, ok)`; every exception is caught and returned as `(message, False)` | `config_path="acestep-v15-turbo"`, `device` from `auto` / `cuda` / `mps` / `xpu` / `cpu` (`init_service_setup.py` `_resolve_initialize_device`), `vae_checkpoint="official"` (`model_downloader.DEFAULT_VAE_VARIANT`) |
| Checkpoint directory | same file: `ACESTEP_CHECKPOINTS_DIR` wins over `<project_root>/checkpoints` (`model_downloader.get_checkpoints_dir`) | the worker sets `ACESTEP_CHECKPOINTS_DIR` to the directory doctor found (`SOUND_DESIGN_ACESTEP_CHECKPOINTS`, or the Hugging Face cache snapshot) |
| Downloads | `init_service_downloads.py` `_ensure_models_present` calls `ensure_main_model` when `check_main_model_exists` is false, and `ensure_dit_model` when `check_model_exists(config_path)` is false. `model_downloader.py`: `MAIN_MODEL_COMPONENTS = [acestep-v15-turbo, vae, Qwen3-Embedding-0.6B, acestep-5Hz-lm-1.7B]`; a component exists when its directory holds one of eight weight file names (`_contains_model_weights`); `_smart_download` probes `www.google.com:443` with a raw socket (`_can_access_google`) and then downloads from Hugging Face or ModelScope, falling back to the other | see Decision: checked first, download functions replaced, sockets blocked |
| LM | `inference.py` line 655: `use_lm = (thinking or use_cot_*) and llm_handler is not None and llm_handler.llm_initialized and ...` | `llm_handler=None`, and `thinking` and every `use_cot_*` are false |
| Seed | `inference.py` line 613 and `task_utils.py` `prepare_seeds`: with `use_random_seed=False` and `seeds=[s]`, a batch of one runs with seed `s`; each audio's `params["seed"]` reports the seed used | `GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed])`; the returned seed is compared with the request |
| Output | `inference.py` lines 1047-1060: the audio is written to `save_dir/<key>.<format>` and returned as `audios[i]["path"]`, with `sample_rate` (48000) | `audio_format="wav"` (soundfile PCM), one audio, moved to the run directory's `bgm.wav` |
| Text limits | `GenerationParams` docstring: caption < 512 characters, lyrics < 4096, `"[Instrumental]"` for instrumental songs, duration 10-600 s | the same limits are checked before the worker starts |

`docs/en/INFERENCE.md`'s field list is older than the source (it lacks fields such as `enable_normalization`), so only names present in `acestep/inference.py` are passed. Defaults that are left alone: `inference_steps=8` (turbo), `enable_normalization=True` at `normalization_db=-1.0`, `infer_method="ode"`.

### Decision

- **Command.** `sound-design generate-bgm [--prompt <text>] [--tags a,b,...] --duration <s> -o <out.wav> [--seed N] [--lyrics <file>] [--engine acestep] [--device auto|cuda|mps|xpu|cpu] [--timeout 3600] [--overwrite] [--ffmpeg-skill DIR] [--offline] --json`. The prompt and the tags are joined with ", " into ACE-Step's one caption. Result schema `sound-design/generate-bgm@1`: `output` (path, codec, rate, channels, measured duration, SHA-256), `request` (caption, duration, seed, `seed_source` given / drawn), `engine`, `steps`, `provenance`, `handoff` (audio-production-skill MIX), `warnings`. `audio.bgm_generate` moves to `EXPERIMENTAL`.
- **Checks before any work, cheapest first.**
  1. Arguments: caption non-empty and < 512 characters, seed in `[0, 2**32)`, device, lyrics file (UTF-8, < 4096 characters), timeout.
  2. Output: a `.wav`, not a directory, not existing unless `--overwrite`, not the lyrics file, in an existing directory.
  3. Engine: `OK`, else `ENGINE_UNAVAILABLE` / `MODEL_NOT_FOUND` with the install line; then its duration window.
  4. ffmpeg-skill: located and declares `probe`.
  Nothing starts until all pass.
- **The seed is always fixed.** An omitted seed is drawn in the caller (`secrets.randbelow(2**32)`), so ACE-Step never picks one itself and the provenance holds the seed that ran. A result whose seed differs from the request is `INVALID_RESULT`.
- **No download path is reachable.** Four layers, all in the worker:
  1. Doctor and the engine check ACE-Step's own rule first: every `MAIN_MODEL_COMPONENTS` directory, the 1.7B LM included, must hold a weights file by the `_contains_model_weights` names. If one is missing, the engine is `MISSING` ("ACE-Step would download the main model") and nothing is imported. This tightens ADR-001's file list: ACE-Step demands the 1.7B LM on disk even though this skill never loads it. `acestep-download` fetches it with the main repository.
  2. Inside the worker, ACE-Step's own `check_main_model_exists` / `check_model_exists` run on that directory before the handler is imported. `ACESTEP_CHECKPOINTS_DIR` points ACE-Step at it.
  3. `_smart_download`, `_download_from_huggingface_internal` and `_download_from_modelscope_internal` are replaced with functions that refuse and record the attempt. `_can_access_google` returns false without a socket. Any attempt ends as `MODEL_NOT_FOUND`, even if ACE-Step swallowed the error.
  4. `engines/netguard.py`: before any engine import, the worker makes every socket `connect` / `connect_ex`, `getaddrinfo` and `gethostbyname` to a non-loopback address fail with `NetworkBlocked`, a `ConnectionRefusedError`. This covers ModelScope, Hugging Face and anything else, whichever library tries. `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` stay set.
- **The 5 Hz LM is off.** `llm_handler=None`, `thinking=False`, `use_cot_metas` / `use_cot_caption` / `use_cot_language` false. The LMs are fine-tuned from Qwen3-0.6B / 1.7B / 4B, whose base licences are still unverified (ADR-001 follow-up). There is no flag to turn the LM on in this release. `EngineSpec.lm_status` says so and is published in the contract, doctor (`lm_note`) and every provenance (`engine.lm_status`, `parameters.lm`).
- **Validation and placement.** The worker writes `bgm.wav` in a private `.sound-design-*` directory inside the output's own directory. ffmpeg-skill `probe` must read it as exactly one audio stream and no picture, at the engine's sample rate, lasting the requested duration ± max(0.1 s, 1 %). Anything else is `INVALID_RESULT`, and nothing is written to the output path. Then one rename puts it in place; the run directory is always removed. The WAV is ACE-Step's own and is never re-encoded.
- **Provenance.** It records:
  - the engine's licences and output-use statement;
  - `lm_status`, the model, the ACE-Step version and the API commit;
  - the caption text and SHA-256, the seed and its source;
  - the requested and measured duration and the tolerance;
  - the output SHA-256;
  - the tags and the lyrics SHA-256 (not the lyrics);
  - the requested and resolved device;
  - the exact `GenerationParams` / `GenerationConfig` values passed;
  - the ffmpeg-skill version.
- **Importing ACE-Step.** Only `AceStepEngine.process`, which runs in the worker under `SOUND_DESIGN_ACESTEP_PYTHON`, loads `acestep`. It does so with `importlib.import_module`, and only after every check. The calling process never imports it: tests assert `acestep` is absent from `sys.modules` after a generation, and the step 1 scan for `import acestep` / `import torch` in the sources still passes.
- **Tests: a fake engine, not ACE-Step.** `tests/helpers.py` writes a stdlib-only `acestep` package with the names, signatures and dataclass fields above. A field name that does not exist upstream fails with `TypeError`. It records every call and writes a stereo 48 kHz WAV that depends only on (seed, duration). Modes simulate a wrong length, an engine failure, an initialisation failure, a wrong seed, a download attempt and a hang. The unit tests run the real worker against it, with a stub ffmpeg-skill that reads WAV headers. The integration tests run the CLI against the real ffmpeg-skill `probe`.

### What is verified, and what is not

- **Verified here:** the adapter calls the API exactly as read; with the LM off, no LM path is taken; a fixed seed reaches the engine and comes back; the duration check, placement and clean-up work; no download function, Google probe or non-loopback socket is reachable from the worker. Same seed and duration give byte-identical files *with the deterministic fake engine*.
- **Not verified:** bit-exact determinism and the duration accuracy of the real ACE-Step 1.5 (no install, no weights, no GPU here). `EngineSpec.deterministic` stays `false` and the tool says `deterministic: false`. The 0.1 s / 1 % tolerance is provisional (the DiT works in 25 Hz latent frames, 0.04 s) until it is measured. Speed on CPU is unverified.

### Follow-ups

- On a machine with ACE-Step 1.5 and its weights: generate the same (caption, duration, seed) twice per device type and compare SHA-256, and record the measured duration error. Then set `deterministic` and the tolerance from the measurement. This is roadmap step 2's "verify on the engine".
- Verify the Qwen3-0.6B / 1.7B / 4B base-model licences. Only then add an opt-in LM stage (it needs `LLMHandler.initialize`, which this ADR did not read).
- Step 5 (SKILL.md, CI): an engines job that runs the real ACE-Step if a runner with the weights is available.

## ADR-005 SFX engine: EzAudio (MIT code and weights); the other candidates are non-commercial or revenue-capped

### Context

Roadmap step 3: choose the sound-effects engine. ADR-001 named Stable Audio Open and AudioLDM2 as candidates and left every SFX candidate unverified. This ADR checks them, and the other open text-to-audio models someone would propose, against ADR-001's requirements: code *and* weights allow commercial use of the generated audio; it runs locally and offline; consumer hardware; isolated in a worker; obligations recorded as data. Non-commercial (NC) licences, revenue-capped licences, licences or components we could not read, and weights with no stated licence are not chosen.

Checked 2026-10-31 from primary sources: the licence files in the code repositories, the Hugging Face model cards and their `license` metadata (read through the Hugging Face API, `https://huggingface.co/api/models/<id>`), and the licence files in the model repositories. Code and weights are judged separately, and so is every component loaded at inference (VAE, vocoder, text encoder).

### Candidates

| Candidate | Code licence (source) | Weights licence (source) | Output use / other components | Verdict |
|---|---|---|---|---|
| **EzAudio** (`OpenSound/EzAudio`, `s3_xl`) | **MIT**, "Copyright (c) 2025 Jiarui Hai" (haidog-yaqub/EzAudio `LICENSE`) | **MIT** (HF card metadata `license: mit`; DiT checkpoints and the waveform VAE are in the same repository) | Text encoder `google/flan-t5-xl` **Apache-2.0** (HF card). No statement on the output anywhere; MIT does not restrict it. Training data: AudioSet, VGGSound, AudioCaps (paper), i.e. YouTube clips | **Chosen** (see Findings) |
| Stable Audio Open 1.0 (`stabilityai/stable-audio-open-1.0`) | MIT (Stability-AI/stable-audio-tools `LICENSE`) | **Stability AI Community License** (`LICENSE.md`, "Last Updated: July 5, 2024"). The model page is gated, but the file is served without login, so it was read this time (ADR-001 could not) | §III: commercial use requires registration, and every licence terminates once the licensee and its affiliates exceed **USD 1,000,000** annual revenue; §IV(a) "Powered by Stability AI"; §IV(b) the output may not be used to create or improve a foundation model; revocable. §IV(c)(iii): the user owns the output | **Rejected**: revenue-capped, revocable commercial licence (ADR-001 rejects revenue caps) |
| Stable Audio Open Small (`stabilityai/stable-audio-open-small`) | MIT (stable-audio-tools) | Stability AI Community License (`LICENSE`, the same July 5, 2024 text, USD 1,000,000 cap) | as above | **Rejected**: revenue-capped |
| AudioLDM2 (`cvssp/audioldm2`, `audioldm2-large`, `audioldm2-music`) | **CC BY-NC-SA 4.0** (haoheliu/AudioLDM2 `LICENSE`) | **CC BY-NC-SA 4.0** (HF card metadata, all three) | n/a | **Rejected**: non-commercial code and weights |
| AudioLDM 1 (`cvssp/audioldm-s-full-v2`, `audioldm-l-full`) | not checked | **CC BY-NC-SA 4.0** (HF card metadata) | n/a | **Rejected**: non-commercial weights |
| AudioGen (`facebook/audiogen-medium`) | MIT (audiocraft `LICENSE`, ADR-001) | **CC BY-NC 4.0** (HF card metadata) | n/a | **Rejected**: non-commercial weights |
| Tango / Tango 2 (`declare-lab/tango`, `tango2`) | not checked | **CC BY-NC-SA 4.0** (HF card metadata) | n/a | **Rejected**: non-commercial weights |
| TangoFlux (`declare-lab/TangoFlux`) | declare-lab/TangoFlux `LICENSE.md`: "created for **non-commercial, research-only** purposes", subject to the Stability AI Community License | no `license` in the HF card metadata | n/a | **Rejected**: non-commercial |
| MMAudio (`hkchengrex/MMAudio`) | MIT, "Copyright (c) 2024 Sony Research Inc." (hkchengrex/MMAudio `LICENSE`) | **CC BY-NC 4.0** (HF card metadata) | n/a | **Rejected**: non-commercial weights |
| ThinkSound (`FunAudioLLM/ThinkSound`) | README "License": Apache 2.0, but "The code, models, and dataset are **for research and educational purposes only**. **Commercial use is NOT permitted.**" (FunAudioLLM/ThinkSound `README.md`, master); no `LICENSE` file at `main` or `master` | HF card metadata says `apache-2.0`, contradicted by the README | Ships a VAE fine-tuned from Stable Audio Open under the Stability AI Community License ("Commercial use and redistribution require prior permission from Stability AI"). Video-to-audio first | **Rejected**: commercial use excluded by the authors |
| AudioX (`HKUSTAudio/AudioX`) | not checked | **CC BY-NC 4.0** (HF card metadata) | n/a | **Rejected**: non-commercial weights |
| Auffusion (`auffusion/auffusion`, `auffusion-full-no-adapter`) | not checked | **CC BY-NC-SA 4.0** (HF card metadata) | n/a | **Rejected**: non-commercial weights |
| SoundCTM (`Sony/soundctm`) | MIT, "Copyright (c) 2024 Sony Research Inc." (sony/soundctm `LICENSE`) | SoundCTM weights **CC BY 4.0** (HF card) | Inference also loads `ckpt/audioldm-s-full.ckpt` "for VAE's decoder+Vocoder" (sony/soundctm `README.md`), which upstream is AudioLDM-s-full under **CC BY-NC-SA 4.0**; the student is distilled from TANGO (CC BY-NC-SA 4.0) | **Rejected**: a non-commercial component is required at inference |
| HunyuanVideo-Foley (`tencent/HunyuanVideo-Foley`) | not checked | "Tencent Hunyuan Community License" (HF card metadata `license: other`); the licence text was not read | Video-to-audio, not text-to-SFX | **Unverified**, not chosen |
| GenAU, AudioLCM, Make-An-Audio 2, Kling-Foley | not checked | no public model found at the Hugging Face ids tried (`snap-research/GenAU`, `ByteDance/AudioLCM`, `AIGC-Audio/Make-An-Audio-2`, `kuaishou/Kling-Foley`) | n/a | **Unverified**, not chosen |

### Findings on EzAudio

- **Licences.** Code MIT; DiT and VAE weights MIT on the model card; text encoder Apache-2.0. All three allow commercial use. No credit is required beyond keeping the licence notices with redistributed copies, and this skill redistributes nothing. No watermark is documented: **none known** (not tested on output).
- **Output use.** Neither the licences, the model card nor the README say anything about the generated audio. None of the licences restricts it. ACE-Step's card states commercial use outright (ADR-001); EzAudio's does not. The `EngineSpec.output_use` says exactly this.
- **Training data: a recorded risk.** The paper (arXiv 2409.10819, §I-II) trains the DiT on AudioSet, VGGSound and AudioCaps and the waveform VAE on AudioSet. These datasets are lists of YouTube clips; the audio belongs to the uploaders, who did not license it for training. No licence can grant rights its licensor does not hold. This is not verified either way. It is published as `EngineSpec.training_data` and as `EngineSpec.output_warning`, which every GENERATE_SFX result carries in `warnings` and every provenance in `engine.output_warning`. The ProductionReceipt therefore shows it.
- **Authorship.** The paper notes the work was done during an internship at Tencent AI Lab. The weights are published under MIT by the `OpenSound` organisation; that is the licence this ADR relies on.
- **Hardware.** Not documented. EzAudio-XL is a 28-layer, 1152-wide DiT plus the Flan-T5-XL encoder, loaded in fp32 by `api/ezaudio.py`. `min_vram_gb` stays `null` (**unverified**). It runs on `cuda` or `cpu` (the API passes `device` to torch); MPS is not claimed.
- **Install.** A git checkout with `requirements.txt` (unpinned torch, diffusers, transformers, librosa and others). There is no distribution and no version number. Weights from `OpenSound/EzAudio` go into the checkout's `ckpts/`; Flan-T5-XL goes into the Hugging Face cache.
- **Downloads.** `EzAudio.__init__` downloads a missing checkpoint itself (`download_ckpt`, urllib's `urlretrieve` from huggingface.co). `T5EncoderModel.from_pretrained("google/flan-t5-xl")` would fetch the encoder when online. ADR-006 makes neither reachable.

### Decision

- **Registry.** `ezaudio` is registered for `sfx` and is `DEFAULT_ENGINE["sfx"]`. ACE-Step stays BGM only.
- **EngineSpec.** It records:
  - the code and weights licences, with the Apache-2.0 text encoder;
  - the output-use finding (no statement, no restriction);
  - the training-data risk and the `output_warning`;
  - credit (none) and watermark (none known);
  - duration 1-10 s and devices `cuda` / `cpu`, with `min_vram_gb` unknown;
  - the isolation, install line and sources.

  `version_range` is a capability rule, because EzAudio has no version: a checkout whose `api/ezaudio.py` still has the API ADR-006 read. A checkout without it is `INCOMPATIBLE`.
- **Availability.** Doctor checks every file and never imports anything:
  - `OK` needs all of these: `SOUND_DESIGN_EZAUDIO_DIR` points at a checkout (`api/ezaudio.py`, `src/inference.py`, `ckpts/ezaudio-xl.yml`, `ckpts/vae/config.json`); that API is present; EzAudio's modules (`torch`, `transformers`, `diffusers`, `soundfile`, `yaml`, `librosa`, `einops`) can be found under `SOUND_DESIGN_EZAUDIO_PYTHON` (else this Python); `ckpts/s3/ezaudio_s3_xl.pt` and `ckpts/vae/1m.pt` exist; and a Hugging Face cache snapshot of `google/flan-t5-xl` holds `config.json`, `spiece.model` and both safetensors shards with their index.
  - Otherwise the engine is `MISSING` with the install line.
- **Provenance.** `training_data` and `output_warning` join the engine fields copied into every provenance (BGM too, where `output_warning` is `null`).
- **Isolation.** As ACE-Step: the worker runs under EzAudio's own interpreter. This package never imports EzAudio, torch, diffusers or transformers.

### ecosystem.json

To be applied in AI-video-production-OS, which this repository does not edit (replacing the SFX placeholder of ADR-001):

```json
{"name": "EzAudio", "note": "SFX / MIT(コード・重み) / テキストエンコーダ flan-t5-xl は Apache-2.0 / 生成物の利用に関する記述なし(ライセンス上の制限なし) / 学習データは AudioSet・VGGSound・AudioCaps (YouTube 音源、権利者の許諾なし: リスクとして記録) / 1-10 秒 / 別venv"}
```

### Follow-ups

- Re-check if EzAudio's card or README ever adds output terms, or if `OpenSound/EzAudio` changes licence.
- An engine trained on licensed or public-domain effects only would remove the training-data risk. None of the candidates above documents that and has a commercial licence.
- Read the Tencent Hunyuan Community License if a video-to-audio (Foley) op is ever wanted.

### Sources

- EzAudio: [haidog-yaqub/EzAudio](https://github.com/haidog-yaqub/EzAudio), [LICENSE (MIT)](https://github.com/haidog-yaqub/EzAudio/blob/main/LICENSE), [HF OpenSound/EzAudio (license: mit; files)](https://huggingface.co/OpenSound/EzAudio), [HF google/flan-t5-xl (apache-2.0)](https://huggingface.co/google/flan-t5-xl), [arXiv 2409.10819 (training data)](https://arxiv.org/abs/2409.10819)
- Stable Audio Open: [HF stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0), [LICENSE.md (Stability AI Community License)](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md), [HF stabilityai/stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small), [LICENSE](https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE), [stable-audio-tools LICENSE (MIT)](https://github.com/Stability-AI/stable-audio-tools/blob/main/LICENSE)
- AudioLDM / AudioLDM2: [haoheliu/AudioLDM2 LICENSE (CC BY-NC-SA 4.0)](https://github.com/haoheliu/AudioLDM2/blob/main/LICENSE), [HF cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2), [HF cvssp/audioldm2-large](https://huggingface.co/cvssp/audioldm2-large), [HF cvssp/audioldm2-music](https://huggingface.co/cvssp/audioldm2-music), [HF cvssp/audioldm-s-full-v2](https://huggingface.co/cvssp/audioldm-s-full-v2), [HF cvssp/audioldm-l-full](https://huggingface.co/cvssp/audioldm-l-full)
- Others: [HF facebook/audiogen-medium](https://huggingface.co/facebook/audiogen-medium), [HF declare-lab/tango](https://huggingface.co/declare-lab/tango), [HF declare-lab/tango2](https://huggingface.co/declare-lab/tango2), [declare-lab/TangoFlux LICENSE.md](https://github.com/declare-lab/TangoFlux/blob/main/LICENSE.md), [hkchengrex/MMAudio LICENSE (MIT)](https://github.com/hkchengrex/MMAudio/blob/main/LICENSE), [HF hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio), [FunAudioLLM/ThinkSound README (License)](https://github.com/FunAudioLLM/ThinkSound/blob/master/README.md), [HF FunAudioLLM/ThinkSound](https://huggingface.co/FunAudioLLM/ThinkSound), [HF HKUSTAudio/AudioX](https://huggingface.co/HKUSTAudio/AudioX), [HF auffusion/auffusion](https://huggingface.co/auffusion/auffusion), [sony/soundctm LICENSE (MIT) and README](https://github.com/sony/soundctm), [HF Sony/soundctm (cc-by-4.0)](https://huggingface.co/Sony/soundctm), [HF tencent/HunyuanVideo-Foley](https://huggingface.co/tencent/HunyuanVideo-Foley)

## ADR-006 GENERATE_SFX op: EzAudio's Python API in the worker, whole seconds, no download path reachable, validated through ffmpeg-skill

### Context

Roadmap step 3 also asks for the op: done when there are an ADR and tests. The frame is ADR-004's: checks before any work, the engine in the worker, the result validated through ffmpeg-skill, then one rename.

### EzAudio's API, as read from its source

Read only, from haidog-yaqub/EzAudio at commit `86f18268117dde73bb274401c610e639e4a8fae1` (2025-12-16), and from the files of `OpenSound/EzAudio` (`ckpts/ezaudio-xl.yml`, `ckpts/vae/config.json`). Nothing was installed or run, and no weights were downloaded.

| What | Where (at that commit) | What this skill relies on |
|---|---|---|
| Entry point | `api/ezaudio.py` `class EzAudio`: `__init__(self, model_name, ckpt_path=None, vae_path=None, device='cuda')`; README "Usage" | `EzAudio(model_name="s3_xl", ckpt_path=<checkout>/ckpts/s3/ezaudio_s3_xl.pt, vae_path=<checkout>/ckpts/vae/1m.pt, device=...)` |
| Model table | `api/ezaudio.py` `configs`: `s3_xl` → `ckpts/s3/ezaudio_s3_xl.pt`, config `ckpts/ezaudio-xl.yml`; `vae` → `ckpts/vae/1m.pt` | the same paths, checked by doctor |
| Downloads | `__init__` calls `download_ckpt(configs[...])` only when `ckpt_path` / `vae_path` is `None`; `download_ckpt` fetches the URL with urllib's `urlretrieve` when the local file is missing | both paths passed; `download_ckpt` replaced in the worker |
| Working directory | `load_models` opens `configs[model_name]['config']`, a path relative to the working directory, with `src/utils/utils.py` `load_yaml_with_includes`; `src/modules/stable_vae/__init__.py` `load_vae` reads `config.json` beside the VAE checkpoint | the worker changes into the checkout and puts it first on `sys.path` (it imports `src.*`) |
| Text encoder | `T5Tokenizer` / `T5EncoderModel.from_pretrained(params['text_encoder']['model'])`, `google/flan-t5-xl` in `ezaudio-xl.yml`; `src/inference.py` tokenizes with `max_length` 100 and `truncation=True` | loaded from the Hugging Face cache only (offline variables, socket guard); a prompt of more than 100 words gets a warning |
| Generation | `generate_audio(self, text, length=10, guidance_scale=5, guidance_rescale=0.75, ddim_steps=100, eta=1, random_seed=None, randomize_seed=False) -> (sr, numpy array)`; `length * latent_sr` latent frames | `length` in whole seconds; the four defaults passed explicitly; `random_seed=seed`, `randomize_seed=False` |
| Length | `src/inference.py` `torch.randn((1, codec_dim, audio_frames), ...)`: the frame count must be an integer; `latent_sr: 50`, `sr: 24000` (`ezaudio-xl.yml`); the DiT is configured for 500 frames (`img_size: 500`, 10 s) | durations 1-10 s, whole seconds; mono 24 kHz expected |
| Seed | `src/inference.py`: `torch.Generator(device=device).manual_seed(random_seed)` for the initial noise and every DDIM step (`eta=1`) | the seed is always given and recorded |
| Saving | README "Usage": `sf.write(f'{prompt}.wav', audio, sr)` (soundfile) | `soundfile.write(<run dir>/sfx.wav, audio, sr, subtype="PCM_16")` |

### Decision

- **Command.** `sound-design generate-sfx [--prompt <text>] [--tags a,b,...] --duration <s> -o <out.wav> [--seed N] [--engine ezaudio] [--device auto|cuda|cpu] [--timeout 600] [--overwrite] [--ffmpeg-skill DIR] [--offline] --json`.
  - The prompt and the tags are joined as for BGM.
  - Result schema `sound-design/generate-sfx@1`: `output`, `request`, `engine` (with `training_data`), `steps`, `provenance`, `handoff` (audio-production-skill MIX) and `warnings` (the engine's `output_warning`, plus the truncation warning).
  - `audio.sfx_generate` moves to `EXPERIMENTAL`.
- **Checks before any work, cheapest first** (shared with GENERATE_BGM in `ops._check_common`).
  1. Arguments: prompt and tags non-empty, < 512 characters; seed in `[0, 2**32)`; device `auto` / `cuda` / `cpu`; timeout > 0.
  2. Output: a new `.wav` in an existing directory, unless `--overwrite`.
  3. Engine: `OK`, an SFX engine, and the duration inside its window as whole seconds (`EzAudioEngine.validate`).
  4. ffmpeg-skill: located and declares `probe`.

  Nothing starts until all pass.
- **The seed is always fixed.** An omitted seed is drawn in the caller. A result whose seed differs from the request is `INVALID_RESULT`.
- **No download path is reachable.**
  1. Doctor and the engine refuse before any import unless every checkpoint file and the Flan-T5-XL snapshot are on disk.
  2. Both checkpoint paths are passed, so `download_ckpt` is not called.
  3. `EzAudio.download_ckpt` is replaced with a refusal that records the attempt (`MODEL_NOT_FOUND`).
  4. The Hugging Face offline variables stay set, so `from_pretrained` reads the cache only.
  5. `engines/netguard.py` blocks every non-loopback socket and name lookup before any engine import. A network attempt while loading ends as `ENGINE_FAILED` with `NetworkBlocked`.
- **Device.** `auto` resolves to `cuda` when torch (imported in the worker with `importlib`) reports one, else `cpu`. The resolved device goes into the provenance.
- **Validation and placement.** The worker writes `sfx.wav` in a private `.sound-design-*` directory inside the output's own directory. ffmpeg-skill `probe` must find:
  - exactly one audio stream and no picture;
  - the engine's sample rate (24 kHz; any other rate from EzAudio is `INVALID_RESULT` in the worker already);
  - the requested duration ± max(0.1 s, 1 %).

  Then one rename puts the file in place. The run directory is always removed. The WAV is 16-bit PCM written once, never re-encoded.
- **Provenance.** As GENERATE_BGM, plus:
  - `engine.training_data` and `engine.output_warning`;
  - the exact `generate_audio` arguments;
  - the text encoder;
  - the EzAudio API commit;
  - the checkout's own commit as the engine version, read from `.git` without running git.
- **Tests: a fake checkout, not EzAudio.** `tests/helpers.py` writes a stdlib-only `api/ezaudio.py` with the class, `configs`, method names, signatures and defaults above, so a keyword that does not exist upstream fails with `TypeError`. Like upstream, it:
  - opens the model config relative to the working directory;
  - reads the VAE `config.json`;
  - refuses a non-integer frame count;
  - would call `urlretrieve` from `download_ckpt`.

  It also writes stand-ins for `soundfile` and `torch.cuda.is_available`, plus findable stubs for EzAudio's other modules and the Flan-T5-XL snapshot. Its audio depends only on (seed, length). Modes simulate a wrong length, a wrong sample rate, an engine failure, a load failure, a forced download, a network attempt while loading, and a hang. The unit tests run the real worker against it through a stub ffmpeg-skill; the integration tests run the CLI against the real ffmpeg-skill `probe`.

### What is verified, and what is not

- **Verified here:**
  - the adapter calls the API exactly as read;
  - the checkpoint paths are explicit and the working directory is the checkout;
  - a fixed seed reaches the engine and comes back;
  - the duration check, placement and clean-up work;
  - no download function or non-loopback socket is reachable from the worker.

  Same seed and length give byte-identical files *with the deterministic fake engine*.
- **Not verified:**
  - bit-exact determinism, duration accuracy, speed and memory use of the real EzAudio (no install, no weights, no GPU here); `EngineSpec.deterministic` stays `false`;
  - whether `torch.load` of the `.pt` checkpoints works with torch >= 2.6, whose `weights_only=True` default may refuse them (EzAudio's own code, unpinned torch);
  - durations under 10 s on the real model, which was trained on 10 s (500-frame) clips.

### Follow-ups

- On a machine with EzAudio and its weights: generate the same (prompt, length, seed) twice per device and compare SHA-256; record the measured duration, the peak GPU memory and the time. Then set `deterministic`, `min_vram_gb` and the tolerance from those measurements.
- Step 5 (SKILL.md, CI): an engines job that runs the real EzAudio if a runner with the weights is available.

## ADR-007 EzAudio runs only after the user is told: `--acknowledge-ezaudio-training-data`, `CONFIRMATION_REQUIRED`, recorded in the provenance

### Context

ADR-005 chose EzAudio for SFX. Its code and weights are MIT, but it was trained on AudioSet, VGGSound and AudioCaps: YouTube clips whose audio was not licensed for training. ADR-005 and ADR-006 record this as `engine.training_data` and `engine.output_warning` in every result and provenance. A warning arrives only after the effect is made, though, and an orchestrator can pass it on unread.

The owner decided: "whenever EzAudio is used, tell me first." So the user must hear the caveat before EzAudio runs, not after.

### Decision

- **The engine declares it.** `EngineSpec` gains `confirmation_flag`, `confirmation_reason` and `confirmation_reason_ja`. `confirmation_required` is true when a flag is set, and `to_dict()` publishes it.
  - EzAudio sets `confirmation_flag = "--acknowledge-ezaudio-training-data"`.
  - ACE-Step sets none.
  - The check reads the spec, so a future engine opts in the same way.
- **The flag.** `generate-sfx --acknowledge-ezaudio-training-data` (`SfxRequest.acknowledge_ezaudio_training_data`). It follows the CLI's existing style (`--overwrite`, `--offline`): a long kebab-case `store_true` flag. It names the engine and the subject, so passing it states what was acknowledged. Only the boolean `True` counts in the library API.
- **Refusal before any work.** Without the flag, GENERATE_SFX on EzAudio ends with the new error `CONFIRMATION_REQUIRED`. The order of checks becomes:
  1. arguments (pure checks);
  2. output path (pure checks);
  3. **confirmation** (`ops.require_confirmation`, new);
  4. engine inspection;
  5. ffmpeg-skill.

  Steps 1 and 2 stay first, so the user is never asked about a call that would fail anyway. The confirmation comes before the engine is inspected, ffmpeg-skill is located, the worker starts or anything is written, so a missing engine or ffmpeg-skill is reported only after the user has agreed.
- **The error.** `CONFIRMATION_REQUIRED` has exit code 8 (its own code, so a caller can branch without parsing JSON) and is not retryable: the same call fails again until the flag is added.
  - `message` is the English text, a newline, then the Japanese text. Each states the training-data caveat and the exact flag to add.
  - `details` carries `engine`, `op`, `flag`, `message_en`, `message_ja`, `training_data`, `approval: "CONFIRM"` and `adr`.
- **Provenance.** With the flag, `provenance.confirmation` (and the result's `confirmation`) is `{"required": true, "engine": "ezaudio", "flag", "acknowledged": true, "acknowledged_at": "<UTC, ISO 8601>", "reason", "adr"}`.
  - `acknowledged_at` is taken when the check passes, before the engine runs.
  - `engine.confirmation_required` joins the provenance engine fields.
  - For BGM `provenance.confirmation` is `null`.
- **Contract and doctor say so up front.** An orchestrator (video-production-agent) learns the requirement without running the op.
  - Every generation tool in `contract --json` and every op in `doctor --json` (`checks.operations.<op>`) gains `confirmation_required` and `confirmation`. For SFX: `{"engine", "flag", "reason", "reason_ja", "error_code": "CONFIRMATION_REQUIRED", "approval": "CONFIRM", "adr"}`.
  - The SFX `argv` lists the flag without brackets, because it is required.
  - `policy.confirmation` describes the protocol: surface a CONFIRM step, tell the user `confirmation.reason`, and pass the flag only after the user agrees.
  - `engines.registered.<id>.confirmation_required` mirrors the spec.
  - Human `doctor` output prints a "confirm first" line under the op.
- **Tests.**
  - Unit: without the flag (and with a non-boolean value), the refusal comes before `select_engine`, `EzAudioEngine.inspect`, `usable_skill`, `FfmpegSkill.locate` and `run_engine` (all patched to fail if reached). The fake engine logs no call, the stub ffmpeg-skill probes nothing, and no file or run directory is left.
  - Unit: the message text in both languages; the CLI's exit code 8 as JSON and as human output; the confirmation coming before a missing engine or ffmpeg-skill.
  - Unit: success with the flag; the provenance record and its time window; contract and doctor fields.
  - Integration (real ffmpeg-skill): refusal without the flag; acknowledgement recorded with it.
  - Every existing SFX test now passes the flag; none was loosened.

### What this does not do

- It does not check that a person actually saw the message. The flag is the caller's statement that the user was told; the provenance records that statement and its time.
- It does not change ADR-005: EzAudio stays the SFX engine, and `output_warning` stays in every result's `warnings`.
- video-production-agent is not changed here. It reads `confirmation_required` from the contract or doctor and turns it into a CONFIRM decision on its side.

## ADR-008 Library mode: the user's own files with a declared licence, chosen deterministically, fitted and measured through ffmpeg-skill, no engine

### Context

Roadmap step 4: "import user BGM/SFX with a licence field", done when it "refuses a missing licence". ADR-002 declared the capability `audio.library_import` (tool `library-import`) as `PROPOSED`. Library mode is the non-AI path: music beds and effects the user already has (bought packs, commissioned tracks, CC0 / CC BY downloads) instead of ACE-Step or EzAudio output. The owner set three rules for it:
- EzAudio is never used or invoked by library mode, and nothing is downloaded. The EzAudio confirmation gate (ADR-007) is not touched.
- A file whose licence is unknown is reported as unknown and is never presented as cleared. No licence data is invented.
- Durations and loudness are measured through ffmpeg-skill, not assumed. ffmpeg-skill is finished and is not changed.

A file says nothing reliable about its own licence. Tags in a WAV or MP3 are optional and often wrong, and reading them would be guessing. The user has to declare it, and the skill has to keep that declaration apart from anything it measures.

### Decision

- **No engine.** `library.py` imports no engine, no worker and no network module. A unit test scans its source for them. Every library test patches every engine entry point (`ops.run_engine`, `ops.select_engine`, `ops.require_confirmation`, `service.run_engine`, `EzAudioEngine.inspect` / `process`, `AceStepEngine.inspect` / `process`) to fail if reached. Results and provenance say `engine: null`, and the provenance says `ai_generated: false`. Contract (`policy.library_engine`, each library tool's `engine: null`, `engines: []`) and doctor (`checks.operations.library`: `engine: null`, `confirmation_required: false`, a note) publish it.
- **The library is a folder plus one manifest.** `sound-library.json` at the folder's root, schema `sound-design/library@1`:
  - `files` is keyed by the library-relative path, `/`-separated, in Unicode NFC. macOS hands out NFD names; a manifest typed on Windows or Linux is NFC; both sides are normalised, so `ドア.wav` matches either way.
  - Each entry holds what the user declares: `licence`, `kind` (`bgm` / `sfx`), `tags`, `mood`, `loopable`, `source`, `author`, `credit`. `library-import` also records the facts it measured (`sha256`, `measured`: duration, codec, rate, channels, loudness), `imported_from` (a file name, never a path), `imported_at` and `imported_by`.
  - It is read as UTF-8 with an optional byte-order mark (Windows Notepad) and written as UTF-8, not ASCII-escaped, because people read and edit it. It is written whole and replaced with one rename. ADR-003 still holds for every CLI document: those stay ASCII.
  - Hidden files and folders (a leading `.`, including this skill's temporary files) and the manifest itself are not library files. Audio files are found by extension: `.wav .flac .mp3 .m4a .aac .ogg .oga .opus .aif .aiff .caf`.
  One manifest per folder was chosen over a sidecar per file: one file to read, back up and review, and a library can be indexed in one pass.
- **Licence honesty.** `licence_info` reports exactly what is declared:
  - `licence` is the declared text, or `"unknown"` when the file has no entry, or its licence is missing, not text, empty or a placeholder word (`unknown`, `unspecified`, `none`, `n/a`, `?`, `tbd`, ...). `declared_as` keeps what was written.
  - `status` is `declared` or `unknown`; `usable` follows it; `verified` is always `false`. The note says the licence is the user's declaration, and that this skill neither verifies nor clears licences (already in `not_provided`). The word "cleared" is never used for a file.
  - Warnings state facts of the declared text, never a verdict: a non-commercial term (NC), a no-derivatives term (ND) when the file was trimmed or looped, and a Creative Commons attribution licence with no `credit` declared. A file that changed since its import (SHA-256) is flagged, because the licence was declared for the imported bytes.
- **"Refuses a missing licence".** `library-import` checks the licence first, before the input, the library or ffmpeg-skill, and refuses a missing, empty or placeholder licence with the new error `LICENCE_REQUIRED`: exit code 2 (an input problem, like `INVALID_ARGUMENT`), not retryable. The message is English then Japanese, as `CONFIRMATION_REQUIRED` (ADR-007); `details` carries `message_en`, `message_ja`, `licence: "unknown"` and `declared_as`. `library-pick --file` refuses a file without a declared licence the same way. Without `--file`, such a file is never chosen and is listed under `excluded` with the reason. `library-index` lists it with `licence: "unknown"`.
- **`library-import <file> --library <dir> --licence <SPDX|text> --kind bgm|sfx [--tags] [--mood] [--loopable] [--source] [--author] [--credit] [--name <relative path>] [--no-loudness] [--overwrite]`.**
  - Order: licence, then arguments, then the input file and the library, then the destination, then ffmpeg-skill.
  - The input must be audio only by `ffmpeg-skill/probe` (one audio stream, no picture, a duration). Its loudness is measured by `ffmpeg-skill/loudness --measure-only` unless `--no-loudness`.
  - A file outside the library is copied in (verified by SHA-256, placed with one rename) and the input is never modified. A file already inside the library is registered where it is (`copied: false`); `--name` cannot move it.
  - `--name` must be relative, inside the library and free of `..`.
  - An existing file or manifest entry is replaced only with `--overwrite`.
  - Schema `sound-design/library-import@1`. `audio.library_import` moves to `EXPERIMENTAL`.
- **`library-index <dir> [--loudness]`.** Read-only. Every audio file, in path order, with its licence block, kind, tags, mood, loopable, SHA-256, the duration and stream facts measured by `ffmpeg-skill/probe`, and with `--loudness` its EBU R128 integrated loudness, true peak and LRA. Also listed: unreadable files, manifest entries whose file is gone, files changed since import, and a summary (files, declared, unknown licence, unreadable, missing). Schema `sound-design/library-index@1`.
- **`library-pick <dir> -o <out.wav> --kind bgm|sfx [--tags] [--mood] [--duration <s>] [--file <relative path>] [--crossfade 0.5] [--overwrite]`.** Schema `sound-design/library-pick@1`.
  - Eligible: in the manifest with a declared licence; of the requested kind; matching at least one requested tag or mood (NFKC + casefold, so `Calm`, `calm` and `ｃａｌｍ` match); audio only by `ffmpeg-skill/probe`; and fittable (below).
  - Ranked by: more matched tags and moods first; then fit `none` < `trim` < `loop`; then the smaller |measured - requested| duration; then the library-relative path in code-point order; then the SHA-256. Nothing is random and nothing depends on the order the file system lists files in, so the same library and query give the same file. The result lists every ranked file and every excluded file with its reasons.
  - `--file` skips the choice but not the licence, kind or fit checks.
- **Fitting (`plan_fit`, pure).** The tolerance is GENERATE_BGM's: max(0.1 s, 1 %).
  - `none`: no `--duration`, or the file is within the tolerance. A PCM WAV is copied byte for byte; any other format is converted to PCM WAV by `ffmpeg-skill/cut --accurate` over its whole length.
  - `trim`: a longer file. The first `duration` seconds, by `ffmpeg-skill/cut --accurate` (sample-accurate, re-encoded to PCM). There is no fade: fades are mixing, which audio-production-skill MIX owns; the result warns that the cut is hard.
  - `loop`: a shorter file, only when the manifest marks it `loopable`. `repeats` copies are joined by `ffmpeg-skill/join` with an audio crossfade x = min(`--crossfade`, a quarter of the file). repeats = max(2, ceil((target - x) / (source - x))), the fewest copies with repeats * source - (repeats - 1) * x >= target. Then the join is trimmed by `cut --accurate`. At most 64 copies. With x = 0, `join --transition none` is used.
  - Refused: a shorter file that is not loopable, or one that needs more than 64 copies. With `--file` this is `INVALID_ARGUMENT`; otherwise the file is excluded with the reason.
- **Validation and placement.** As GENERATE_BGM: the work happens in a private `.sound-design-*` directory beside the output. `ffmpeg-skill/probe` must find one audio stream and no picture, the source's sample rate, and the expected duration within the tolerance; anything else is `INVALID_RESULT`. `ffmpeg-skill/loudness --measure-only` measures the result; it is reported, never normalised (loudness targets are MIX's). One rename places the WAV; nothing is written to the output path otherwise. The output may not be a library file (`OUTPUT_IS_INPUT`), and the library is never modified by `library-pick` or `library-index`.
- **Provenance (`build_library_provenance`).** `op: "library"`, `engine: null`, `ai_generated: false`, `source` (library-relative file, SHA-256, kind, tags, mood, loopable, measured facts), `licence` (the declared block, `verified: false`, credit), `selection` (rule, query, matched tags), `fit` (the plan), `duration` (requested, measured, tolerance), `output.sha256`, the ffmpeg-skill version, `network: false`. No absolute path is recorded. The result's `handoff` is audio-production-skill MIX, which places the WAV with its licence and credit.
- **ffmpeg-skill.** `TOOLS_USED` grows from `probe` to `probe`, `cut`, `join` and `loudness`. Their flags are checked against the live contract like `probe`'s (capability detection), and all four declare `audio_only`. Nothing in ffmpeg-skill was changed.

### ffmpeg-skill gaps found (ffmpeg-skill is not changed; recorded here)

- `loop` refuses an audio-only input ("input has no video stream"; its contract has `audio_only: false`). Library mode loops with `join` (N copies of the same file, audio crossfade), then `cut`. `join` reports the repeated input as a `duplicate_inputs` warning, which is expected here.
- `join --transition fade --duration 0` does not join end to end: two 3 s files gave 5.08 s, not 6 s. `--transition none` gives exactly 6 s, so a crossfade of 0 uses it.
- `cut` without `--accurate` stream-copies a WAV at packet precision (1.259 s for 1.25 s). Library mode always passes `--accurate`.

### What this does not do

- It does not verify, look up or clear any licence, and it reads no licence from file tags. The declared text is carried as declared.
- It does not normalise loudness, fade, duck or mix. MIX does.
- It does not change GENERATE_BGM, GENERATE_SFX or the EzAudio confirmation (ADR-007).

### Tests

- Unit (`tests/test_library.py`, no FFmpeg; a stub ffmpeg-skill that cuts and joins PCM WAVs with the `wave` module):
  - licence reporting: missing and placeholder licences are unknown, the declared text is kept, `verified` is false, the warnings, and `LICENCE_REQUIRED` in both languages;
  - the fitting maths: the fewest repeats on a grid, the crossfade cap, exact multiples, the refusals;
  - selection: exclusion reasons; rank order and tie-breaks; the same choice for every permutation of the input;
  - indexing: Japanese names, an NFD file name against an NFC key, hidden and non-audio files, an unreadable file, a changed file, a missing file, a manifest with a BOM, invalid manifests, read-only;
  - import: refusal before any ffmpeg-skill call, copy and manifest, registration in place, `--name` safety, `OUTPUT_EXISTS` / `--overwrite`;
  - pick: loop, trim, whole-file copy, determinism, `--file` refusals, no eligible file, output checks, the provenance, the CLI;
  - every engine entry point patched to fail; a static scan of `library.py`; contract and doctor fields; the `LICENCE_REQUIRED` CLI document on a cp1252 console with a Japanese file name (ADR-003).
- Integration (real ffmpeg-skill and FFmpeg): WAVs written by the test under Japanese names in a Japanese-named library, plus a FLAC made by `ffmpeg-skill/cut`. Covered: refusal without a licence; import; index with loudness and one unknown file; trim to 7.5 s and loop to 10 s, both checked by the real `probe`; the same choice and SHA-256 twice; FLAC converted to WAV; a PCM WAV used as is; `--file` refused for the unknown file.
