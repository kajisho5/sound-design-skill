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

- **Registry.** The engine registry holds one engine, `acestep` (ACE-Step 1.5), the default for GENERATE_BGM. No SFX engine is registered: `DEFAULT_ENGINE["sfx"]` is `null` until the step 3 ADR.
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
