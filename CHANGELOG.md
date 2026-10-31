# Changelog

## 0.1.0 - unreleased

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
