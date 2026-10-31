# Changelog

## 0.1.0 - unreleased

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
