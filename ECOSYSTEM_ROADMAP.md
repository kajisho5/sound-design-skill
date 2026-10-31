# Roadmap — sound-design-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **音 (Audio)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Scaffold: one-line README. Engine in ecosystem.json: ACE-Step 1.5 (MIT, commercial use of output stated, <4 GB VRAM). No SFX engine named yet (since step 3: EzAudio, ADR-005).

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| BGM生成 | experimental | GENERATE_BGM (ACE-Step 1.5), step 2; or the user's own file through library mode, step 4 |
| 効果音 | experimental | GENERATE_SFX (EzAudio), step 3; or the user's own file through library mode, step 4 |

## Wiring

None. Generated assets should be placed through audio-production MIX (it owns ミックス), not mixed here.

## Steps

- [x] **1. Skeleton + contract/doctor + ACE-Step worker** ([#3](https://github.com/kajisho5/sound-design-skill/pull/3), [docs/decisions.md](docs/decisions.md) ADR-001 engines, ADR-002 skeleton, ADR-003 ASCII JSON; `sound-design doctor` reports ACE-Step `OK` / `MISSING` / `INCOMPATIBLE` with an install line, the device and GPU memory; OS static checks PASS) — doctor reports device/VRAM
  - Done when: OS static checks PASS
- [x] **2. GENERATE_BGM op** ([#4](https://github.com/kajisho5/sound-design-skill/pull/4), [docs/decisions.md](docs/decisions.md) ADR-004; `sound-design generate-bgm` through ACE-Step 1.5's Python API as read from its source, DiT only (LM off until the Qwen3 base licences are verified), no Hugging Face / ModelScope path reachable, WAV validated through ffmpeg-skill probe; duration tolerance and seed determinism tested with a fake engine, the real-engine measurement is an ADR-004 follow-up) — prompt/tags, duration, seed; provenance with engine, seed, licence
  - Done when: duration within tolerance; seed determinism (verify on the engine)
- [x] **3. SFX engine ADR + GENERATE_SFX** ([#5](https://github.com/kajisho5/sound-design-skill/pull/5), [docs/decisions.md](docs/decisions.md) ADR-005, ADR-006; EzAudio chosen: MIT code and weights, flan-t5-xl Apache-2.0, no output terms stated, its YouTube training audio recorded as a warning in every result; Stable Audio Open rejected (Community License caps commercial use at USD 1M revenue), AudioLDM2 rejected (CC BY-NC-SA code and weights), ten more checked; `sound-design generate-sfx` through EzAudio's Python API as read from its source, no download path reachable, WAV validated through ffmpeg-skill probe; tested with a fake engine, real-engine measurements are an ADR-006 follow-up; by the owner's decision EzAudio runs only with `--acknowledge-ezaudio-training-data`, else `CONFIRMATION_REQUIRED`, the acknowledgement recorded in the provenance and exposed in contract and doctor: [#6](https://github.com/kajisho5/sound-design-skill/pull/6), ADR-007) — candidates to verify: Stable Audio Open, AudioLDM2 (licences)
  - Done when: ADR + tests
- [x] **4. Library mode** ([#7](https://github.com/kajisho5/sound-design-skill/pull/7), [docs/decisions.md](docs/decisions.md) ADR-008; no engine (never ACE-Step or EzAudio, nothing downloaded): `sound-design library-import` refuses a missing licence with `LICENCE_REQUIRED` and records the declared licence, kind, tags, mood and measurements in `sound-library.json`; `library-index` reports every file's licence as declared or `unknown` (never cleared) with its duration / loudness measured by ffmpeg-skill; `library-pick` chooses deterministically, trims or loops through ffmpeg-skill cut / join, validates with probe and records source SHA-256, licence and fit in the provenance) — import user BGM/SFX with a licence field
  - Done when: refuses a missing licence
- [ ] **5. SKILL.md, CI; (agent side) BGM -> MIX** (skill side done: [#8](https://github.com/kajisho5/sound-design-skill/pull/8), [docs/decisions.md](docs/decisions.md) ADR-009; `SKILL.md` kept in step with the contract by `SkillMdTests`, which run in every job of `tests.yml` (unit on Linux / macOS / Windows x Python 3.9 / 3.12, integration against the real ffmpeg-skill, nothing skipped); the agent-side SkillSpec and BGM -> MIX plan remain in video-production-agent)
  - Done when: end-to-end plan
- [x] **6. ffmpeg-skill 3.0.0** — its contract_version 2.0 read without a warning; CI integration also against 3.0.0 ([#PRNUM](https://github.com/kajisho5/sound-design-skill/pull/PRNUM), ADR-010)

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
