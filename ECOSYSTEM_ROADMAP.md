# Roadmap — sound-design-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **音 (Audio)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Scaffold: one-line README. Engine in ecosystem.json: ACE-Step 1.5 (MIT, commercial use of output stated, <4 GB VRAM). No SFX engine named yet (since step 3: EzAudio, ADR-005).

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| BGM生成 | experimental | GENERATE_BGM (ACE-Step 1.5), step 2 |
| 効果音 | experimental | GENERATE_SFX (EzAudio), step 3 |

## Wiring

None. Generated assets should be placed through audio-production MIX (it owns ミックス), not mixed here.

## Steps

- [x] **1. Skeleton + contract/doctor + ACE-Step worker** ([#3](https://github.com/kajisho5/sound-design-skill/pull/3), [docs/decisions.md](docs/decisions.md) ADR-001 engines, ADR-002 skeleton, ADR-003 ASCII JSON; `sound-design doctor` reports ACE-Step `OK` / `MISSING` / `INCOMPATIBLE` with an install line, the device and GPU memory; OS static checks PASS) — doctor reports device/VRAM
  - Done when: OS static checks PASS
- [x] **2. GENERATE_BGM op** ([#4](https://github.com/kajisho5/sound-design-skill/pull/4), [docs/decisions.md](docs/decisions.md) ADR-004; `sound-design generate-bgm` through ACE-Step 1.5's Python API as read from its source, DiT only (LM off until the Qwen3 base licences are verified), no Hugging Face / ModelScope path reachable, WAV validated through ffmpeg-skill probe; duration tolerance and seed determinism tested with a fake engine, the real-engine measurement is an ADR-004 follow-up) — prompt/tags, duration, seed; provenance with engine, seed, licence
  - Done when: duration within tolerance; seed determinism (verify on the engine)
- [x] **3. SFX engine ADR + GENERATE_SFX** ([#5](https://github.com/kajisho5/sound-design-skill/pull/5), [docs/decisions.md](docs/decisions.md) ADR-005, ADR-006; EzAudio chosen: MIT code and weights, flan-t5-xl Apache-2.0, no output terms stated, its YouTube training audio recorded as a warning in every result; Stable Audio Open rejected (Community License caps commercial use at USD 1M revenue), AudioLDM2 rejected (CC BY-NC-SA code and weights), ten more checked; `sound-design generate-sfx` through EzAudio's Python API as read from its source, no download path reachable, WAV validated through ffmpeg-skill probe; tested with a fake engine, real-engine measurements are an ADR-006 follow-up) — candidates to verify: Stable Audio Open, AudioLDM2 (licences)
  - Done when: ADR + tests
- [ ] **4. Library mode** — import user BGM/SFX with a licence field
  - Done when: refuses a missing licence
- [ ] **5. SKILL.md, CI; (agent side) BGM -> MIX**
  - Done when: end-to-end plan

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
