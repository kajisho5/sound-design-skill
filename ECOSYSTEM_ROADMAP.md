# Roadmap — sound-design-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **音 (Audio)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Scaffold: one-line README. Engine in ecosystem.json: ACE-Step 1.5 (MIT, commercial use of output stated, <4 GB VRAM). No SFX engine named yet.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| BGM生成 | missing |  |
| 効果音 | missing | SFX engine to choose |

## Wiring

None. Generated assets should be placed through audio-production MIX (it owns ミックス), not mixed here.

## Steps

- [ ] **1. Skeleton + contract/doctor + ACE-Step worker** — doctor reports device/VRAM
  - Done when: OS static checks PASS
- [ ] **2. GENERATE_BGM op** — prompt/tags, duration, seed; provenance with engine, seed, licence
  - Done when: duration within tolerance; seed determinism (verify on the engine)
- [ ] **3. SFX engine ADR + GENERATE_SFX** — candidates to verify: Stable Audio Open, AudioLDM2 (licences)
  - Done when: ADR + tests
- [ ] **4. Library mode** — import user BGM/SFX with a licence field
  - Done when: refuses a missing licence
- [ ] **5. SKILL.md, CI; (agent side) BGM -> MIX**
  - Done when: end-to-end plan

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
