# Roadmap — sound-design-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **音 (Audio)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

0.1.0: GENERATE_BGM (ACE-Step 1.5, MIT, commercial use of output stated, <4 GB VRAM), GENERATE_SFX (EzAudio, ADR-005) and library mode, experimental; SKILL.md and CI (step 5). `scaffold` in ecosystem.json.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| BGM生成 | experimental | GENERATE_BGM (ACE-Step 1.5), step 2; or the user's own file through library mode, step 4 |
| 効果音 | experimental | GENERATE_SFX (EzAudio), step 3; or the user's own file through library mode, step 4 |

## Wiring

The agent plans `bgm.library` / `bgm.prompt` as a CONFIRM bed that becomes the last input of audio-production MIX (step 5, video-production-agent#107); sound effects are not wired. Generated assets are placed through audio-production MIX (it owns ミックス), not mixed here.

## Steps

- [x] **1. Skeleton + contract/doctor + ACE-Step worker** ([#3](https://github.com/kajisho5/sound-design-skill/pull/3), [docs/decisions.md](docs/decisions.md) ADR-001 engines, ADR-002 skeleton, ADR-003 ASCII JSON; `sound-design doctor` reports ACE-Step `OK` / `MISSING` / `INCOMPATIBLE` with an install line, the device and GPU memory; OS static checks PASS) — doctor reports device/VRAM
  - Done when: OS static checks PASS
- [x] **2. GENERATE_BGM op** ([#4](https://github.com/kajisho5/sound-design-skill/pull/4), [docs/decisions.md](docs/decisions.md) ADR-004; `sound-design generate-bgm` through ACE-Step 1.5's Python API as read from its source, DiT only (LM off until the Qwen3 base licences are verified), no Hugging Face / ModelScope path reachable, WAV validated through ffmpeg-skill probe; duration tolerance and seed determinism tested with a fake engine, the real-engine measurement is an ADR-004 follow-up) — prompt/tags, duration, seed; provenance with engine, seed, licence
  - Done when: duration within tolerance; seed determinism (verify on the engine)
- [x] **3. SFX engine ADR + GENERATE_SFX** ([#5](https://github.com/kajisho5/sound-design-skill/pull/5), [docs/decisions.md](docs/decisions.md) ADR-005, ADR-006; EzAudio chosen: MIT code and weights, flan-t5-xl Apache-2.0, no output terms stated, its YouTube training audio recorded as a warning in every result; Stable Audio Open rejected (Community License caps commercial use at USD 1M revenue), AudioLDM2 rejected (CC BY-NC-SA code and weights), ten more checked; `sound-design generate-sfx` through EzAudio's Python API as read from its source, no download path reachable, WAV validated through ffmpeg-skill probe; tested with a fake engine, real-engine measurements are an ADR-006 follow-up; by the owner's decision EzAudio runs only with `--acknowledge-ezaudio-training-data`, else `CONFIRMATION_REQUIRED`, the acknowledgement recorded in the provenance and exposed in contract and doctor: [#6](https://github.com/kajisho5/sound-design-skill/pull/6), ADR-007) — candidates to verify: Stable Audio Open, AudioLDM2 (licences)
  - Done when: ADR + tests
- [x] **4. Library mode** ([#7](https://github.com/kajisho5/sound-design-skill/pull/7), [docs/decisions.md](docs/decisions.md) ADR-008; no engine (never ACE-Step or EzAudio, nothing downloaded): `sound-design library-import` refuses a missing licence with `LICENCE_REQUIRED` and records the declared licence, kind, tags, mood and measurements in `sound-library.json`; `library-index` reports every file's licence as declared or `unknown` (never cleared) with its duration / loudness measured by ffmpeg-skill; `library-pick` chooses deterministically, trims or loops through ffmpeg-skill cut / join, validates with probe and records source SHA-256, licence and fit in the provenance) — import user BGM/SFX with a licence field
  - Done when: refuses a missing licence
- [x] **5. SKILL.md, CI; (agent side) BGM -> MIX** (skill side done: [#8](https://github.com/kajisho5/sound-design-skill/pull/8), [docs/decisions.md](docs/decisions.md) ADR-009; `SKILL.md` kept in step with the contract by `SkillMdTests`, which run in every job of `tests.yml` (unit on Linux / macOS / Windows x Python 3.9 / 3.12, integration against the real ffmpeg-skill, nothing skipped); agent side: [video-production-agent#107](https://github.com/kajisho5/video-production-agent/pull/107) (its ADR-092) plans `bgm.library` (library-pick) or `bgm.prompt` (generate-bgm) as a CONFIRM `audio.bgm` whose bed, at the mix's length, is the last input of audio-production MIX; the end-to-end plan runs on the real Skills in its CI)
  - Done when: end-to-end plan
- [x] **6. ffmpeg-skill 3.0.0** — its contract_version 2.0 read without a warning; CI integration also against 3.0.0 ([#9](https://github.com/kajisho5/sound-design-skill/pull/9), ADR-010)
- [ ] **7. Loopable beds and timed stingers** — a BGM bed with a seamless loop point (measured: no click, matching level and spectrum across the seam) for loops in an edit, and stinger sounds (whoosh, hit, rise) of an exact length with their peak at a stated time, so a transition's sound lands on its cut point. Field review 2026-11-05 (Velisa ROADMAP, third round)
  - なぜ要るか：ループ素材の BGM と、スティンガーの効果音をカット点に合わせて用意できる
  - Done when: a looped bed repeated 10 times passes qc-skill's loop seam check; a stinger's peak is within 10 ms of the stated time
  - Skill side ([#14](https://github.com/kajisho5/sound-design-skill/pull/14), ADR-011): `loop-bed`, `stinger` and `seam-check`, no engine, edits through ffmpeg-skill; a fixture loop repeated 10 times passes this skill's `seam-check` at all nine seams and a naive one fails; whoosh / hit / rise stingers land within 10 ms. Open: qc-skill has no loop seam check (proposal to Velisa's ROADMAP), and no real ACE-Step / EzAudio output was looped
- [ ] **8. Tempo-targeted generation and a tempo map** — GENERATE_BGM at a requested BPM and length, ending on a bar line, and the beat grid written as data beside the audio (no re-measuring needed downstream)
  - なぜ要るか：「120BPM、15秒」の文字PVに、ぴったり合う音楽と拍の情報を同時に渡せる
  - Done when: ffmpeg-skill `scenes --beats` on the output agrees with the written grid within 20 ms
  - Skill side ([#15](https://github.com/kajisho5/sound-design-skill/pull/15), ADR-012): `generate-bgm --bpm [--time-signature]` passes the tempo to ACE-Step, rounds the length to whole bars and writes `<name>.beats.json`. Open: no real ACE-Step output to measure, and ffmpeg-skill `scenes --beats` refuses audio-only input (on a video wrapper its phase was 40-60 ms off a synthetic click track); proposal to Velisa's ROADMAP
- [ ] **9. Sound-effect cue sheet** — SFX_CUES: from cut points and emphasis times the caller gives, a cue sheet of library or generated effects (pop, whoosh, ding) with levels and a minimum spacing, for audio-production MIX to place; the skill never decides where the emphasis is
  - なぜ要るか：YouTube 風の効果音を、カットや強調テロップの時刻に揃えて置ける
  - Done when: a fixture with 10 cut times yields a cue sheet whose effects sit at those times and respect the spacing
- [ ] **10. Stock music and SFX with licence records** — library mode reads royalty-free sources the user has downloaded (with their licence text or URL), records the licence, the attribution line and whether the track is registered for Content ID when the source says so, and refuses a file without a licence as today
  - なぜ要るか：BGM の権利表記漏れと、Content ID の申し立てを事前に避けられる
  - Done when: a fixture track with a CC BY licence yields its attribution line in the receipt; a track marked Content ID registered is warned about

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (3.x, 42 tools, contract_version 2.0; treated as complete, proposals go to Velisa's ROADMAP). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
