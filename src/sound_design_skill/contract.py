"""Machine-readable contract (`sound-design contract --json`, alias `skill --json`), in the shape the AI Video
Production OS registry reads (skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict

from . import PACKAGE_NAME, SKILL_ID, __version__
from .device import ACESTEP_TIERS, FIT_STATUSES, REFUSED_FITS, VRAM_TOLERANCE_GB
from .doctor import DOCTOR_SCHEMA, IMPLEMENTED_OPS, OP_STEP
from .engines import DEFAULT_ENGINE, confirmation_for, engine_ids, engines_for, get_engine
from .engines import ezaudio
from .engines.acestep import DEFAULT_MODEL, DEVICES, MODEL_IDS
from .errors import ERROR_TABLE
from .ffmpeg_skill import FLAGS_USED, SUPPORTED_CONTRACT_VERSION, TOOLS_USED
from .provenance import ENGINE_FIELDS
from . import pcm
from . import tempo
from . import cues
from . import stock
from . import shaping

CONTRACT_VERSION = "1.0"
CAPABILITY_IDS = {"bgm": "audio.bgm_generate", "sfx": "audio.sfx_generate", "library": "audio.library_import"}
TOOL_NAMES = {"bgm": "generate-bgm", "sfx": "generate-sfx", "library": "library-import"}
_OP_TOOLS = {
    "bgm": {"description": "A music bed from a prompt / tags, a target duration and a seed; instrumental unless lyrics are "
                           "given; validated through ffmpeg-skill/probe and placed with one rename",
            "inputs": ["prompt and / or tags (joined into one caption, < 512 characters)", "duration in seconds (engine range)",
                       "seed (drawn when omitted; always recorded)"],
            "argv": ["generate-bgm", "--prompt <text>", "[--tags <a,b,...>]", "--duration <seconds>", "-o <out.wav>",
                     "[--seed <int>]", f"[--bpm <{tempo.BPM_RANGE[0]}-{tempo.BPM_RANGE[1]}>]",
                     "[--time-signature " + "|".join(map(str, tempo.TIME_SIGNATURES)) + "]", "[--engine acestep]",
                     "[--model " + "|".join(MODEL_IDS) + "]", "[--lyrics <file>]", f"[--device {'|'.join(DEVICES)}]",
                     "[--timeout 3600]", "[--overwrite]", "[--ffmpeg-skill <dir>]", "[--offline]", "--json"],
            "schema": "sound-design/generate-bgm@1",
            "duration_tolerance": "max(0.1 s, 1 % of the requested duration), measured by ffmpeg-skill/probe",
            "language_model": "off (Qwen3 base-model licences unverified; ADR-004)",
            "tempo": {"flags": ["--bpm", "--time-signature"], "engine_params": "GenerationParams.bpm (int) / timesignature",
                      "duration": "with --bpm: the whole number of bars nearest --duration (a half bar rounds up), inside "
                                  "the engine's range, so the bed ends on a bar line",
                      "beat_grid": f"<output>{tempo.GRID_SUFFIX} beside the WAV (schema {tempo.GRID_SCHEMA}) and "
                                   "tempo_map in the result: every beat, downbeats, bar lines; first beat assumed at 0 s; "
                                   "measured: false (ADR-012)"},
            "models": {"flag": "--model", "request_key": "model", "default": DEFAULT_MODEL, "choices": list(MODEL_IDS),
                       "specs": "engines.registered.acestep.models (licence, output use, files, GPU memory, download "
                                "size per model)",
                       "readiness": "doctor checks.operations.bgm.models",
                       "checked_before_any_work": {"files": "MODEL_NOT_FOUND, details.install = the model's download line",
                                                   "gpu_memory": "INSUFFICIENT_VRAM for a model whose vram_check is "
                                                                 "refuse, when the measured GPU memory (+ "
                                                                 f"{VRAM_TOLERANCE_GB:g} GB) is below its min_vram_gb or "
                                                                 "no GPU is found; not measured: allowed with a warning; "
                                                                 "details.default_model is the fallback"},
                       "recorded_in": ["request.model", "engine.model", "engine.weights_license", "engine.output_use",
                                       "provenance.engine.model", "provenance.parameters.model"],
                       "adr": "docs/decisions.md ADR-016"}},
    "sfx": {"description": "A sound effect from a prompt / tags, a duration in whole seconds and a seed (EzAudio, mono "
                           "24 kHz); validated through ffmpeg-skill/probe and placed with one rename",
            "inputs": ["prompt and / or tags (joined, < 512 characters; the text encoder reads 100 tokens)",
                       "duration in whole seconds (engine range; EzAudio 1-10)", "seed (drawn when omitted; always recorded)"],
            "argv": ["generate-sfx", "--prompt <text>", "[--tags <a,b,...>]", "--duration <seconds>", "-o <out.wav>",
                     "[--seed <int>]", "[--engine ezaudio]", f"[--device {'|'.join(ezaudio.DEVICES)}]", "[--timeout 600]",
                     "[--overwrite]", "[--ffmpeg-skill <dir>]", "[--offline]", ezaudio.ACK_FLAG, "--json"],
            "schema": "sound-design/generate-sfx@1",
            "duration_tolerance": "max(0.1 s, 1 % of the requested duration), measured by ffmpeg-skill/probe",
            "output_warning": ezaudio.OUTPUT_WARNING},
    "library": {"description": "Register one user-supplied BGM / SFX file in a library folder with its declared licence "
                               "(copied in, or registered in place); refuses a missing licence (LICENCE_REQUIRED); "
                               "measured through ffmpeg-skill probe / loudness; no engine (ADR-008)",
                "inputs": ["audio file", "library folder", "licence (required; SPDX id or the licence text; never inferred)",
                           "kind bgm|sfx", "tags / mood", "loopable", "source / author / credit"],
                "argv": ["library-import", "<file>", "--library <dir>", "--licence <SPDX|text>", "--kind bgm|sfx",
                         "[--tags <a,b,...>]", "[--mood <a,b,...>]", "[--loopable]", "[--source <url|text>]",
                         "[--author <text>]", "[--credit <text>]", "[--title <text>]", "[--provider <name>]",
                         "[--licence-url <url>]", "[--licence-file <text file>]", "[--attribution <line>]",
                         "[--content-id registered|not-registered]", "[--name <relative path>]", "[--no-loudness]",
                         "[--overwrite]", "[--ffmpeg-skill <dir>]", "[--timeout 600]", "--json"],
                "schema": "sound-design/library-import@1",
                "manifest": "sound-library.json at the library root (schema sound-design/library@1), UTF-8, keyed by "
                            "the library-relative POSIX path in Unicode NFC"},
}
_LIBRARY_TOOLS = [
    {"tool_id": f"{SKILL_ID}/library-index", "role": "diagnostic", "writes_media": False, "mutates_input": False,
     "deterministic": True, "implemented": True, "roadmap_step": 4, "engine": None, "engines": [],
     "description": "Every audio file in a library folder with its licence status (declared / unknown: a file without "
                    "a declared licence is reported as licence \"unknown\" and never used), kind, tags, mood and the "
                    "duration measured by ffmpeg-skill/probe (with --loudness, EBU R128 via ffmpeg-skill/loudness "
                    "--measure-only); read-only",
     "argv": ["library-index", "<dir>", "[--loudness]", "[--ffmpeg-skill <dir>]", "[--timeout 600]", "--json"],
     "schema": "sound-design/library-index@1"},
    {"tool_id": f"{SKILL_ID}/library-pick", "role": "selection", "writes_media": True, "mutates_input": False,
     "deterministic": True, "implemented": True, "roadmap_step": 4, "engine": None, "engines": [],
     "description": "One library file chosen deterministically (declared licence, kind, tags / mood, fit), trimmed or "
                    "looped with a crossfade to the requested duration through ffmpeg-skill cut / join, validated by "
                    "ffmpeg-skill/probe, loudness measured, placed as a WAV with its provenance (source SHA-256, the "
                    "licence as declared, the selection and the fit); no engine (ADR-008)",
     "inputs": ["library folder", "kind bgm|sfx (or --file)", "tags / mood", "duration in seconds (optional)"],
     "argv": ["library-pick", "<dir>", "-o <out.wav>", "--kind bgm|sfx", "[--tags <a,b,...>]", "[--mood <a,b,...>]",
              "[--duration <seconds>]", "[--file <relative path>]", "[--crossfade 0.5]", "[--overwrite]",
              "[--ffmpeg-skill <dir>]", "[--timeout 600]", "--json"],
     "schema": "sound-design/library-pick@1",
     "selection": "eligible: declared licence, requested kind, >= 1 requested tag / mood, audio only, fittable; ranked "
                  "by matched tags + moods, fit (none < trim < loop), |measured - requested| duration, library-relative "
                  "path, SHA-256",
     "duration_tolerance": "max(0.1 s, 1 % of the requested duration), measured by ffmpeg-skill/probe",
     "fit": {"none": "within the tolerance: used whole (PCM WAV copied as is, other formats converted)",
             "trim": "longer: the first <duration> seconds, ffmpeg-skill/cut --accurate, no fade (MIX fades)",
             "loop": "shorter and marked loopable: N copies joined by ffmpeg-skill/join with a crossfade (at most a "
                     "quarter of the file), then cut; at most 64 copies",
             "refused": "shorter and not loopable"}},
]


_SHAPING_TOOLS = [
    {"tool_id": f"{SKILL_ID}/loop-bed", "role": "shaping", "writes_media": True, "mutates_input": False,
     "deterministic": True, "implemented": True, "roadmap_step": 7, "engine": None, "engines": [],
     "description": "A bed whose end runs seamlessly into its start, for loops in an edit: the loop point is searched "
                    "near --duration for the place whose continuation best matches the source's start, the "
                    "continuation is crossfaded into the start (ffmpeg-skill cut / join), and the wrap is measured "
                    "(click, level, spectrum) and refused if it fails; no engine (ADR-011)",
     "inputs": ["source WAV (a generate-bgm / library-pick result or the user's file), at least duration + crossfade",
                "loop length in seconds", "search window", "crossfade"],
     "argv": ["loop-bed", "<in.wav>", "-o <loop.wav>", "--duration <seconds>", f"[--search {shaping.DEFAULT_SEARCH:g}]",
              f"[--crossfade {shaping.DEFAULT_LOOP_CROSSFADE:g}]", "[--overwrite]", "[--ffmpeg-skill <dir>]",
              "[--timeout 600]", "--json"],
     "schema": shaping.LOOP_SCHEMA,
     "construction": "crossfade(source[L, L + x] -> source[0, x]) ++ source[x, L]: the loop's last sample is "
                     "source[L - 1] and its first source[L]"},
    {"tool_id": f"{SKILL_ID}/stinger", "role": "shaping", "writes_media": True, "mutates_input": False,
     "deterministic": True, "implemented": True, "roadmap_step": 7, "engine": None, "engines": [],
     "description": "A transition sound (whoosh, hit, rise) of exactly --duration seconds whose peak (the loudest 10 ms) "
                    "is at --peak-at seconds, so it lands on the cut: cut through ffmpeg-skill (5 / 10 ms fades where it "
                    "was cut short), padded with digital silence, re-measured; refused unless the peak is within "
                    f"{shaping.PEAK_TOLERANCE * 1000:g} ms and the length exact; no engine (ADR-011)",
     "inputs": ["source sound", "exact duration in seconds", "peak time in seconds", "type (recorded)"],
     "argv": ["stinger", "<in.wav>", "-o <out.wav>", "--duration <seconds>", "--peak-at <seconds>",
              f"[--type {'|'.join(shaping.STINGER_TYPES)}]", "[--overwrite]", "[--ffmpeg-skill <dir>]",
              "[--timeout 600]", "--json"],
     "schema": shaping.STINGER_SCHEMA, "peak_tolerance_s": shaping.PEAK_TOLERANCE, "peak": pcm.PEAK_WINDOW},
    {"tool_id": f"{SKILL_ID}/seam-check", "role": "diagnostic", "writes_media": False, "mutates_input": False,
     "deterministic": True, "implemented": True, "roadmap_step": 7, "engine": None, "engines": [],
     "description": "Click, level and spectrum across each seam of a WAV (every --period seconds and / or at --seams); "
                    "exit 1 when a seam fails; read-only. The loop seam check qc-skill does not have (ADR-011)",
     "argv": ["seam-check", "<file.wav>", "[--period <seconds>]", "[--seams <t,t,...>]",
              f"[--window {pcm.SEAM_WINDOW:g}]", "[--ffmpeg-skill <dir>]", "[--timeout 600]", "--json"],
     "schema": shaping.SEAM_SCHEMA, "limits": pcm.seam_limits(), "method": shaping.SEAM_METHOD},
]


_CUE_TOOLS = [
    {"tool_id": f"{SKILL_ID}/sfx-cues", "role": "planning", "writes_media": False, "mutates_input": False,
     "deterministic": True, "implemented": True, "roadmap_step": 9, "engine": None, "engines": [],
     "description": "SFX_CUES: a cue sheet of sound effects (pop, whoosh, ding, ...) at the cut points and emphasis times "
                    "the caller gives, for audio-production-skill MIX to place: one cue per event at the caller's time "
                    "(the effect's peak or onset on it), a gain to a stated level, events closer than a minimum spacing "
                    "dropped, never moved; effects from the library (declared licence) or a named WAV; the skill never "
                    "decides where an emphasis is and mixes nothing (ADR-013). Opt-in: a minimum reuse spacing for the "
                    "same sound file (the next library file in rank order, else the event is dropped) and a character "
                    "filter on the library's tags and mood (ADR-015)",
     "inputs": ["cut times and / or emphasis times (or an events JSON)", "library folder and / or NAME=WAV files",
                "effect per event kind", "minimum spacing", "level", "minimum reuse spacing (optional, off by default)",
                "character words per effect or per event (optional)"],
     "argv": ["sfx-cues", "[--cuts <t,t,...>]", "[--emphasis <t,t,...>]", "[--events <events.json>]", "[--library <dir>]",
              "[--effect cut|emphasis=<name>]", "[--file <name>=<wav>]", f"[--min-spacing {cues.DEFAULT_SPACING:g}]",
              "[--min-reuse-spacing <seconds>]", "[--character <effect>=<word,...>]",
              f"[--level {cues.DEFAULT_LEVEL:g}]", "[-o <cues.json>]", "[--overwrite]", "[--ffmpeg-skill <dir>]",
              "[--timeout 600]", "--json"],
     "schema": cues.CUES_SCHEMA, "default_effects": dict(cues.DEFAULT_EFFECTS),
     "align": {"peak": sorted(cues.PEAK_EFFECTS), "onset": "every other effect (first sample at 10 % of the peak)"},
     "reuse": {"flag": "--min-reuse-spacing", "request_key": "min_reuse_spacing", "default": None,
               "range_s": [0.0, cues.MAX_REUSE_SPACING], "same_file": "the same SHA-256", "rule": cues.REUSE_RULE,
               "recorded_in": ["substitutions", "cues[].substituted", "dropped[].by = min_reuse_spacing"]},
     "character": {"flag": "--character", "request_key": "characters", "event_key": "character",
                   "matches": "every word in the library file's tags or mood (NFKC, case-insensitive)",
                   "vocabulary": list(cues.CHARACTER_VOCABULARY), "vocabulary_closed": False,
                   "rule": cues.CHARACTER_RULE}},
]


def tools() -> list:
    out = []
    for op, spec in _OP_TOOLS.items():
        out.append({"tool_id": f"{SKILL_ID}/{TOOL_NAMES[op]}", "role": "generation" if op != "library" else "ingest",
                    "writes_media": True, "mutates_input": False, "deterministic": op == "library",
                    "implemented": IMPLEMENTED_OPS[op],
                    "roadmap_step": OP_STEP[op], "engine": DEFAULT_ENGINE.get(op), "engines": engines_for(op),
                    "confirmation_required": confirmation_for(op) is not None, "confirmation": confirmation_for(op), **spec})
    out += [dict(t) for t in _LIBRARY_TOOLS]
    out += [dict(t) for t in _SHAPING_TOOLS]
    out += [dict(t) for t in _CUE_TOOLS]
    out += [
        {"tool_id": f"{SKILL_ID}/doctor", "role": "diagnostic", "writes_media": False, "mutates_input": False, "deterministic": False,
         "implemented": True, "description": "What is installed and missing (engine, weights, ffmpeg-skill), compute device and "
                                             "GPU memory, and how to get what is missing; never downloads",
         "argv": ["doctor", "[--offline]", "[--ffmpeg-skill <dir>]", "--json"], "schema": DOCTOR_SCHEMA},
        {"tool_id": f"{SKILL_ID}/engines", "role": "diagnostic", "writes_media": False, "mutates_input": False, "deterministic": False,
         "implemented": True, "description": "Registered engines with status, version, licences and output-use terms",
         "argv": ["engines", "[--offline]", "--json"]},
        {"tool_id": f"{SKILL_ID}/contract", "role": "metadata", "writes_media": False, "mutates_input": False, "deterministic": True,
         "implemented": True, "description": "This contract", "argv": ["contract", "--json"]},
    ]
    return out


def contract() -> Dict[str, Any]:
    deps = [
        {"id": "ffmpeg-skill", "kind": "skill", "required": True, "for": [CAPABILITY_IDS[o] for o in CAPABILITY_IDS],
         "version_range": "any release whose contract declares the tools and flags below (capability detection)",
         "tools": list(TOOLS_USED), "flags": {t: list(f) for t, f in FLAGS_USED.items()},
         "contract_version": SUPPORTED_CONTRACT_VERSION,
         "located_by": ["--ffmpeg-skill", "$SOUND_DESIGN_FFMPEG_SKILL_DIR", "$VIDEO_AGENT_FFMPEG_SKILL_DIR",
                        "~/.claude/skills/ffmpeg-skill", "./vendor/ffmpeg-skill", "../ffmpeg-skill"]},
    ]
    for eid in engine_ids():
        s = get_engine(eid).spec
        deps.append({"id": eid, "kind": "engine-" + s.kind, "required": False, "for": [CAPABILITY_IDS[o] for o in s.ops],
                     "version_range": s.version_range, "license": s.code_license, "weights_license": s.weights_license,
                     "output_use": s.output_use, "credit_required": s.credit_required, "watermark": s.watermark,
                     "install": s.install, "located_by": list(s.located_by), "downloaded_at_run_time": False})
    return {
        "skill_id": SKILL_ID,
        "id": SKILL_ID,
        "skill": PACKAGE_NAME,
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "provides": [{"id": CAPABILITY_IDS[op], "lifecycle": "EXPERIMENTAL" if IMPLEMENTED_OPS[op] else "PROPOSED",
                      "tool_id": f"{SKILL_ID}/{TOOL_NAMES[op]}"} for op in CAPABILITY_IDS],
        "dependencies": deps,
        "not_provided": ["mixing BGM / SFX into the programme, ducking, loudness targets (audio-production-skill MIX owns ミックス)",
                         "voice, narration or dubbing (dubbing-skill)", "noise reduction / voice isolation (voice-enhance-skill)",
                         "music recognition, licence clearance or copyright checks of generated music",
                         "downloading engines or model weights (one-time manual step; doctor says how)",
                         "network access at run time", "direct ffmpeg calls (every media step is an ffmpeg-skill tool)"],
        "handoff": {"cues": "audio-production-skill MIX places each cue's file at its start with its gain; the times are "
                            "the caller's, the skill adds and moves none; each cue's file and licence are under "
                            "sources[cue.source]",
                    "loop": "the editor or audio-production-skill MIX repeats the loop end to end (no crossfade)",
                    "stinger": "audio-production-skill MIX places the stinger starting peak_at seconds before the cut",
                    "bgm": "audio-production-skill MIX places the generated WAV (with its provenance) under the programme",
                    "sfx": "audio-production-skill MIX places the generated effect (with its provenance) in the programme",
                    "library": "audio-production-skill MIX places the fitted library WAV (with its provenance, licence and "
                               "credit) in the programme"},
        "tools": tools(),
        "engines": {"registered": {eid: get_engine(eid).spec.to_dict() for eid in engine_ids()},
                    "default": dict(DEFAULT_ENGINE),
                    "selection_adr": "docs/decisions.md ADR-001 (BGM), ADR-005 (SFX)",
                    "optional": True,
                    "statuses": ["OK", "MISSING", "INCOMPATIBLE"]},
        "stock": {"reported_by": "doctor checks.stock_sources", "sources": [x["id"] for x in stock.SOURCES],
                  "statuses": ["MISSING (API key not set)", "KEY_SET", "MANUAL (no API)"], "fetch": stock.FETCH_NOTE},
        "device": {"reported_by": "doctor checks.device", "detected_by": ["nvidia-smi --query-gpu", "platform + sysctl hw.memsize (Apple Silicon)",
                                                                          "rocm-smi on PATH (noticed, not measured)",
                                                                          "xpu-smi on PATH (Intel XPU; noticed, not measured)"],
                   "statuses": ["OK", "CPU_ONLY", "UNKNOWN"], "acestep_tiers": [t for _, t in ACESTEP_TIERS],
                   "model_fit": {"statuses": list(FIT_STATUSES), "refused": list(REFUSED_FITS),
                                 "tolerance_gb": VRAM_TOLERANCE_GB, "compared": "the largest GPU found (Apple Silicon: "
                                                                                "the unified memory)",
                                 "xpu": "an Intel GPU is only noticed through xpu-smi; with --device xpu and no GPU "
                                        "found, the memory counts as not measured (UNKNOWN), not as NO_GPU"}},
        "policy": {"output_use": "each engine's output-use terms and training-data claim are published in its EngineSpec "
                                 "and copied into the provenance of every file; an engine's output_warning is copied into "
                                 "every result's warnings",
                   "credit": "an engine's credit (credit_required) is carried into the provenance and must appear in the "
                             "published video's description or credits",
                   "watermark": "an engine's watermark is never removed or bypassed; it is recorded in the provenance",
                   "library_licence": "library mode uses a user file only with a declared licence: library-import "
                                      "refuses a missing, empty or placeholder (\"unknown\") licence with "
                                      "LICENCE_REQUIRED; library-index reports such files as licence \"unknown\"; "
                                      "library-pick never chooses them. A declared licence is recorded exactly as "
                                      "declared with verified: false; nothing is inferred or cleared (ADR-008)",
                   "stock_licences": "a stock track the user downloaded is imported with its licence (SPDX id or name), "
                                     "the licence URL and / or text (copied into the library's licences/ folder, "
                                     "SHA-256 recorded), title, provider, author, and the source's Content ID statement "
                                     "(registered / not-registered, else unknown); the receipt carries the attribution "
                                     "line (declared, the credit, or composed in TASL order for a CC BY licence from "
                                     "declared parts only) and warns about a Content ID registered track (ADR-014)",
                   "library_engine": "library mode runs no engine (never ACE-Step or EzAudio, no worker) and downloads "
                                     "nothing; every media step is an ffmpeg-skill tool (ADR-008)",
                   "shaping": "loop-bed and stinger run no engine; they edit a WAV the caller already has through "
                              "ffmpeg-skill only, measure the result from its samples, and refuse a result that fails "
                              "its measurement (a loop's wrap, a stinger's peak time and length); the only audio written "
                              "by this skill itself is digital silence used as stinger padding (ADR-011)",
                   "models": "an engine's models each carry their own weights licence, output-use statement and "
                             "training-data claim as their model card states them; the model that ran fills "
                             "engine.model / weights_license / output_use / training_data in the result and the provenance, "
                             "so a non-default model never inherits the default's statement (ADR-016)",
                   "confirmation": "an engine whose spec names a confirmation_flag (EzAudio: "
                                   f"{ezaudio.ACK_FLAG}) is refused with CONFIRMATION_REQUIRED before any work unless "
                                   "the caller passes that flag; an orchestrator surfaces it as a CONFIRM step, tells the "
                                   "user the tool's confirmation.reason, and passes the flag only after the user agrees; "
                                   "the acknowledgement and its time go into provenance.confirmation (ADR-007)"},
        "provenance": {"fields": ["skill", "op", *[f"engine.{f}" for f in ENGINE_FIELDS], "prompt.text", "prompt.sha256", "seed",
                                  "duration.requested", "duration.measured", "duration.tolerance", "output.sha256", "parameters",
                                  "parameters.model (BGM: id, selected_by, repository, weights_license, output_use, vram)",
                                  "ffmpeg_skill.version", "confirmation.acknowledged", "confirmation.acknowledged_at",
                                  "confirmation.flag", "network"],
                       "library_fields": ["skill", "op", "engine (null)", "ai_generated (false)", "source.file",
                                          "source.sha256", "source.measured", "licence.licence", "licence.status",
                                          "licence.verified", "licence.credit", "licence.attribution",
                                          "licence.attribution_source", "licence.content_id", "licence.licence_url",
                                          "licence.licence_text_sha256", "selection.rule", "selection.query",
                                          "fit.mode", "duration.requested", "duration.measured", "output.sha256",
                                          "ffmpeg_skill.version", "network"],
                       "shaping_fields": ["skill", "op", "engine (null)", "ai_generated (as the source)",
                                          "source.file", "source.sha256", "source.measured", "parameters", "result",
                                          "output.sha256", "ffmpeg_skill.version", "network"],
                       "status": "OBSERVED"},
        "execution": {"canonical_invocation": ["sound-design", "<command>", "...", "--json"], "network": False, "shell": False,
                      "engine_process": "worker subprocess in its own process group (under the engine's own interpreter "
                                        "when configured), killed on --timeout; Hugging Face libraries forced offline, "
                                        "engine download functions replaced, every non-loopback socket refused (ADR-004)",
                      "media": "through ffmpeg-skill tools only", "input_is_read_only": True,
                      "overwrites": "never unless --overwrite; an output equal to an input is refused",
                      "offline_flag": "accepted everywhere; every registered engine works with no network"},
        "errors": {code: {"exit_code": ex, "retryable": rt} for code, (ex, rt) in ERROR_TABLE.items()},
        "error_shape": {"ok": False, "error": {"code": "<ERRORS key>", "message": "str", "retryable": "bool", "details": "object"}},
    }
