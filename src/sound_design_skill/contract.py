"""Machine-readable contract (`sound-design contract --json`, alias `skill --json`), in the shape the AI Video
Production OS registry reads (skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict

from . import PACKAGE_NAME, SKILL_ID, __version__
from .device import ACESTEP_TIERS
from .doctor import DOCTOR_SCHEMA, IMPLEMENTED_OPS, OP_STEP
from .engines import DEFAULT_ENGINE, engine_ids, engines_for, get_engine
from .engines import ezaudio
from .engines.acestep import DEVICES
from .errors import ERROR_TABLE
from .ffmpeg_skill import FLAGS_USED, SUPPORTED_CONTRACT_VERSION, TOOLS_USED
from .provenance import ENGINE_FIELDS

CONTRACT_VERSION = "1.0"
CAPABILITY_IDS = {"bgm": "audio.bgm_generate", "sfx": "audio.sfx_generate", "library": "audio.library_import"}
TOOL_NAMES = {"bgm": "generate-bgm", "sfx": "generate-sfx", "library": "library-import"}
_OP_TOOLS = {
    "bgm": {"description": "A music bed from a prompt / tags, a target duration and a seed; instrumental unless lyrics are "
                           "given; validated through ffmpeg-skill/probe and placed with one rename",
            "inputs": ["prompt and / or tags (joined into one caption, < 512 characters)", "duration in seconds (engine range)",
                       "seed (drawn when omitted; always recorded)"],
            "argv": ["generate-bgm", "--prompt <text>", "[--tags <a,b,...>]", "--duration <seconds>", "-o <out.wav>",
                     "[--seed <int>]", "[--engine acestep]", "[--lyrics <file>]", f"[--device {'|'.join(DEVICES)}]",
                     "[--timeout 3600]", "[--overwrite]", "[--ffmpeg-skill <dir>]", "[--offline]", "--json"],
            "schema": "sound-design/generate-bgm@1",
            "duration_tolerance": "max(0.1 s, 1 % of the requested duration), measured by ffmpeg-skill/probe",
            "language_model": "off (Qwen3 base-model licences unverified; ADR-004)"},
    "sfx": {"description": "A sound effect from a prompt / tags, a duration in whole seconds and a seed (EzAudio, mono "
                           "24 kHz); validated through ffmpeg-skill/probe and placed with one rename",
            "inputs": ["prompt and / or tags (joined, < 512 characters; the text encoder reads 100 tokens)",
                       "duration in whole seconds (engine range; EzAudio 1-10)", "seed (drawn when omitted; always recorded)"],
            "argv": ["generate-sfx", "--prompt <text>", "[--tags <a,b,...>]", "--duration <seconds>", "-o <out.wav>",
                     "[--seed <int>]", "[--engine ezaudio]", f"[--device {'|'.join(ezaudio.DEVICES)}]", "[--timeout 600]",
                     "[--overwrite]", "[--ffmpeg-skill <dir>]", "[--offline]", "--json"],
            "schema": "sound-design/generate-sfx@1",
            "duration_tolerance": "max(0.1 s, 1 % of the requested duration), measured by ffmpeg-skill/probe",
            "output_warning": ezaudio.OUTPUT_WARNING},
    "library": {"description": "Import a user-supplied BGM / SFX file into the asset library; refuses a missing licence",
                "inputs": ["audio file", "licence (required)", "source / author"],
                "argv": ["library-import", "<file>", "--licence <text|SPDX>", "[--source <url|text>]", "[--kind bgm|sfx]", "--json"]},
}


def tools() -> list:
    out = []
    for op, spec in _OP_TOOLS.items():
        out.append({"tool_id": f"{SKILL_ID}/{TOOL_NAMES[op]}", "role": "generation" if op != "library" else "ingest",
                    "writes_media": True, "mutates_input": False, "deterministic": False, "implemented": IMPLEMENTED_OPS[op],
                    "roadmap_step": OP_STEP[op], "engine": DEFAULT_ENGINE.get(op), "engines": engines_for(op), **spec})
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
        "handoff": {"bgm": "audio-production-skill MIX places the generated WAV (with its provenance) under the programme",
                    "sfx": "audio-production-skill MIX places the generated effect (with its provenance) in the programme"},
        "tools": tools(),
        "engines": {"registered": {eid: get_engine(eid).spec.to_dict() for eid in engine_ids()},
                    "default": dict(DEFAULT_ENGINE),
                    "selection_adr": "docs/decisions.md ADR-001 (BGM), ADR-005 (SFX)",
                    "optional": True,
                    "statuses": ["OK", "MISSING", "INCOMPATIBLE"]},
        "device": {"reported_by": "doctor checks.device", "detected_by": ["nvidia-smi --query-gpu", "platform + sysctl hw.memsize (Apple Silicon)",
                                                                          "rocm-smi on PATH (noticed, not measured)"],
                   "statuses": ["OK", "CPU_ONLY", "UNKNOWN"], "acestep_tiers": [t for _, t in ACESTEP_TIERS]},
        "policy": {"output_use": "each engine's output-use terms and training-data claim are published in its EngineSpec "
                                 "and copied into the provenance of every file; an engine's output_warning is copied into "
                                 "every result's warnings",
                   "credit": "an engine's credit (credit_required) is carried into the provenance and must appear in the "
                             "published video's description or credits",
                   "watermark": "an engine's watermark is never removed or bypassed; it is recorded in the provenance",
                   "library_licence": "a library import without a licence is refused (roadmap step 4)"},
        "provenance": {"fields": ["skill", "op", *[f"engine.{f}" for f in ENGINE_FIELDS], "prompt.text", "prompt.sha256", "seed",
                                  "duration.requested", "duration.measured", "duration.tolerance", "output.sha256", "parameters",
                                  "ffmpeg_skill.version", "network"],
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
