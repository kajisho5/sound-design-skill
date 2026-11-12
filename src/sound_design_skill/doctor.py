"""`sound-design doctor`: what is installed, what is missing and how to get it, and which compute device and how much
GPU memory this machine has. Never downloads, never imports an engine or torch (the engine is looked up by
distribution metadata and weights files; the device by nvidia-smi / sysctl), so it is fast and safe to run anywhere.
Media capabilities are detected through ffmpeg-skill (its contract and its own doctor).

status: ok (ffmpeg-skill usable, every engine OK) | degraded (an engine MISSING / INCOMPATIBLE) | fail (Python or
ffmpeg-skill not usable). The device never fails doctor: ACE-Step documents CPU support, so a CPU-only machine is a
warning. Each op reports whether it is implemented and whether its engine is ready: BGM on ACE-Step 1.5 (ADR-004),
SFX on EzAudio (ADR-005, ADR-006), library mode with no engine (ADR-008), and whether its engine needs the user's
confirmation flag first (EzAudio does, ADR-007). BGM also lists every selectable model (ADR-016) with its files, the GPU
memory it requires against what was measured, whether it is ready and how to get it; a model other than the default
being absent is not a problem (it is opt-in), but one on disk that this GPU cannot run is a warning."""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Optional

from . import SKILL_ID, __version__
from .device import detect
from .engines import DEFAULT_ENGINE, confirmation_for, engine_ids, engines_for, get_engine
from .engines.base import OK, model_readiness
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill
from . import stock

DOCTOR_SCHEMA = f"{SKILL_ID}/doctor@1"
# op -> implemented? / roadmap step; GENERATE_BGM since step 2 (ADR-004), GENERATE_SFX since step 3 (ADR-006), library
# mode since step 4 (ADR-008), loop-bed and stinger since step 7 (ADR-011)
IMPLEMENTED_OPS: Dict[str, bool] = {"bgm": True, "sfx": True, "library": True, "loop": True, "stinger": True}
OP_STEP = {"bgm": 2, "sfx": 3, "library": 4, "loop": 7, "stinger": 7}
NO_ENGINE_NOTE = {"library": "no engine (never ACE-Step or EzAudio): the user's own BGM / SFX files with a declared "
                                "licence, chosen, fitted and measured through ffmpeg-skill (ADR-008)",
                  "loop": "no engine: a WAV the caller has, looped through ffmpeg-skill cut / join, its wrap measured "
                          "(ADR-011)",
                  "stinger": "no engine: a WAV the caller has, cut / faded / padded through ffmpeg-skill so its peak "
                             "lands at a stated time (ADR-011)"}


def engines_report(offline: bool = False) -> Dict[str, Any]:
    return {eid: get_engine(eid).report(offline=offline) for eid in engine_ids()}


def models_report(op: str, engines: Dict[str, Any], device: Dict[str, Any]) -> Dict[str, Any]:
    """checks.operations.<op>.models (ADR-016): readiness of each model of the op's default engine, keyed by model id."""
    eid = DEFAULT_ENGINE.get(op)
    if not eid:
        return {}
    engine = get_engine(eid)
    out = {}
    for m in engine.spec.models:
        r = model_readiness(engine, m, device, engines[eid]["status"])
        r["flag"] = None if m.default else f"--model {m.id}"
        r["refusal"] = None if r["refusal"] is None else {k: r["refusal"][k] for k in ("code", "message")}
        out[m.id] = r
    return out


def doctor(ffmpeg_skill_dir: Optional[str] = None, offline: bool = False) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    problems: List[str] = []
    warnings: List[str] = []
    py_ok = sys.version_info >= (3, 9)
    checks["python"] = {"status": "OK" if py_ok else "MISSING", "version": platform.python_version(), "platform": platform.system(),
                        "install": None if py_ok else "Python >= 3.9"}
    if not py_ok:
        problems.append("python: needs >= 3.9")

    try:
        skill = FfmpegSkill.locate(ffmpeg_skill_dir)
        info = skill.info()
        checks["ffmpeg_skill"] = dict(info.to_dict(), status="OK" if info.supported else "INCOMPATIBLE")
        problems += ["ffmpeg-skill: " + p for p in info.problems]
        warnings += ["ffmpeg-skill: " + w for w in info.warnings]
        if info.supported:
            ffdoc = skill.doctor()
            for tool in ("ffmpeg", "ffprobe"):
                found = ffdoc.get(tool)
                checks[tool] = {"status": "OK" if found else "MISSING", "version": found or None, "detected_by": "ffmpeg-skill doctor"}
                if not found:
                    problems.append(f"{tool}: not detected by ffmpeg-skill doctor")
        else:
            for tool in ("ffmpeg", "ffprobe"):
                checks[tool] = {"status": "UNKNOWN", "detail": "not checked: ffmpeg-skill not usable"}
    except SoundDesignError as e:
        checks["ffmpeg_skill"] = {"status": "MISSING", "detail": e.message, "tried": e.details.get("tried"),
                                  "install": "git clone https://github.com/kajisho5/ffmpeg-skill ~/.claude/skills/ffmpeg-skill "
                                             "(or pass --ffmpeg-skill / set SOUND_DESIGN_FFMPEG_SKILL_DIR)"}
        for tool in ("ffmpeg", "ffprobe"):
            checks[tool] = {"status": "UNKNOWN", "detail": "not checked: ffmpeg-skill missing"}
        problems.append("ffmpeg-skill: " + e.message)

    device = detect()
    checks["device"] = device
    if device["status"] == "CPU_ONLY":
        warnings.append("device: no GPU found; ACE-Step can run on CPU but slowly (speed unverified)")
    elif device.get("meets_acestep_minimum") is False:
        warnings.append(f"device: {device['vram_total_mb']} MiB GPU memory is below the 4 GB ACE-Step 1.5 documents")

    engines = engines_report(offline=offline)
    checks["engines"] = engines
    media_ok = checks["ffmpeg_skill"]["status"] == "OK" and all(checks[t]["status"] == "OK" for t in ("ffmpeg", "ffprobe"))
    ops: Dict[str, Any] = {}
    for op in IMPLEMENTED_OPS:
        eid = DEFAULT_ENGINE.get(op)
        engine_status = engines[eid]["status"] if eid else None
        ops[op] = {"implemented": IMPLEMENTED_OPS[op], "roadmap_step": OP_STEP[op], "engine": eid, "engine_status": engine_status,
                   "alternatives": [e for e in engines_for(op) if e != eid],
                   "ready": bool(IMPLEMENTED_OPS[op] and media_ok and (eid is None or engine_status == OK)),
                   "confirmation_required": confirmation_for(op) is not None, "confirmation": confirmation_for(op)}
        if op in NO_ENGINE_NOTE:
            ops[op]["note"] = NO_ENGINE_NOTE[op]
        if eid and get_engine(eid).spec.models:
            ops[op]["default_model"] = get_engine(eid).spec.default_model
            ops[op]["models"] = models_report(op, engines, device)
            for mid, m in ops[op]["models"].items():
                if (not m["default"] and m["status"] == OK and m["refusal"]
                        and m["refusal"]["code"] == "INSUFFICIENT_VRAM"):
                    warnings.append(f"model {mid}: on disk, but {op} with {m['flag']} will be refused: {m['vram']['detail']}")
                elif not m["default"] and m["status"] == OK and m["vram"]["status"] == "UNKNOWN":
                    warnings.append(f"model {mid}: needs >= {m['min_vram_gb']:g} GB GPU memory; {m['vram']['detail']}")
    checks["operations"] = ops
    checks["stock_sources"] = stock.report()       # step 10 (ADR-014): reported only; never called, never a problem
    missing_engines = sorted(eid for eid, e in engines.items() if e["status"] != OK)
    status = "fail" if problems else ("degraded" if missing_engines else "ok")
    return {
        "schema": DOCTOR_SCHEMA,
        "ok": status != "fail",
        "status": status,
        "skill": {"id": SKILL_ID, "version": __version__},
        "offline": bool(offline),
        "checks": checks,
        "unavailable_engines": missing_engines,
        "problems": problems + [f"engine {eid}: {engines[eid]['status']} ({engines[eid].get('detail')})" for eid in missing_engines],
        "warnings": warnings,
        "network": "never used; nothing is downloaded (engine weights are a one-time manual step, see install)",
    }
