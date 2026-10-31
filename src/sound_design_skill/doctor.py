"""`sound-design doctor`: what is installed, what is missing and how to get it, and which compute device and how much
GPU memory this machine has. Never downloads, never imports an engine or torch (the engine is looked up by
distribution metadata and weights files; the device by nvidia-smi / sysctl), so it is fast and safe to run anywhere.
Media capabilities are detected through ffmpeg-skill (its contract and its own doctor).

status: ok (ffmpeg-skill usable, every engine OK) | degraded (an engine MISSING / INCOMPATIBLE) | fail (Python or
ffmpeg-skill not usable). The device never fails doctor: ACE-Step documents CPU support, so a CPU-only machine is a
warning. Each op reports whether it is implemented and whether its engine is ready: BGM on ACE-Step 1.5 (ADR-004),
SFX on EzAudio (ADR-005, ADR-006), and whether its engine needs the user's confirmation flag first (EzAudio does,
ADR-007)."""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Optional

from . import SKILL_ID, __version__
from .device import detect
from .engines import DEFAULT_ENGINE, confirmation_for, engine_ids, engines_for, get_engine
from .engines.base import OK
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill

DOCTOR_SCHEMA = f"{SKILL_ID}/doctor@1"
# op -> implemented? / roadmap step; GENERATE_BGM since step 2 (ADR-004), GENERATE_SFX since step 3 (ADR-006)
IMPLEMENTED_OPS: Dict[str, bool] = {"bgm": True, "sfx": True, "library": False}
OP_STEP = {"bgm": 2, "sfx": 3, "library": 4}
NO_ENGINE_NOTE = {"library": "no engine: imports user-supplied BGM / SFX with a licence field"}


def engines_report(offline: bool = False) -> Dict[str, Any]:
    return {eid: get_engine(eid).report(offline=offline) for eid in engine_ids()}


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
    checks["operations"] = ops
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
