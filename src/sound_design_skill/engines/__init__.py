"""Engine registry. Only engines listed here can be selected or run by the worker; test engines are never
registered. ACE-Step 1.5 is the BGM engine ADR-001 verified; EzAudio is the SFX engine ADR-005 verified (the other SFX
candidates there are non-commercial, revenue-capped or carry a non-commercial component). An engine whose spec names
a confirmation_flag runs only when the caller passes that flag (ADR-007; EzAudio)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from ..errors import SoundDesignError
from .acestep import AceStepEngine
from .base import Engine
from .ezaudio import EzAudioEngine

ENGINES: Dict[str, Type[Engine]] = {cls.spec.id: cls for cls in (AceStepEngine, EzAudioEngine)}
DEFAULT_ENGINE: Dict[str, Optional[str]] = {"bgm": AceStepEngine.spec.id, "sfx": EzAudioEngine.spec.id}


def engine_ids() -> List[str]:
    return sorted(ENGINES)


def get_engine(engine_id: str) -> Engine:
    cls = ENGINES.get(engine_id)
    if cls is None:
        raise SoundDesignError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"engines": engine_ids()})
    return cls()


def engines_for(op: str) -> List[str]:
    return [eid for eid in engine_ids() if op in ENGINES[eid].spec.ops]


def confirmation_for(op: str, engine_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """What the user must be told before the engine for `op` (the default one when `engine_id` is None) runs, and the
    flag that says they were (ADR-007). None when that engine needs no confirmation, is unknown or does not do `op`
    (select_engine reports those)."""
    eid = engine_id or DEFAULT_ENGINE.get(op)
    cls = ENGINES.get(eid) if eid else None
    if cls is None or op not in cls.spec.ops or not cls.spec.confirmation_required:
        return None
    s = cls.spec
    return {"engine": eid, "flag": s.confirmation_flag, "reason": s.confirmation_reason, "reason_ja": s.confirmation_reason_ja,
            "error_code": "CONFIRMATION_REQUIRED", "approval": "CONFIRM", "adr": "docs/decisions.md ADR-007"}
