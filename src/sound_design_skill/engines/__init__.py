"""Engine registry. Only engines listed here can be selected or run by the worker; test engines are never
registered. ACE-Step 1.5 is the BGM engine ADR-001 verified. No SFX engine is registered yet: its ADR is roadmap
step 3 (candidates there are unverified until that ADR checks code and weights licences)."""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from ..errors import SoundDesignError
from .acestep import AceStepEngine
from .base import Engine

ENGINES: Dict[str, Type[Engine]] = {cls.spec.id: cls for cls in (AceStepEngine,)}
DEFAULT_ENGINE: Dict[str, Optional[str]] = {"bgm": AceStepEngine.spec.id, "sfx": None}


def engine_ids() -> List[str]:
    return sorted(ENGINES)


def get_engine(engine_id: str) -> Engine:
    cls = ENGINES.get(engine_id)
    if cls is None:
        raise SoundDesignError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"engines": engine_ids()})
    return cls()


def engines_for(op: str) -> List[str]:
    return [eid for eid in engine_ids() if op in ENGINES[eid].spec.ops]
