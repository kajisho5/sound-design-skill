"""Tempo-targeted GENERATE_BGM (roadmap step 8, ADR-012): the length a requested tempo and time signature allow (a
whole number of bars, so the bed ends on a bar line) and the beat grid written beside the audio as data. Pure: no
process, no file system except the one writer.

ACE-Step 1.5 takes the tempo and the time signature as GenerationParams.bpm (an integer, 30-300) and
GenerationParams.timesignature ("2", "3", "4" or "6" for 2/4, 3/4, 4/4, 6/8), and passes both to its DiT with the LM
off (acestep/inference.py at ca1e85fe). Whether the real engine keeps that tempo, and where its first beat falls, has
not been measured (no engine here): the grid is the requested one, marked `measured: false`."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

from . import SKILL_ID, __version__
from .errors import SoundDesignError

GRID_SCHEMA = f"{SKILL_ID}/beat-grid@1"
GRID_SUFFIX = ".beats.json"
BPM_RANGE = (30, 300)                          # GenerationParams docstring: "30 ~ 300"
TIME_SIGNATURES = (2, 3, 4, 6)                 # GenerationParams docstring: 2 for 2/4, 3 for 3/4, 4 for 4/4, 6 for 6/8
DEFAULT_TIME_SIGNATURE = 4
GRID_NOTE = ("the requested grid: ACE-Step was given this tempo and time signature (GenerationParams.bpm / "
             "timesignature) and the length of a whole number of bars; the first beat is assumed at 0 s; the real "
             "engine's adherence is not measured (ADR-012)")


def signature_text(ts: int) -> str:
    return "6/8" if ts == 6 else f"{ts}/4"


def check_tempo(bpm: Any, time_signature: Any) -> Tuple[Optional[int], Optional[int]]:
    """(bpm, beats per bar), or (None, None) when no tempo was asked for. Pure checks."""
    if bpm is None:
        if time_signature is not None:
            raise SoundDesignError("INVALID_ARGUMENT", "--time-signature needs --bpm", {"time_signature": time_signature})
        return None, None
    if isinstance(bpm, bool) or not isinstance(bpm, int) or not BPM_RANGE[0] <= bpm <= BPM_RANGE[1]:
        raise SoundDesignError("INVALID_ARGUMENT", f"--bpm must be a whole number from {BPM_RANGE[0]} to {BPM_RANGE[1]} "
                               "(the engine takes an integer tempo)", {"bpm": bpm})
    ts = DEFAULT_TIME_SIGNATURE if time_signature is None else time_signature
    if isinstance(ts, bool) or ts not in TIME_SIGNATURES:
        raise SoundDesignError("INVALID_ARGUMENT", "--time-signature must be one of " + ", ".join(map(str, TIME_SIGNATURES))
                               + " (2/4, 3/4, 4/4, 6/8)", {"time_signature": time_signature})
    return bpm, ts


def plan_tempo(duration: float, bpm: int, beats_per_bar: int, lo: float, hi: float) -> Dict[str, Any]:
    """The bed's length at `bpm`: the whole number of bars nearest `duration` (a half bar rounds up), kept inside the
    engine's range [lo, hi]. Pure."""
    beat = 60.0 / bpm
    bar = beat * beats_per_bar
    bars = max(1, int(math.floor(duration / bar + 0.5)))
    while bars * bar < lo - 1e-9:
        bars += 1
    while bars * bar > hi + 1e-9 and bars > 1:
        bars -= 1
    length = round(bars * bar, 6)
    if not lo - 1e-9 <= length <= hi + 1e-9:
        raise SoundDesignError("INVALID_ARGUMENT", f"no whole number of {signature_text(beats_per_bar)} bars at {bpm} BPM fits "
                               f"the engine's {lo:g}-{hi:g} s", {"bpm": bpm, "bar": bar})
    return {"bpm": bpm, "time_signature": signature_text(beats_per_bar), "beats_per_bar": beats_per_bar,
            "beat_s": round(beat, 9), "bar_s": round(bar, 9), "bars": bars, "beats": bars * beats_per_bar,
            "duration": length, "requested_duration": float(duration),
            "adjusted": abs(length - float(duration)) > 1e-6}


def beat_grid(plan: Dict[str, Any], audio: Dict[str, Any]) -> Dict[str, Any]:
    """The grid document written beside the WAV: every beat with its bar and count, the downbeats and the end."""
    beat, n, per = plan["beat_s"], plan["beats"], plan["beats_per_bar"]
    beats = [{"index": i, "time": round(i * beat, 6), "bar": i // per + 1, "beat": i % per + 1} for i in range(n)]
    return {
        "schema": GRID_SCHEMA,
        "skill": {"id": SKILL_ID, "version": __version__},
        "audio": dict(audio),
        "bpm": plan["bpm"], "time_signature": plan["time_signature"], "beats_per_bar": per,
        "beat_s": beat, "bar_s": plan["bar_s"], "bars": plan["bars"], "duration": plan["duration"],
        "first_beat": 0.0, "ends_on_bar_line": True,
        "beats": beats,
        "downbeats": [b["time"] for b in beats if b["beat"] == 1],
        "bar_lines": [round(k * plan["bar_s"], 6) for k in range(plan["bars"] + 1)],
        "source": "requested", "measured": False, "note": GRID_NOTE,
    }


def grid_path(wav_path: str) -> str:
    root, _ = os.path.splitext(wav_path)
    return root + GRID_SUFFIX


def write_grid(path: str, doc: Dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return path
