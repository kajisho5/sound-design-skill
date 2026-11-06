"""SFX_CUES (roadmap step 9, ADR-013): a cue sheet of sound effects (pop, whoosh, ding, ...) for audio-production-skill
MIX to place, at the cut points and emphasis times the caller gives. This skill never decides where an emphasis is or
adds a cue of its own: every cue comes from one caller event, and an event is only ever dropped (too close to the one
before it), never moved. Nothing is mixed or rendered here; no engine runs.

  sfx-cues  events (--cuts / --emphasis times, or an --events JSON file) -> one effect per event kind or per event,
            from the library (the deterministic library-pick choice among kind sfx files tagged with the effect name,
            declared licence required) or from a WAV the caller names (--file name=path: a generate-sfx or stinger
            result, its provenance travels with it). Each effect is measured once (ffmpeg-skill/probe, its samples):
            its anchor is its peak (whoosh, rise, ...) or its onset (pop, ding, hit, ...), so that start = time - anchor
            puts the anchor on the event; its gain brings its sample peak to --level dBFS (or the event's own level).
            Events closer than --min-spacing to the last kept cue are dropped and listed with the reason.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import SKILL_ID, __version__
from . import pcm as P
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill
from .library import library_dir, licence_warnings, load_manifest, measure, rank, scan, screen
from .ops import usable_skill
from .provenance import sha256_file

CUES_SCHEMA = f"{SKILL_ID}/sfx-cues@1"
EVENTS_SCHEMA = f"{SKILL_ID}/sfx-events@1"
DEFAULT_EFFECTS = {"cut": "whoosh", "emphasis": "pop"}
EVENT_KINDS = ("cut", "emphasis")
PEAK_EFFECTS = frozenset({"whoosh", "swoosh", "swish", "rise", "riser", "swell", "sweep", "reverse"})
ALIGNS = ("peak", "onset")
DEFAULT_SPACING = 0.25
DEFAULT_LEVEL = -12.0
ONSET_FRACTION = 0.1          # onset: the first sample at or above 10 % (-20 dB) of the effect's sample peak
MAX_EVENTS = 10000
DEFAULT_TIMEOUT = 600.0


@dataclass
class CuesRequest:
    cuts: List[float] = field(default_factory=list)
    emphasis: List[float] = field(default_factory=list)
    events_path: Optional[str] = None
    library_dir: Optional[str] = None
    effects: Dict[str, str] = field(default_factory=dict)      # event kind -> effect name (--effect cut=whoosh)
    files: Dict[str, str] = field(default_factory=dict)        # effect name -> WAV path (--file whoosh=w.wav)
    min_spacing: float = DEFAULT_SPACING
    level: float = DEFAULT_LEVEL
    output_path: Optional[str] = None
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def default_align(effect: str) -> str:
    return "peak" if effect.casefold() in PEAK_EFFECTS else "onset"


def _num(v: Any, what: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or not lo <= v <= hi:
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be a number from {lo:g} to {hi:g}", {what: v})
    return float(v)


def _word(v: Any, what: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be a non-empty name", {what: v})
    return v.strip()


def read_events(req: CuesRequest) -> List[Dict[str, Any]]:
    """The caller's events, each {time, kind, effect?, level?, align?, label?, source_index}, in the caller's order."""
    raw: List[Dict[str, Any]] = [{"time": t, "kind": "cut"} for t in req.cuts]
    raw += [{"time": t, "kind": "emphasis"} for t in req.emphasis]
    if req.events_path is not None:
        try:
            with open(req.events_path, encoding="utf-8-sig") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            raise SoundDesignError("INVALID_ARGUMENT", f"--events is not readable UTF-8 JSON: {e}", {"events": req.events_path})
        rows = doc.get("events") if isinstance(doc, dict) else doc
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise SoundDesignError("INVALID_ARGUMENT", "--events needs a list of objects (or {\"events\": [...]})")
        raw += rows
    if not raw:
        raise SoundDesignError("INVALID_ARGUMENT", "give the event times: --cuts, --emphasis and / or --events")
    if len(raw) > MAX_EVENTS:
        raise SoundDesignError("INVALID_ARGUMENT", f"at most {MAX_EVENTS} events", {"events": len(raw)})
    out = []
    for i, r in enumerate(raw):
        kind = r.get("kind", "cut")
        if kind not in EVENT_KINDS:
            raise SoundDesignError("INVALID_ARGUMENT", f"event {i}: kind must be one of {', '.join(EVENT_KINDS)}", {"event": r})
        ev = {"source_index": i, "time": _num(r.get("time"), f"event {i} time", 0.0, 86400.0), "kind": kind}
        if r.get("effect") is not None:
            ev["effect"] = _word(r["effect"], f"event {i} effect")
        if r.get("level") is not None:
            ev["level"] = _num(r["level"], f"event {i} level", -60.0, 0.0)
        if r.get("align") is not None:
            if r["align"] not in ALIGNS:
                raise SoundDesignError("INVALID_ARGUMENT", f"event {i}: align must be peak or onset", {"event": r})
            ev["align"] = r["align"]
        if isinstance(r.get("label"), str):
            ev["label"] = r["label"]
        out.append(ev)
    return out


def space(events: List[Dict[str, Any]], min_spacing: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Events in time order (ties: the caller's order); one closer than `min_spacing` to the last kept event is
    dropped, never moved. Pure. Returns (kept, dropped)."""
    kept, dropped = [], []
    for ev in sorted(events, key=lambda e: (e["time"], e["source_index"])):
        if kept and ev["time"] - kept[-1]["time"] < min_spacing - 1e-9:
            dropped.append(dict(ev, reason=f"{ev['time'] - kept[-1]['time']:.3f}s after the cue at {kept[-1]['time']:.3f}s "
                                           f"(minimum spacing {min_spacing:g}s)"))
        else:
            kept.append(ev)
    return kept, dropped


def anchors(pcm: P.Pcm) -> Dict[str, Any]:
    """Where an effect's peak and onset are, and its sample peak."""
    pk = P.peak(pcm)
    top = max((abs(v) for v in pcm.mono), default=0.0)
    onset = next((i for i, v in enumerate(pcm.mono) if abs(v) >= ONSET_FRACTION * top), 0) if top > 0 else 0
    return {"peak": pk["time"], "onset": round(onset / pcm.rate, 6), "sample_peak_dbfs": pk["sample_peak_dbfs"],
            "duration": round(pcm.duration, 6)}


def _library_effect(skill: FfmpegSkill, library: str, files: Dict[str, str], entries: Dict[str, Dict[str, Any]],
                    effect: str) -> Optional[Dict[str, Any]]:
    """library-pick's deterministic choice among kind sfx files tagged `effect` (declared licence, readable)."""
    rows, _ = screen(files, entries, "sfx", [effect], [])
    measured = []
    for r in rows:
        try:
            r["measured"] = measure(skill, r["path"])
        except SoundDesignError:
            continue
        r["duration"], r["sha256"] = r["measured"]["duration"], sha256_file(r["path"])
        measured.append(r)
    ranked, _ = rank(measured, None)
    return ranked[0] if ranked else None


def sfx_cues(req: CuesRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    events = read_events(req)
    spacing = _num(req.min_spacing, "--min-spacing", 0.0, 60.0)
    level = _num(req.level, "--level", -60.0, 0.0)
    timeout = _num(req.timeout, "--timeout", 0.001, 1e9)
    effects = dict(DEFAULT_EFFECTS)
    for k, v in (req.effects or {}).items():
        if k not in EVENT_KINDS:
            raise SoundDesignError("INVALID_ARGUMENT", f"--effect takes {'|'.join(EVENT_KINDS)}=<name>", {"effect": k})
        effects[k] = _word(v, f"--effect {k}")
    files = {_word(k, "--file name"): v for k, v in (req.files or {}).items()}
    for name, path in files.items():
        if not isinstance(path, str) or not os.path.isfile(path):
            raise SoundDesignError("INPUT_NOT_FOUND", f"--file {name}={path} is not a file", {"file": path})
    dst = None
    if req.output_path is not None:
        dst = os.path.abspath(req.output_path)
        if not dst.lower().endswith(".json"):
            raise SoundDesignError("INVALID_ARGUMENT", "the cue sheet is written as .json", {"output": req.output_path})
        if os.path.lexists(dst) and not req.overwrite:
            raise SoundDesignError("OUTPUT_EXISTS", f"{req.output_path} exists; pass --overwrite to replace it", {"output": dst})
        if not os.path.isdir(os.path.dirname(dst)):
            raise SoundDesignError("INVALID_ARGUMENT", f"the output directory {os.path.dirname(dst)} does not exist",
                                   {"output": dst})
    for ev in events:
        ev.setdefault("effect", effects[ev["kind"]])
    kept, dropped = space(events, spacing)
    needed = sorted({ev["effect"] for ev in kept})
    library = library_dir(req.library_dir) if req.library_dir is not None else None
    missing = [n for n in needed if n not in files and library is None]
    if missing:
        raise SoundDesignError("INPUT_NOT_FOUND", "no source for effect(s) " + ", ".join(missing) + ": give --library or "
                               "--file <name>=<wav>", {"effects": missing})
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    lib_files, entries = (scan(library), load_manifest(library)[0]) if library else ({}, {})
    sources: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    run_dir = tempfile.mkdtemp(prefix="sound-design-cues-")
    try:
        for name in needed:
            if name in files:
                path = os.path.abspath(files[name])
                facts = measure(skill, path)
                src = {"from": "file", "file": os.path.basename(path), "sha256": sha256_file(path), "measured": facts,
                       "licence": None, "note": "a WAV the caller named (e.g. a generate-sfx or stinger result): its own "
                                                "provenance carries its engine and licence"}
            else:
                row = _library_effect(skill, library, lib_files, entries, name)
                if row is None:
                    raise SoundDesignError("INPUT_NOT_FOUND", f"no library file of kind sfx is tagged {name!r} with a "
                                           "declared licence", {"effect": name, "library": library})
                path, facts = row["path"], row["measured"]
                src = {"from": "library", "file": row["key"], "sha256": row["sha256"], "measured": facts,
                       "licence": row["licence"]}
                warnings += licence_warnings(row["key"], row["licence"])
            wav = os.path.join(run_dir, f"{len(sources)}.wav")
            skill.cut(path, wav, duration=facts["duration"])
            src["anchors"] = anchors(P.read_wav(wav))
            sources[name] = src
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    cues = []
    for n, ev in enumerate(kept):
        src = sources[ev["effect"]]
        align = ev.get("align") or default_align(ev["effect"])
        anchor = src["anchors"][align]
        start = round(ev["time"] - anchor, 6)
        target = ev.get("level", level)
        cue = {"index": n, "time": ev["time"], "kind": ev["kind"], "effect": ev["effect"], "align": align,
               "anchor_in_file": anchor, "start": max(0.0, start), "file_offset": round(max(0.0, -start), 6),
               "duration": round(src["anchors"]["duration"] - max(0.0, -start), 6), "level_dbfs": target,
               "gain_db": round(target - src["anchors"]["sample_peak_dbfs"], 2), "file": src["file"],
               "from": src["from"], "source_index": ev["source_index"]}
        if "label" in ev:
            cue["label"] = ev["label"]
        if start < 0:
            warnings.append(f"cue at {ev['time']:.3f}s: the {ev['effect']} would start before 0 s; it starts at 0 with the "
                            f"first {-start:.3f}s of the file skipped (file_offset)")
        cues.append(cue)
    gaps = [round(b["time"] - a["time"], 6) for a, b in zip(cues, cues[1:])]
    sheet = {
        "schema": CUES_SCHEMA, "skill": {"id": SKILL_ID, "version": __version__},
        "min_spacing": spacing, "level_dbfs": level,
        "cues": cues, "dropped": dropped,
        "sources": sources,
        "rule": "one cue per caller event, at the caller's time; an event closer than min_spacing to the last kept cue "
                "is dropped, never moved; start = time - anchor_in_file (anchor: the effect's peak or onset); gain_db "
                "brings the effect's sample peak to level_dbfs; nothing is placed or mixed here",
        "summary": {"events": len(events), "cues": len(cues), "dropped": len(dropped),
                    "smallest_gap": min(gaps) if gaps else None},
        "handoff": {"to": "audio-production-skill MIX", "note": "place each cue's file at start (skipping file_offset) "
                                                                "with gain_db; the times are the caller's"},
    }
    if dst is not None:
        fd, tmp = tempfile.mkstemp(prefix=".sound-design-", suffix=".json", dir=os.path.dirname(dst))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(sheet, fh, indent=2, ensure_ascii=True)
                fh.write("\n")
            if os.path.lexists(dst) and not req.overwrite:
                raise SoundDesignError("OUTPUT_EXISTS", f"{dst} appeared while working", {"output": dst})
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return dict(sheet, ok=True, op="sfx-cues", output={"path": dst}, engine=None,
                ffmpeg_skill={"version": sinfo.version, "contract_version": sinfo.contract_version},
                warnings=sorted(set(warnings), key=warnings.index))
