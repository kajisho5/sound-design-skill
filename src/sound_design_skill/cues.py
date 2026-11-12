"""SFX_CUES (roadmap step 9, ADR-013): a cue sheet of sound effects (pop, whoosh, ding, ...) for audio-production-skill
MIX to place, at the cut points and emphasis times the caller gives. This skill never decides where an emphasis is or
adds a cue of its own: every cue comes from one caller event, and an event is only ever dropped (too close to the one
before it, or, when asked, to the last play of the same sound file), never moved. Nothing is mixed or rendered here;
no engine runs.

  sfx-cues  events (--cuts / --emphasis times, or an --events JSON file) -> one effect per event kind or per event,
            from the library (the deterministic library-pick choice among kind sfx files tagged with the effect name,
            declared licence required) or from a WAV the caller names (--file name=path: a generate-sfx or stinger
            result, its provenance travels with it). Each effect is measured once (ffmpeg-skill/probe, its samples):
            its anchor is its peak (whoosh, rise, ...) or its onset (pop, ding, hit, ...), so that start = time - anchor
            puts the anchor on the event; its gain brings its sample peak to --level dBFS (or the event's own level).
            Events closer than --min-spacing to the last kept cue are dropped and listed with the reason.
            ADR-015, both opt-in:
            --character EFFECT=WORD[,WORD] (or an event's "character") keeps only the library files for that effect
            whose tags or mood carry every word; nothing is inferred and no file gains a tag.
            --min-reuse-spacing S: an event whose effect would play the same sound file (the same SHA-256) as a cue
            kept less than S seconds earlier takes the next file in that effect's library ranking that no kept cue
            played within S seconds (rank order, never random); with none, or for a named --file, it is dropped.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import SKILL_ID, __version__
from . import pcm as P
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill
from .library import library_dir, licence_warnings, load_manifest, match, measure, rank, scan, screen, word_list
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
MAX_REUSE_SPACING = 86400.0
DEFAULT_TIMEOUT = 600.0
EPS = 1e-9
# Words to describe an effect's character in a library's tags or mood (library-import --tags / --mood), so that a
# caller and a library use the same ones. A suggestion, not a closed list: --character matches any word the files
# carry, and never a word they do not.
CHARACTER_VOCABULARY = ("bright", "dark", "warm", "metallic", "soft", "hard", "punchy", "subtle", "short", "long",
                        "high", "low", "rising", "falling", "airy", "playful")
REUSE_RULE = ("off unless min_reuse_spacing is set; then an event whose effect would play the same sound file (the same "
              "SHA-256) as a cue kept less than min_reuse_spacing seconds earlier takes the first file in that effect's "
              "library ranking that no kept cue played within that time (rank order, never random: the first-ranked "
              "file whenever it is free), recorded under substitutions; with no such file, or for a named --file, the "
              "event is dropped with the reason, never moved; the min_spacing check against the last kept cue comes "
              "first, and a dropped event is not a cue for either rule")
CHARACTER_RULE = ("a character word (per effect: --character pop=bright; per event: character, which replaces the "
                  "effect's) keeps only the library files for that effect whose tags or mood carry every word "
                  "(NFKC, case-insensitive); nothing is inferred and no file gains a tag; no matching file is "
                  "INPUT_NOT_FOUND; a named --file has no tags, so a character for it is INVALID_ARGUMENT")


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
    min_reuse_spacing: Optional[float] = None                  # seconds; None = off (ADR-015)
    characters: Dict[str, Any] = field(default_factory=dict)   # effect name -> character word(s) (--character pop=bright)


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


def character_words(v: Any, what: str) -> List[str]:
    """Character words, compared like tags and moods (NFKC, case-insensitive), without repeats and sorted: every one
    is required, so their order says nothing."""
    return sorted(word_list(v, what))


def read_events(req: CuesRequest) -> List[Dict[str, Any]]:
    """The caller's events, each {time, kind, effect?, level?, align?, character?, label?, source_index}, in the
    caller's order."""
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
        if r.get("character") is not None:
            ev["character"] = character_words(r["character"], f"event {i} character")
        if isinstance(r.get("label"), str):
            ev["label"] = r["label"]
        out.append(ev)
    return out


def _played(blocked: List[Dict[str, Any]], now: float) -> str:
    return ", ".join(f"{b['file']} at {b['played_at']:.3f}s ({now - b['played_at']:.3f}s earlier)" for b in blocked)


Candidates = Callable[[Dict[str, Any]], Tuple[List[Tuple[str, str]], bool]]


def space(events: List[Dict[str, Any]], min_spacing: float, min_reuse_spacing: Optional[float] = None,
          candidates: Optional[Candidates] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Events in time order (ties: the caller's order), each kept or dropped, never moved. Pure.
      - one closer than `min_spacing` to the last kept event is dropped (`by: min_spacing`);
      - with `min_reuse_spacing` (ADR-015), `candidates(event)` gives the sound files the event's effect can use, in
        rank order, as (identity, name) pairs, and whether they are a named file the caller chose (never replaced).
        The event takes the first one that no kept event played less than `min_reuse_spacing` earlier (`pick`, its
        0-based rank; `passed_over`, the files above it that were too recent). With none free it is dropped
        (`by: min_reuse_spacing`).
    A dropped event is not a cue: neither rule measures from it. Returns (kept, dropped)."""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    last_played: Dict[str, float] = {}                 # file identity -> time of the last kept event that played it
    for ev in sorted(events, key=lambda e: (e["time"], e["source_index"])):
        if kept and ev["time"] - kept[-1]["time"] < min_spacing - EPS:
            dropped.append(dict(ev, reason=f"{ev['time'] - kept[-1]['time']:.3f}s after the cue at {kept[-1]['time']:.3f}s "
                                           f"(minimum spacing {min_spacing:g}s)", by="min_spacing"))
            continue
        if min_reuse_spacing is None or candidates is None:
            kept.append(ev)
            continue
        options, fixed = candidates(ev)
        blocked: List[Dict[str, Any]] = []
        for pick, (ident, name) in enumerate(options):
            prev = last_played.get(ident)
            if prev is not None and ev["time"] - prev < min_reuse_spacing - EPS:
                blocked.append({"file": name, "played_at": prev})
                continue
            last_played[ident] = ev["time"]
            kept.append(dict(ev, pick=pick, passed_over=blocked))
            break
        else:
            limit = f"minimum reuse spacing {min_reuse_spacing:g}s"
            if fixed:
                why = f"the named file {_played(blocked, ev['time'])}; a named --file is never replaced ({limit})"
            elif len(blocked) == 1:
                why = f"{_played(blocked, ev['time'])}, the only eligible file for {ev.get('effect')} ({limit})"
            else:
                why = (f"every eligible file for {ev.get('effect')} played less than {min_reuse_spacing:g}s earlier: "
                       f"{_played(blocked, ev['time'])} ({limit})")
            dropped.append(dict(ev, reason=why, by="min_reuse_spacing"))
    return kept, dropped


def anchors(pcm: P.Pcm) -> Dict[str, Any]:
    """Where an effect's peak and onset are, and its sample peak."""
    pk = P.peak(pcm)
    top = max((abs(v) for v in pcm.mono), default=0.0)
    onset = next((i for i, v in enumerate(pcm.mono) if abs(v) >= ONSET_FRACTION * top), 0) if top > 0 else 0
    return {"peak": pk["time"], "onset": round(onset / pcm.rate, 6), "sample_peak_dbfs": pk["sample_peak_dbfs"],
            "duration": round(pcm.duration, 6)}


def _library_effect(skill: FfmpegSkill, files: Dict[str, str], entries: Dict[str, Dict[str, Any]], effect: str,
                    character: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
    """library-pick's deterministic ranking of the kind sfx files tagged `effect` (declared licence, readable) whose
    tags or mood carry every `character` word. The first row is library-pick's choice."""
    rows, _ = screen(files, entries, "sfx", [effect], [])
    if character:
        rows = [r for r in rows if set(match(r["entry"], list(character), list(character))) == set(character)]
    measured = []
    for r in rows:
        try:
            r["measured"] = measure(skill, r["path"])
        except SoundDesignError:
            continue
        r["duration"], r["sha256"] = r["measured"]["duration"], sha256_file(r["path"])
        measured.append(r)
    ranked, _ = rank(measured, None)
    return ranked


Selector = Tuple[str, Tuple[str, ...]]                 # (effect, character words)


def _selector(ev: Dict[str, Any]) -> Selector:
    return ev["effect"], tuple(ev.get("character") or ())


class _Sources:
    """The sound files each selector (an effect and its character words) can use, in rank order, each file probed
    once and measured from its samples only when a cue uses it."""

    def __init__(self, skill: FfmpegSkill, library: Optional[str], files: Dict[str, str], run_dir: str):
        self.skill, self.library, self.files, self.run_dir = skill, library, files, run_dir
        self.lib_files, self.entries = (scan(library), load_manifest(library)[0]) if library else ({}, {})
        self.ranked: Dict[Selector, List[Dict[str, Any]]] = {}
        self.measured: Dict[str, Dict[str, Any]] = {}
        self.warnings: List[str] = []

    def options(self, sel: Selector) -> List[Dict[str, Any]]:
        if sel not in self.ranked:
            self.ranked[sel] = self._resolve(*sel)
        return self.ranked[sel]

    def candidates(self, ev: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], bool]:
        rows = self.options(_selector(ev))
        return [(r["sha256"], r["file"]) for r in rows], rows[0]["from"] == "file"

    def _resolve(self, effect: str, character: Tuple[str, ...]) -> List[Dict[str, Any]]:
        if effect in self.files:
            path = os.path.abspath(self.files[effect])
            return [{"from": "file", "path": path, "file": os.path.basename(path), "sha256": sha256_file(path),
                     "measured": measure(self.skill, path), "licence": None, "rank": None}]
        if self.library is None:
            raise SoundDesignError("INPUT_NOT_FOUND", f"no source for effect(s) {effect}: give --library or --file "
                                   "<name>=<wav>", {"effects": [effect]})
        rows = _library_effect(self.skill, self.lib_files, self.entries, effect, character)
        if not rows:
            carries = f" and carries {', '.join(character)} in its tags or mood" if character else ""
            raise SoundDesignError("INPUT_NOT_FOUND", f"no library file of kind sfx is tagged {effect!r}{carries} with a "
                                   "declared licence", dict({"effect": effect, "library": self.library},
                                                            **({"character": list(character)} if character else {})))
        return [{"from": "library", "path": r["path"], "file": r["key"], "sha256": r["sha256"], "measured": r["measured"],
                 "licence": r["licence"], "rank": n} for n, r in enumerate(rows, 1)]

    def source(self, sel: Selector, pick: int) -> Dict[str, Any]:
        row = self.options(sel)[pick]
        if row["from"] == "file":
            src = {"from": "file", "file": row["file"], "sha256": row["sha256"], "measured": row["measured"],
                   "licence": None, "note": "a WAV the caller named (e.g. a generate-sfx or stinger result): its own "
                                            "provenance carries its engine and licence"}
        else:
            src = {"from": "library", "file": row["file"], "sha256": row["sha256"], "measured": row["measured"],
                   "licence": row["licence"]}
            self.warnings += licence_warnings(row["file"], row["licence"])
        if row["path"] not in self.measured:
            wav = os.path.join(self.run_dir, f"{len(self.measured)}.wav")
            self.skill.cut(row["path"], wav, duration=row["measured"]["duration"])
            self.measured[row["path"]] = anchors(P.read_wav(wav))
        src["anchors"] = self.measured[row["path"]]
        src.update(effect=sel[0], character=list(sel[1]), rank=row["rank"])
        return src


def _source_keys(used: List[Tuple[str, Tuple[str, ...], int]]) -> Dict[Tuple[str, Tuple[str, ...], int], str]:
    """The key of each used source in the sheet's `sources`: the effect name for an effect's first-ranked file with no
    character words (as before ADR-015), `effect[word,...]` with them, `@<rank>` for a file the reuse rule chose
    lower in the ranking."""
    out: Dict[Tuple[str, Tuple[str, ...], int], str] = {}
    taken = set()
    for use in sorted(used, key=lambda u: (bool(u[1]) or u[2] > 0, u)):     # plain effect names first: never renamed
        effect, character, pick = use
        key = base = effect + (f"[{','.join(character)}]" if character else "") + (f"@{pick + 1}" if pick else "")
        n = 2
        while key in taken:
            key, n = f"{base}#{n}", n + 1
        taken.add(key)
        out[use] = key
    return out


def sfx_cues(req: CuesRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    events = read_events(req)
    spacing = _num(req.min_spacing, "--min-spacing", 0.0, 60.0)
    reuse = None if req.min_reuse_spacing is None else _num(req.min_reuse_spacing, "--min-reuse-spacing", 0.0,
                                                            MAX_REUSE_SPACING)
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
    per_effect = {_word(k, "--character effect"): character_words(v, f"--character {k}")
                  for k, v in (req.characters or {}).items()}
    for name in per_effect:
        if name in files and per_effect[name]:
            raise SoundDesignError("INVALID_ARGUMENT", f"--character {name}: {name} is a named --file, which has no tags "
                                   "or mood to match (a character filters library files)", {"effect": name})
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
        words = ev["character"] if "character" in ev else per_effect.get(ev["effect"], [])
        ev.pop("character", None)
        if words:
            if ev["effect"] in files:
                raise SoundDesignError("INVALID_ARGUMENT", f"event {ev['source_index']}: {ev['effect']} is a named --file, "
                                       "which has no tags or mood to match its character", {"event": ev["source_index"]})
            ev["character"] = words
    warnings: List[str] = [f"--character {name}: no event uses the effect {name}" for name in per_effect
                           if not any(ev["effect"] == name for ev in events)]
    library = library_dir(req.library_dir) if req.library_dir is not None else None
    kept, dropped = space(events, spacing)
    missing = sorted({ev["effect"] for ev in kept if ev["effect"] not in files and library is None})
    if missing:
        raise SoundDesignError("INPUT_NOT_FOUND", "no source for effect(s) " + ", ".join(missing) + ": give --library or "
                               "--file <name>=<wav>", {"effects": missing})
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    run_dir = tempfile.mkdtemp(prefix="sound-design-cues-")
    try:
        pool = _Sources(skill, library, files, run_dir)
        if reuse is not None:
            kept, dropped = space(events, spacing, reuse, pool.candidates)
        used = sorted({_selector(ev) + (ev.get("pick", 0),) for ev in kept})
        keys = _source_keys(used)
        sources = {keys[u]: pool.source(u[:2], u[2]) for u in used}
        warnings += pool.warnings
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    cues, substitutions = [], []
    for n, ev in enumerate(kept):
        pick = ev.get("pick", 0)
        key = keys[_selector(ev) + (pick,)]
        src = sources[key]
        align = ev.get("align") or default_align(ev["effect"])
        anchor = src["anchors"][align]
        start = round(ev["time"] - anchor, 6)
        target = ev.get("level", level)
        cue = {"index": n, "time": ev["time"], "kind": ev["kind"], "effect": ev["effect"], "align": align,
               "anchor_in_file": anchor, "start": max(0.0, start), "file_offset": round(max(0.0, -start), 6),
               "duration": round(src["anchors"]["duration"] - max(0.0, -start), 6), "level_dbfs": target,
               "gain_db": round(target - src["anchors"]["sample_peak_dbfs"], 2), "file": src["file"],
               "from": src["from"], "source": key, "source_index": ev["source_index"]}
        if "character" in ev:
            cue["character"] = ev["character"]
        if "label" in ev:
            cue["label"] = ev["label"]
        if pick:
            first = ev["passed_over"][0]["file"]
            reason = f"{_played(ev['passed_over'], ev['time'])} (minimum reuse spacing {reuse:g}s)"
            cue["substituted"] = {"for": first, "rank": pick + 1, "reason": reason}
            substitutions.append({"cue": n, "time": ev["time"], "source_index": ev["source_index"], "effect": ev["effect"],
                                  "file": src["file"], "instead_of": first, "rank": pick + 1, "reason": reason})
        if start < 0:
            warnings.append(f"cue at {ev['time']:.3f}s: the {ev['effect']} would start before 0 s; it starts at 0 with the "
                            f"first {-start:.3f}s of the file skipped (file_offset)")
        cues.append(cue)
    gaps = [round(b["time"] - a["time"], 6) for a, b in zip(cues, cues[1:])]
    sheet = {
        "schema": CUES_SCHEMA, "skill": {"id": SKILL_ID, "version": __version__},
        "min_spacing": spacing, "min_reuse_spacing": reuse, "level_dbfs": level,
        "cues": cues, "dropped": dropped, "substitutions": substitutions,
        "sources": sources,
        "rule": "one cue per caller event, at the caller's time; an event closer than min_spacing to the last kept cue "
                "is dropped, never moved; start = time - anchor_in_file (anchor: the effect's peak or onset); gain_db "
                "brings the effect's sample peak to level_dbfs; nothing is placed or mixed here",
        "reuse_rule": REUSE_RULE, "character_rule": CHARACTER_RULE,
        "summary": {"events": len(events), "cues": len(cues), "dropped": len(dropped),
                    "smallest_gap": min(gaps) if gaps else None},
        "handoff": {"to": "audio-production-skill MIX", "note": "place each cue's file at start (skipping file_offset) "
                                                                "with gain_db; the times are the caller's; each cue's "
                                                                "file and licence are under sources[cue.source]"},
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
