"""Library mode (roadmap step 4, ADR-008): BGM and SFX from the user's own files instead of an engine. No engine runs
here (never ACE-Step or EzAudio, no engine process) and nothing is downloaded: every file comes from a local folder the user
supplies, and every media step is an ffmpeg-skill tool.

The library is a folder of audio files plus one manifest, `sound-library.json`, at its root. The manifest holds what
the files cannot say about themselves, per file (keyed by the library-relative POSIX path, Unicode NFC): the licence as
the user declares it, kind (bgm / sfx), tags, mood, loopable, source, author and credit. Nothing in it is invented by
this skill: a file without an entry, or whose licence is missing, empty or a word such as "unknown", is reported with
`licence: "unknown"` and `status: "unknown"`, and is never used. A declared licence is recorded as declared, with
`verified: false`; this skill does not clear licences.

  library-import  registers one file: refuses a missing licence (LICENCE_REQUIRED) before anything else, probes the
                  file through ffmpeg-skill (one audio stream, no picture), copies it into the library (or registers it
                  in place when it is already there) and writes its manifest entry with the measured facts
  library-index   lists every audio file in the folder with its licence status (declared / unknown), kind, tags and
                  mood, its duration measured by ffmpeg-skill/probe and, with --loudness, its EBU R128 loudness
                  (ffmpeg-skill/loudness --measure-only); read-only
  library-pick    chooses one file deterministically (rules below), fits it to a requested duration (trim, or a
                  crossfaded loop for a file marked loopable) through ffmpeg-skill cut / join, validates the result
                  through ffmpeg-skill/probe, measures its loudness, and places it as a WAV with its provenance

Selection (library-pick), in this order:
  1. eligible   in the manifest with a declared licence; of the requested --kind; matching at least one requested
                tag or mood (case- and width-insensitive, NFKC) when any is given; readable by ffmpeg-skill/probe as
                audio only; and fittable: within the tolerance of --duration, longer (trimmed), or shorter and marked
                loopable within MAX_REPEATS copies
  2. ranked by  more matched tags and moods; then no fit < trim < loop; then the smaller |measured - requested|
                duration; then the library-relative path (code-point order); then the SHA-256
Every excluded file is reported with its reasons. The same library contents and query give the same choice.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import SKILL_ID, __version__
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill
from .ops import check_output, tolerance_for, usable_skill
from .provenance import build_library_provenance, sha256_file

MANIFEST_NAME = "sound-library.json"
MANIFEST_SCHEMA = f"{SKILL_ID}/library@1"
IMPORT_SCHEMA = f"{SKILL_ID}/library-import@1"
INDEX_SCHEMA = f"{SKILL_ID}/library-index@1"
PICK_SCHEMA = f"{SKILL_ID}/library-pick@1"
CAPABILITY = "audio.library_import"
KINDS = ("bgm", "sfx")
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".caf")
# a licence written as one of these says nothing about the rights: it is reported as unknown, never as declared
UNKNOWN_LICENCE_WORDS = frozenset({"unknown", "unspecified", "none", "n/a", "na", "?", "tbd", "todo", "unclear", "-"})
DEFAULT_CROSSFADE = 0.5      # seconds between loop repeats, capped at a quarter of the file
MAX_REPEATS = 64
DEFAULT_TIMEOUT = 600.0
FIT_RANK = {"none": 0, "trim": 1, "loop": 2}
LICENCE_NOTE = ("declared by the user in sound-library.json; this skill records it and does not verify or clear it "
                "(licence clearance is not provided)")
SELECTION_RULE = ("eligible: declared licence, requested kind, >= 1 requested tag / mood matched, audio only, fittable; "
                  "ranked by: matched tags + moods (more first), fit (none < trim < loop), |measured - requested| "
                  "duration, library-relative path, SHA-256")
_NC = re.compile(r"(?<![a-z])nc(?![a-z])|non[- ]?commercial", re.I)
_ND = re.compile(r"(?<![a-z])nd(?![a-z])|no[- ]?deriv", re.I)
_BY = re.compile(r"(?<![a-z])cc[- ]?by(?![a-z])|creative commons attribution", re.I)


# ---------------------------------------------------------------------------------------------------- pure helpers

def norm_key(relpath: str) -> str:
    """The manifest key of a library file: its path relative to the library, '/'-separated, Unicode NFC (macOS file
    systems hand out NFD names; a manifest typed on Windows or Linux is NFC). Empty and "." segments are dropped."""
    parts = unicodedata.normalize("NFC", relpath.replace("\\", "/")).split("/")
    return "/".join(p for p in parts if p not in ("", "."))


def norm_word(text: str) -> str:
    """Tags and moods compare case- and width-insensitively (NFKC + casefold): "Calm", "calm" and "ｃａｌｍ" match."""
    return unicodedata.normalize("NFKC", text).strip().casefold()


def word_list(values: Any, what: str) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be a list of strings", {what: values})
    out: List[str] = []
    for v in values:
        w = norm_word(v)
        if w and w not in out:
            out.append(w)
    return out


def licence_info(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The licence of one library file exactly as declared. `licence` is the declared text, or "unknown" when the
    file has no manifest entry or its licence is missing, not text, empty or a placeholder word; `status` is
    "declared" or "unknown"; `verified` is always false. Nothing is inferred or filled in."""
    raw = entry.get("licence") if isinstance(entry, dict) else None
    text = raw.strip() if isinstance(raw, str) else None
    declared = bool(text) and text.casefold() not in UNKNOWN_LICENCE_WORDS

    def opt(key: str) -> Optional[str]:
        v = entry.get(key) if isinstance(entry, dict) else None
        return v.strip() if isinstance(v, str) and v.strip() else None

    return {"licence": text if declared else "unknown", "status": "declared" if declared else "unknown",
            "declared_as": raw if isinstance(raw, str) else None, "verified": False, "usable": declared,
            "declared_in": MANIFEST_NAME if isinstance(entry, dict) else None,
            "source": opt("source"), "author": opt("author"), "credit": opt("credit"),
            "note": LICENCE_NOTE if declared else "no licence declared for this file: it is reported as unknown and "
                                                  "is not used until its licence is declared (library-import --licence)"}


def licence_warnings(key: str, info: Dict[str, Any], fitted: bool = False) -> List[str]:
    """Plain facts about the declared text that the user should see; never a judgement that the file is cleared."""
    if info["status"] != "declared":
        return [f"{key}: licence unknown; not used"]
    text = info["licence"]
    out = []
    if _NC.search(text):
        out.append(f"{key}: the declared licence {text!r} names a non-commercial term")
    if fitted and _ND.search(text):
        out.append(f"{key}: the declared licence {text!r} names a no-derivatives term; trimming or looping may be an "
                   "adaptation it does not allow")
    if _BY.search(text) and not info.get("credit"):
        out.append(f"{key}: the declared licence {text!r} is a Creative Commons attribution licence and no credit is "
                   "declared; add one (library-import --credit) and show it in the video's credits")
    return out


def plan_fit(source: float, target: Optional[float], loopable: bool, crossfade: float = DEFAULT_CROSSFADE,
             max_repeats: int = MAX_REPEATS) -> Dict[str, Any]:
    """How a file of `source` seconds reaches `target` seconds. Pure. Returns a plan whose `mode` is
      none  no target, or the file is already within max(0.1 s, 1 %) of it: used whole
      trim  the file is longer: the first `target` seconds (a sample-accurate cut)
      loop  the file is shorter and loopable: `repeats` copies joined with a `crossfade` (at most a quarter of the
            file), `joined_duration` = repeats * source - (repeats - 1) * crossfade >= target, then trimmed
      None  not fittable (`reason` says why)."""
    if not source or source <= 0:
        return {"mode": None, "reason": "the file has no measurable duration", "source": source, "target": target}
    if target is None:
        return {"mode": "none", "source": source, "target": None, "tolerance": None, "output_duration": source}
    tol = tolerance_for(target)
    base = {"source": source, "target": target, "tolerance": tol}
    if abs(source - target) <= tol:
        return dict(base, mode="none", output_duration=source)
    if source > target:
        return dict(base, mode="trim", trimmed=round(source - target, 6), output_duration=target)
    if not loopable:
        return dict(base, mode=None, reason=f"{source:.3f}s is shorter than the requested {target:.3f}s and the file is "
                                            "not marked loopable")
    x = round(min(max(crossfade, 0.0), source / 4.0), 6)
    repeats = max(2, math.ceil((target - x) / (source - x) - 1e-9))
    if repeats > max_repeats:
        return dict(base, mode=None, reason=f"looping {source:.3f}s to {target:.3f}s needs {repeats} repeats "
                                            f"(at most {max_repeats})")
    joined = round(repeats * source - (repeats - 1) * x, 6)
    return dict(base, mode="loop", repeats=repeats, crossfade=x, joined_duration=joined,
                trimmed=round(joined - target, 6), output_duration=target)


def match(entry: Dict[str, Any], tags: List[str], moods: List[str]) -> List[str]:
    """The requested tags and moods this entry carries (normalised, in request order)."""
    have_tags = set(word_list(entry.get("tags"), "tags")) if isinstance(entry.get("tags"), (list, str)) else set()
    have_moods = set(word_list(entry.get("mood"), "mood")) if isinstance(entry.get("mood"), (list, str)) else set()
    return [t for t in tags if t in have_tags] + [m for m in moods if m in have_moods]


def screen(files: Dict[str, str], manifest: Dict[str, Dict[str, Any]], kind: Optional[str], tags: List[str],
           moods: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Step 1 of the selection without measuring anything: licence, kind and tags / mood. Returns (kept, excluded);
    each kept row carries key, path, entry, licence and matched."""
    tags, moods = word_list(tags, "tags"), word_list(moods, "mood")
    kept, excluded = [], []
    for key in sorted(files):
        entry = manifest.get(key)
        info = licence_info(entry)
        reasons = []
        if info["status"] != "declared":
            reasons.append("licence unknown" if entry is not None else "no manifest entry (licence unknown)")
        e = entry or {}
        if entry is not None and kind is not None and e.get("kind") != kind:
            reasons.append(f"kind {e.get('kind')!r}, not {kind!r}")
        matched = match(e, tags, moods) if entry is not None else []
        if entry is not None and (tags or moods) and not matched:
            reasons.append("no requested tag or mood")
        if reasons:
            excluded.append({"file": key, "licence": info["licence"], "reasons": reasons})
        else:
            kept.append({"key": key, "path": files[key], "entry": e, "licence": info, "matched": matched})
    return kept, excluded


def rank(rows: List[Dict[str, Any]], target: Optional[float], crossfade: float = DEFAULT_CROSSFADE
         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Steps 1 (fit) and 2 of the selection on screened rows that carry a measured `duration` (and `sha256`). Pure and
    independent of the input order. Returns (ranked, excluded)."""
    ranked, excluded = [], []
    for row in rows:
        plan = plan_fit(float(row.get("duration") or 0.0), target, bool(row["entry"].get("loopable") is True), crossfade)
        if plan["mode"] is None:
            excluded.append({"file": row["key"], "licence": row["licence"]["licence"], "reasons": [plan["reason"]]})
            continue
        ranked.append(dict(row, fit=plan))
    ranked.sort(key=lambda r: (-len(r["matched"]), FIT_RANK[r["fit"]["mode"]],
                               round(abs(r["fit"]["source"] - target), 6) if target is not None else 0.0,
                               r["key"], r.get("sha256") or ""))
    return ranked, sorted(excluded, key=lambda x: x["file"])


def safe_key(relpath: str, what: str) -> str:
    """A library-relative path given by the user: relative, inside the library, no '..'."""
    key = norm_key(relpath or "")
    parts = key.split("/")
    if (not key or os.path.isabs(relpath) or relpath.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", key)
            or ".." in parts):
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be a path relative to the library, without '..'",
                               {what: relpath})
    return key


# ------------------------------------------------------------------------------------------- library on disk

def library_dir(path: str) -> str:
    if not isinstance(path, str) or not os.path.isdir(path):
        raise SoundDesignError("INPUT_NOT_FOUND", f"the library {path} is not a directory", {"library": path})
    return os.path.abspath(path)


def load_manifest(library: str) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """(entries by NFC key, whether the manifest exists). UTF-8, a byte-order mark accepted (Windows Notepad)."""
    path = os.path.join(library, MANIFEST_NAME)
    if not os.path.exists(path):
        return {}, False
    try:
        with open(path, encoding="utf-8-sig") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise SoundDesignError("INVALID_ARGUMENT", f"{MANIFEST_NAME} is not readable UTF-8 JSON: {e}", {"manifest": path})
    files = doc.get("files") if isinstance(doc, dict) else None
    if not isinstance(files, dict):
        raise SoundDesignError("INVALID_ARGUMENT", f"{MANIFEST_NAME} needs an object \"files\" keyed by library-relative "
                               "path", {"manifest": path})
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in files.items():
        key = norm_key(k)
        if not isinstance(v, dict):
            raise SoundDesignError("INVALID_ARGUMENT", f"{MANIFEST_NAME}: the entry for {k!r} is not an object", {"file": k})
        if key in out:
            raise SoundDesignError("INVALID_ARGUMENT", f"{MANIFEST_NAME}: {k!r} is listed twice", {"file": k})
        out[key] = v
    return out, True


def save_manifest(library: str, entries: Dict[str, Dict[str, Any]]) -> str:
    """Written whole and replaced with one rename. UTF-8 (not ASCII-escaped: people read and edit this file)."""
    path = os.path.join(library, MANIFEST_NAME)
    fd, tmp = tempfile.mkstemp(prefix=".sound-library-", suffix=".json", dir=library)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"schema": MANIFEST_SCHEMA, "files": {k: entries[k] for k in sorted(entries)}}, fh, indent=2,
                      ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def scan(library: str) -> Dict[str, str]:
    """Audio files under the library by NFC key -> absolute path, in key order. Hidden files and folders (a leading
    '.', including this skill's own temporary files) and the manifest are skipped; symbolic links to folders are not
    followed."""
    out: Dict[str, str] = {}
    for root, dirs, names in os.walk(library):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith(".") or name == MANIFEST_NAME or os.path.splitext(name)[1].lower() not in AUDIO_EXTENSIONS:
                continue
            full = os.path.join(root, name)
            out[norm_key(os.path.relpath(full, library))] = full
    return dict(sorted(out.items()))


def measure(skill: FfmpegSkill, path: str) -> Dict[str, Any]:
    """ffmpeg-skill/probe facts of a library file; it must be audio only. Raises UNSUPPORTED_MEDIA."""
    facts = skill.probe(path)
    a = facts.get("audio") if isinstance(facts.get("audio"), dict) else {}
    counts = facts.get("stream_counts") if isinstance(facts.get("stream_counts"), dict) else {}
    if not a or facts.get("video") or int(counts.get("video") or 0):
        raise SoundDesignError("UNSUPPORTED_MEDIA", f"{os.path.basename(path)} is not audio only (library files are audio "
                               "files; a video's sound is not a library file)", {"tool": "ffmpeg-skill/probe"})
    duration = float(facts.get("duration") or 0.0)
    if duration <= 0:
        raise SoundDesignError("UNSUPPORTED_MEDIA", f"{os.path.basename(path)} has no measurable duration",
                               {"tool": "ffmpeg-skill/probe"})
    return {"duration": duration, "codec": a.get("codec"), "sample_rate": a.get("sample_rate"), "channels": a.get("channels"),
            "audio_streams": int(counts.get("audio") or 1), "measured_by": "ffmpeg-skill/probe"}


def _check_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise SoundDesignError("INVALID_ARGUMENT", "--timeout must be > 0", {"timeout": timeout})
    return float(timeout)


def _check_kind(kind: Optional[str], required: bool) -> Optional[str]:
    if kind is None and not required:
        return None
    if kind not in KINDS:
        raise SoundDesignError("INVALID_ARGUMENT", f"--kind must be one of {', '.join(KINDS)}", {"kind": kind})
    return kind


def _opt_text(value: Any, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be text", {what: value})
    return value.strip() or None


def _skill_doc(info: Any) -> Dict[str, Any]:
    return {"version": info.version, "contract_version": info.contract_version}


# ------------------------------------------------------------------------------------------------ library-import

@dataclass
class LibraryImportRequest:
    input_path: str
    library_dir: str
    licence: Optional[str]
    kind: Optional[str]
    tags: List[str] = field(default_factory=list)
    mood: List[str] = field(default_factory=list)
    loopable: bool = False
    source: Optional[str] = None
    author: Optional[str] = None
    credit: Optional[str] = None
    name: Optional[str] = None
    loudness: bool = True
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def require_licence(licence: Any, what: str) -> Dict[str, Any]:
    """Refuses a missing licence (roadmap step 4: "done when: refuses a missing licence"), in English and Japanese."""
    info = licence_info({"licence": licence} if licence is not None else None)
    if info["status"] != "declared":
        en = (f"{what} has no declared licence. A library file is used only with its licence as you know it (an SPDX "
              "id such as CC-BY-4.0, or the name / text of the licence you bought it under); declare it with "
              "library-import --licence.")
        ja = (f"{what} のライセンスが宣言されていません。ライセンスが分かっている素材だけを使います。"
              "library-import の --licence で宣言してください。")
        raise SoundDesignError("LICENCE_REQUIRED", f"{en}\n{ja}", {"file": what, "licence": info["licence"],
                                                                  "declared_as": info["declared_as"], "message_en": en,
                                                                  "message_ja": ja, "adr": "docs/decisions.md ADR-008"})
    return info


def library_import(req: LibraryImportRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """library-import: one user-supplied file into the library with its declared licence. Checks first (licence,
    arguments, input, destination), then ffmpeg-skill/probe (and loudness), then the copy and the manifest entry, each
    placed with one rename. The input is never modified."""
    name = os.path.basename(req.input_path or "") or "input"
    info = require_licence(req.licence, name)
    kind = _check_kind(req.kind, required=True)
    tags, moods = word_list(req.tags, "tags"), word_list(req.mood, "mood")
    texts = {k: _opt_text(getattr(req, k), k) for k in ("source", "author", "credit")}
    timeout = _check_timeout(req.timeout)
    if not isinstance(req.input_path, str) or not os.path.isfile(req.input_path):
        raise SoundDesignError("INPUT_NOT_FOUND", f"{req.input_path} is not a file", {"input": req.input_path})
    library = library_dir(req.library_dir)
    src = os.path.abspath(req.input_path)
    real_lib = os.path.realpath(library)
    real_src = os.path.realpath(src)
    try:
        in_place = os.path.commonpath([real_lib, real_src]) == real_lib
    except ValueError:                                   # another drive (Windows): not inside the library
        in_place = False
    if in_place:
        key = norm_key(os.path.relpath(real_src, real_lib))
        if req.name is not None and safe_key(req.name, "--name") != key:
            raise SoundDesignError("INVALID_ARGUMENT", "the file is already in the library; it is registered where it is "
                                   "(--name cannot move it)", {"input": req.input_path, "name": req.name})
        if any(p.startswith(".") for p in key.split("/")) or key == MANIFEST_NAME:
            raise SoundDesignError("INVALID_ARGUMENT", "hidden files and the manifest are not library files", {"input": key})
    else:
        key = safe_key(req.name if req.name is not None else os.path.basename(src), "--name")
    if os.path.splitext(key)[1].lower() not in AUDIO_EXTENSIONS:
        raise SoundDesignError("INVALID_ARGUMENT", f"library files are audio files ({', '.join(AUDIO_EXTENSIONS)})",
                               {"file": key})
    dst = os.path.join(library, *key.split("/"))
    entries, had_manifest = load_manifest(library)
    if not in_place and os.path.lexists(dst) and not req.overwrite:
        raise SoundDesignError("OUTPUT_EXISTS", f"{key} is already in the library; pass --overwrite to replace it", {"file": key})
    if key in entries and not req.overwrite:
        raise SoundDesignError("OUTPUT_EXISTS", f"{key} already has a manifest entry; pass --overwrite to replace it",
                               {"file": key})
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    facts = measure(skill, src)
    loud = skill.loudness(src) if req.loudness else None
    sha = sha256_file(src)
    steps: List[Dict[str, Any]] = [{"tool": "ffmpeg-skill/probe", "result": facts}]
    if loud is not None:
        steps.append({"tool": "ffmpeg-skill/loudness", "result": loud})
    if not in_place:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".sound-design-", suffix=os.path.splitext(key)[1], dir=os.path.dirname(dst))
        os.close(fd)
        try:
            shutil.copyfile(src, tmp)
            if sha256_file(tmp) != sha:
                raise SoundDesignError("INTERNAL_ERROR", "the copy differs from the input", {"file": key})
            if os.path.lexists(dst) and not req.overwrite:
                raise SoundDesignError("OUTPUT_EXISTS", f"{key} appeared during the import", {"file": key})
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    entry: Dict[str, Any] = {"licence": info["licence"], "kind": kind, "tags": tags, "mood": moods,
                             "loopable": req.loopable is True, **{k: v for k, v in texts.items() if v is not None},
                             "sha256": sha, "measured": dict(facts, loudness=loud),
                             "imported_from": os.path.basename(src), "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "imported_by": f"{SKILL_ID} {__version__}"}
    entries[key] = entry
    save_manifest(library, entries)
    licence = licence_info(entry)
    return {
        "schema": IMPORT_SCHEMA,
        "ok": True,
        "op": "library-import",
        "capability": CAPABILITY,
        "skill": {"id": SKILL_ID, "version": __version__},
        "library": {"path": library, "manifest": MANIFEST_NAME, "manifest_created": not had_manifest},
        "file": {"key": key, "path": dst, "copied": not in_place, "sha256": sha, "measured": facts, "loudness": loud},
        "licence": licence,
        "entry": entry,
        "steps": steps,
        "engine": None,
        "ffmpeg_skill": _skill_doc(sinfo),
        "warnings": licence_warnings(key, licence),
    }


# ------------------------------------------------------------------------------------------------- library-index

@dataclass
class LibraryIndexRequest:
    library_dir: str
    loudness: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def library_index(req: LibraryIndexRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """library-index: every audio file in the library with its licence status and measured facts. Read-only."""
    timeout = _check_timeout(req.timeout)
    library = library_dir(req.library_dir)
    entries, had_manifest = load_manifest(library)
    files = scan(library)
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    rows, warnings = [], []
    for key, path in files.items():
        entry = entries.get(key)
        info = licence_info(entry)
        row: Dict[str, Any] = {"file": key, "licence": info, "kind": (entry or {}).get("kind"),
                               "tags": word_list((entry or {}).get("tags"), "tags") if entry else [],
                               "mood": word_list((entry or {}).get("mood"), "mood") if entry else [],
                               "loopable": bool((entry or {}).get("loopable") is True), "in_manifest": entry is not None}
        sha = sha256_file(path)
        row["sha256"] = sha
        try:
            row.update(measured=measure(skill, path), readable=True, loudness=None)
        except SoundDesignError as e:
            row.update(measured=None, readable=False, loudness=None, problem=e.message)
            warnings.append(f"{key}: {e.message}")
        if row["readable"] and req.loudness:
            try:
                row["loudness"] = skill.loudness(path)
            except SoundDesignError as e:
                warnings.append(f"{key}: loudness not measured: {e.message}")
        if entry is not None and entry.get("sha256") and entry["sha256"] != sha:
            row["changed_since_import"] = True
            warnings.append(f"{key}: the file differs from the one imported (SHA-256); re-import it to confirm its licence")
        warnings += [w for w in licence_warnings(key, info) if info["status"] == "declared"]
        rows.append(row)
    missing = sorted(k for k in entries if k not in files)
    warnings += [f"{k}: in {MANIFEST_NAME} but not in the library" for k in missing]
    unknown = [r["file"] for r in rows if r["licence"]["status"] != "declared"]
    return {
        "schema": INDEX_SCHEMA,
        "ok": True,
        "op": "library-index",
        "capability": CAPABILITY,
        "skill": {"id": SKILL_ID, "version": __version__},
        "library": {"path": library, "manifest": MANIFEST_NAME if had_manifest else None},
        "files": rows,
        "summary": {"files": len(rows), "declared": len(rows) - len(unknown), "unknown_licence": len(unknown),
                    "unreadable": sum(1 for r in rows if not r["readable"]), "missing_files": len(missing)},
        "unknown_licence": unknown,
        "missing_files": missing,
        "engine": None,
        "ffmpeg_skill": _skill_doc(sinfo),
        "warnings": warnings,
    }


# -------------------------------------------------------------------------------------------------- library-pick

@dataclass
class LibraryPickRequest:
    library_dir: str
    output_path: str
    kind: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mood: List[str] = field(default_factory=list)
    duration: Optional[float] = None
    file: Optional[str] = None
    crossfade: float = DEFAULT_CROSSFADE
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def _check_pick(req: LibraryPickRequest) -> Tuple[Optional[str], List[str], List[str], Optional[float], float]:
    kind = _check_kind(req.kind, required=False)
    tags, moods = word_list(req.tags, "tags"), word_list(req.mood, "mood")
    d = req.duration
    if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 < d <= 3600 or d != d):
        raise SoundDesignError("INVALID_ARGUMENT", "--duration must be > 0 and <= 3600 seconds", {"duration": d})
    x = req.crossfade
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 10:
        raise SoundDesignError("INVALID_ARGUMENT", "--crossfade must be 0-10 seconds", {"crossfade": x})
    if req.file is None and kind is None:
        raise SoundDesignError("INVALID_ARGUMENT", "give --kind (bgm or sfx) to choose from the library, or --file")
    _check_timeout(req.timeout)
    return kind, tags, moods, None if d is None else float(d), float(x)


def _validate_fit(skill: FfmpegSkill, path: str, rate: Any, expected: float) -> Dict[str, Any]:
    try:
        facts = skill.probe(path)
    except SoundDesignError as e:
        raise SoundDesignError("INVALID_RESULT", f"the fitted file is not readable: {e.message}")
    a = facts.get("audio") if isinstance(facts.get("audio"), dict) else {}
    counts = facts.get("stream_counts") if isinstance(facts.get("stream_counts"), dict) else {}
    problems = []
    if not a or facts.get("video") or int(counts.get("audio") or 1) != 1:
        problems.append("expected one audio stream and no video")
    if a and rate and int(a.get("sample_rate") or 0) != int(rate):
        problems.append(f"sample rate {a.get('sample_rate')}, the library file's is {rate}")
    measured = float(facts.get("duration") or 0.0)
    tol = tolerance_for(expected)
    if abs(measured - expected) > tol:
        problems.append(f"duration {measured:.3f}s, expected {expected:.3f}s (+/- {tol:.3f}s)")
    if problems:
        raise SoundDesignError("INVALID_RESULT", "the fitted file failed validation: " + "; ".join(problems),
                               {"problems": problems})
    return facts


def _fit(skill: FfmpegSkill, row: Dict[str, Any], run_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    """The fitted WAV in `run_dir`, through ffmpeg-skill only. Returns (path, steps)."""
    plan, src, facts = row["fit"], row["path"], row["measured"]
    out = os.path.join(run_dir, "library.wav")
    steps: List[Dict[str, Any]] = []
    if plan["mode"] == "none":
        if src.lower().endswith(".wav") and str(facts.get("codec") or "").startswith("pcm_"):
            shutil.copyfile(src, out)                     # already PCM WAV: the user's bytes, unchanged
            steps.append({"tool": f"{SKILL_ID}/copy", "note": "PCM WAV used as is"})
        else:
            skill.cut(src, out, duration=plan["source"])
            steps.append({"tool": "ffmpeg-skill/cut", "start": 0.0, "duration": plan["source"], "accurate": True,
                          "note": "converted to PCM WAV"})
    elif plan["mode"] == "trim":
        skill.cut(src, out, duration=plan["target"])
        steps.append({"tool": "ffmpeg-skill/cut", "start": 0.0, "duration": plan["target"], "accurate": True})
    else:
        joined = os.path.join(run_dir, "loop.wav")
        skill.join([src] * plan["repeats"], joined, plan["crossfade"])
        steps.append({"tool": "ffmpeg-skill/join", "inputs": plan["repeats"], "crossfade": plan["crossfade"],
                      "transition": "fade" if plan["crossfade"] > 0 else "none"})
        skill.cut(joined, out, duration=plan["target"])
        steps.append({"tool": "ffmpeg-skill/cut", "start": 0.0, "duration": plan["target"], "accurate": True})
    return out, steps


def library_pick(req: LibraryPickRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """library-pick: one library file chosen by the rules in this module's docstring, fitted to --duration and
    placed as a WAV at `req.output_path` with its provenance. Nothing is written to the output path unless every step
    passes; the library is never modified."""
    kind, tags, moods, target, crossfade = _check_pick(req)
    library = library_dir(req.library_dir)
    files = scan(library)
    dst = check_output(req.output_path, req.overwrite, tuple(files.values()))
    entries, had_manifest = load_manifest(library)
    if req.file is not None:
        key = safe_key(req.file, "--file")
        if key not in files:
            raise SoundDesignError("INPUT_NOT_FOUND", f"{key} is not an audio file in the library", {"file": key})
        require_licence(licence_info(entries.get(key))["licence"] if key in entries else None, key)
        e = entries[key]
        if kind is not None and e.get("kind") != kind:
            raise SoundDesignError("INVALID_ARGUMENT", f"{key} is kind {e.get('kind')!r}, not {kind!r}", {"file": key})
        pool = {key: files[key]}
        rows, excluded = screen(pool, entries, kind, [], [])
        for r in rows:
            r["matched"] = match(r["entry"], tags, moods)
    else:
        rows, excluded = screen(files, entries, kind, tags, moods)
    timeout = float(req.timeout)
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    measured_rows = []
    for r in rows:
        try:
            r["measured"] = measure(skill, r["path"])
        except SoundDesignError as e:
            if req.file is not None:
                raise
            excluded.append({"file": r["key"], "licence": r["licence"]["licence"], "reasons": [e.message]})
            continue
        r["duration"] = r["measured"]["duration"]
        r["sha256"] = sha256_file(r["path"])
        measured_rows.append(r)
    ranked, unfit = rank(measured_rows, target, crossfade)
    excluded = sorted(excluded + unfit, key=lambda x: x["file"])
    if not ranked:
        if req.file is not None:
            raise SoundDesignError("INVALID_ARGUMENT", f"{req.file} cannot be fitted: " + "; ".join(unfit[0]["reasons"]),
                                   {"file": req.file})
        raise SoundDesignError("INPUT_NOT_FOUND", "no library file is eligible for this request"
                               + ("" if had_manifest else f" (the library has no {MANIFEST_NAME})"),
                               {"library": library, "excluded": excluded})
    chosen = ranked[0]
    plan = chosen["fit"]
    expected = plan["output_duration"]
    run_dir = tempfile.mkdtemp(prefix=".sound-design-", dir=os.path.dirname(dst))
    try:
        wav, steps = _fit(skill, chosen, run_dir)
        facts = _validate_fit(skill, wav, chosen["measured"].get("sample_rate"), expected)
        loud = skill.loudness(wav)
        steps += [{"tool": "ffmpeg-skill/probe", "result": {"duration": facts.get("duration")}},
                  {"tool": "ffmpeg-skill/loudness", "result": loud}]
        measured = float(facts.get("duration") or 0.0)
        e = chosen["entry"]
        source = {"file": chosen["key"], "sha256": chosen["sha256"], "kind": e.get("kind"),
                  "tags": word_list(e.get("tags"), "tags"), "mood": word_list(e.get("mood"), "mood"),
                  "loopable": e.get("loopable") is True, "measured": chosen["measured"]}
        selection = {"rule": SELECTION_RULE, "query": {"kind": kind, "tags": tags, "mood": moods, "duration": target,
                                                       "file": req.file}, "matched": chosen["matched"],
                     "rank": 1, "candidates": len(ranked)}
        provenance = build_library_provenance(source=source, licence=chosen["licence"], selection=selection, fit=plan,
                                              duration={"requested": target, "measured": measured,
                                                        "tolerance": plan.get("tolerance")},
                                              output_path=wav, ffmpeg_skill=_skill_doc(sinfo))
        if os.path.lexists(dst) and not req.overwrite:
            raise SoundDesignError("OUTPUT_EXISTS", f"{dst} appeared while fitting; pass --overwrite to replace it",
                                   {"output": dst})
        os.replace(wav, dst)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    a = facts.get("audio") or {}
    warnings = licence_warnings(chosen["key"], chosen["licence"], fitted=plan["mode"] != "none")
    if chosen["entry"].get("sha256") and chosen["entry"]["sha256"] != chosen["sha256"]:
        warnings.append(f"{chosen['key']}: the file differs from the one imported (SHA-256); its licence was declared for "
                        "the imported file")
    if plan["mode"] == "trim":
        warnings.append(f"{chosen['key']}: cut at {plan['target']:.3f}s with no fade; audio-production-skill MIX fades it")
    return {
        "schema": PICK_SCHEMA,
        "ok": True,
        "op": "library-pick",
        "capability": CAPABILITY,
        "skill": {"id": SKILL_ID, "version": __version__},
        "output": {"path": dst, "format": "wav", "codec": a.get("codec"), "sample_rate": a.get("sample_rate"),
                   "channels": a.get("channels"), "duration": measured, "sha256": provenance["output"]["sha256"],
                   "loudness": loud},
        "request": selection["query"],
        "selection": {"chosen": chosen["key"], "rule": SELECTION_RULE, "matched": chosen["matched"],
                      "ranked": [{"file": r["key"], "matched": r["matched"], "fit": r["fit"]["mode"],
                                  "duration": r["duration"], "licence": r["licence"]["licence"]} for r in ranked],
                      "excluded": excluded},
        "source": source,
        "fit": plan,
        "licence": chosen["licence"],
        "engine": None,
        "steps": steps,
        "provenance": provenance,
        "handoff": {"to": "audio-production-skill MIX", "path": dst, "note": "placed in the programme there with the "
                                                                              "licence and credit; nothing is mixed here"},
        "warnings": warnings,
    }
