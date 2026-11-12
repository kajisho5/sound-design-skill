"""sound-design CLI.

stdout: with --json exactly one JSON document, on success and failure alike, ASCII only (non-ASCII as \\u escapes,
ADR-003), so it parses the same on any console code page. stderr: diagnostics only.
Exit code: 0 on success; doctor exits 1 when its status is `fail`, seam-check when a seam fails; an error exits errors.EXIT_CODES[code].

  sound-design contract --json     the contract (alias: skill --json)
  sound-design doctor --json       engine, weights, device / GPU memory, ffmpeg-skill, ops [--offline] [--ffmpeg-skill DIR]
  sound-design engines --json      registered engines with status and licences [--offline]
  sound-design generate-bgm --prompt <text> [--tags a,b] --duration <s> -o <out.wav> [--seed N] --json   (ADR-004)
                            [--bpm N [--time-signature 4]]: whole bars, <out>.beats.json beside it      (ADR-012)
  sound-design generate-sfx --prompt <text> [--tags a,b] --duration <s> -o <out.wav> [--seed N]
                            --acknowledge-ezaudio-training-data --json                                  (ADR-006, ADR-007)
  sound-design library-import <file> --library <dir> --licence <SPDX|text> --kind bgm|sfx [--tags a,b] --json (ADR-008)
  sound-design library-index <dir> [--loudness] --json                                                   (ADR-008)
  sound-design library-pick <dir> -o <out.wav> --kind bgm|sfx [--tags a,b] [--mood m] [--duration <s>] --json
                                                                                                         (ADR-008)
  sound-design loop-bed <in.wav> -o <loop.wav> --duration <s> [--search 1] [--crossfade 0.25] --json      (ADR-011)
  sound-design stinger <in.wav> -o <out.wav> --duration <s> --peak-at <s> [--type whoosh|hit|rise] --json (ADR-011)
  sound-design seam-check <file.wav> --period <s> | --seams <t,t,...> --json                              (ADR-011)
  sound-design sfx-cues --cuts <t,t,...> [--emphasis <t,...>] --library <dir> [-o cues.json] --json        (ADR-013)
                        [--min-reuse-spacing <s>] [--character <effect>=<word,...>]                      (ADR-015)
Library mode runs no engine (never ACE-Step or EzAudio) and refuses a file without a declared licence."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import PACKAGE_NAME, SKILL_ID, __version__
from .contract import contract
from .doctor import doctor, engines_report
from .engines.acestep import DEVICES
from .errors import SoundDesignError
from .library import (DEFAULT_CROSSFADE, LibraryImportRequest, LibraryIndexRequest, LibraryPickRequest, library_import,
                      library_index, library_pick)
from .library import DEFAULT_TIMEOUT as LIBRARY_TIMEOUT
from .ops import SFX_DEVICES, SFX_TIMEOUT, BgmRequest, SfxRequest, generate_bgm, generate_sfx
from .service import DEFAULT_TIMEOUT
from .shaping import (DEFAULT_LOOP_CROSSFADE, DEFAULT_SEARCH, STINGER_TYPES, LoopBedRequest, SeamCheckRequest,
                      StingerRequest, loop_bed, seam_check, stinger)
from .shaping import DEFAULT_TIMEOUT as SHAPING_TIMEOUT
from .pcm import SEAM_WINDOW
from .cues import DEFAULT_LEVEL, DEFAULT_SPACING, CuesRequest, sfx_cues
from .cues import DEFAULT_TIMEOUT as CUES_TIMEOUT


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sound-design", description=f"{PACKAGE_NAME} {__version__}: music beds (BGM) and "
                                 "sound effects from local engines (offline)")
    ap.add_argument("--version", action="version", version=f"{PACKAGE_NAME} {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("contract", "skill"):
        c = sub.add_parser(name, help="print the machine-readable contract")
        c.add_argument("--json", action="store_true", help="JSON on stdout (exactly one document)")
    d = sub.add_parser("doctor", help="what is installed, what is missing, device and GPU memory (never downloads)")
    d.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory (default: env / ~/.claude/skills/ffmpeg-skill / ./vendor / ..)")
    d.add_argument("--offline", action="store_true", help="report readiness with no network at all")
    d.add_argument("--json", action="store_true")
    e = sub.add_parser("engines", help="registered engines: status, version, licences, output-use terms")
    e.add_argument("--offline", action="store_true")
    e.add_argument("--json", action="store_true")
    b = sub.add_parser("generate-bgm", help="a music bed (BGM) from a prompt / tags, a duration and a seed (ACE-Step 1.5); WAV")
    b.add_argument("--prompt", help="what the music is (style, instruments, mood, tempo); joined with --tags")
    b.add_argument("--tags", type=_tag_list, default=[], help="comma-separated tags, e.g. \"ambient,piano,70 bpm\"")
    b.add_argument("--duration", type=float, required=True, help="seconds (the engine's range; ACE-Step 10-600)")
    b.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    b.add_argument("--seed", type=int, help="integer in [0, 2**32); drawn and recorded when omitted")
    b.add_argument("--bpm", type=int, help="tempo, 30-300: the length becomes whole bars and a beat grid is written beside "
                                         "the WAV as <name>.beats.json (ADR-012)")
    b.add_argument("--time-signature", type=int, choices=(2, 3, 4, 6), help="beats per bar with --bpm: 2, 3, 4 (default) or "
                                                                            "6 (6/8)")
    b.add_argument("--lyrics", help="UTF-8 lyrics file (default: instrumental)")
    b.add_argument("--engine", help="engine id (default: acestep)")
    b.add_argument("--device", choices=DEVICES, default="auto", help="compute device passed to the engine (default auto)")
    b.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"engine time limit in seconds (default {DEFAULT_TIMEOUT:g})")
    b.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    b.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    b.add_argument("--offline", action="store_true", help="no network (every registered engine already needs none)")
    b.add_argument("--json", action="store_true")
    s = sub.add_parser("generate-sfx", help="a sound effect from a prompt / tags, a duration and a seed (EzAudio); WAV")
    s.add_argument("--prompt", help="what the effect is (source, action, space); joined with --tags")
    s.add_argument("--tags", type=_tag_list, default=[], help="comma-separated tags, e.g. \"door,wooden,close\"")
    s.add_argument("--duration", type=float, required=True, help="whole seconds (the engine's range; EzAudio 1-10)")
    s.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    s.add_argument("--seed", type=int, help="integer in [0, 2**32); drawn and recorded when omitted")
    s.add_argument("--engine", help="engine id (default: ezaudio)")
    s.add_argument("--device", choices=SFX_DEVICES, default="auto", help="compute device passed to the engine (default auto)")
    s.add_argument("--timeout", type=float, default=SFX_TIMEOUT, help=f"engine time limit in seconds (default {SFX_TIMEOUT:g})")
    s.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    s.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    s.add_argument("--offline", action="store_true", help="no network (every registered engine already needs none)")
    s.add_argument("--acknowledge-ezaudio-training-data", action="store_true",
                   help="required with the ezaudio engine: the user was told that EzAudio was trained on YouTube audio "
                        "not licensed for training (ADR-007); recorded with its time in the provenance")
    s.add_argument("--json", action="store_true")
    li = sub.add_parser("library-import", help="register your own BGM / SFX file in a library folder with its licence "
                                               "(refused without one; no engine)")
    li.add_argument("file", help="the audio file (never modified)")
    li.add_argument("--library", required=True, help="the library folder (holds sound-library.json)")
    li.add_argument("--licence", "--license", dest="licence", help="required: the licence as you know it, an SPDX id "
                    "(CC-BY-4.0, CC0-1.0) or the licence text / name you bought it under; \"unknown\" is refused")
    li.add_argument("--kind", choices=("bgm", "sfx"), required=True)
    li.add_argument("--tags", type=_tag_list, default=[], help="comma-separated tags, e.g. \"piano,corporate\"")
    li.add_argument("--mood", type=_tag_list, default=[], help="comma-separated moods, e.g. \"calm,bright\"")
    li.add_argument("--loopable", action="store_true", help="the file loops cleanly (library-pick may repeat it)")
    li.add_argument("--source", help="where it came from (URL or text)")
    li.add_argument("--author", help="author / composer")
    li.add_argument("--credit", help="the credit line the licence asks for")
    li.add_argument("--title", help="the track's title (used in a composed CC BY attribution line)")
    li.add_argument("--provider", help="where you downloaded it (e.g. Freesound, YouTube Audio Library)")
    li.add_argument("--licence-url", "--license-url", dest="licence_url", help="the licence's web address (recorded, never opened)")
    li.add_argument("--licence-file", "--license-file", dest="licence_file",
                    help="the licence text that came with the download (copied into <library>/licences/)")
    li.add_argument("--attribution", help="the attribution line exactly as the source asks for it")
    li.add_argument("--content-id", choices=("registered", "not-registered"),
                    help="only when the source says so: the track is (not) registered for Content ID")
    li.add_argument("--name", help="path inside the library (default: the file name)")
    li.add_argument("--no-loudness", dest="loudness", action="store_false", help="skip the loudness measurement")
    li.add_argument("--overwrite", action="store_true", help="replace a file or manifest entry already in the library")
    li.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    li.add_argument("--timeout", type=float, default=LIBRARY_TIMEOUT, help=f"per ffmpeg-skill tool (default {LIBRARY_TIMEOUT:g} s)")
    li.add_argument("--json", action="store_true")
    lx = sub.add_parser("library-index", help="every audio file in a library folder with its licence status and "
                                              "measured duration (read-only)")
    lx.add_argument("library", help="the library folder")
    lx.add_argument("--loudness", action="store_true", help="also measure each file's EBU R128 loudness")
    lx.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    lx.add_argument("--timeout", type=float, default=LIBRARY_TIMEOUT, help=f"per ffmpeg-skill tool (default {LIBRARY_TIMEOUT:g} s)")
    lx.add_argument("--json", action="store_true")
    lp = sub.add_parser("library-pick", help="choose a library file by kind / tags / mood, fit it to a duration (trim "
                                             "or loop) and write a WAV with its provenance (no engine)")
    lp.add_argument("library", help="the library folder")
    lp.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    lp.add_argument("--kind", choices=("bgm", "sfx"))
    lp.add_argument("--tags", type=_tag_list, default=[], help="comma-separated tags; at least one must match")
    lp.add_argument("--mood", type=_tag_list, default=[], help="comma-separated moods; counted with the tags")
    lp.add_argument("--duration", type=float, help="target seconds (default: the file's own length)")
    lp.add_argument("--file", help="use this library file (path inside the library) instead of choosing")
    lp.add_argument("--crossfade", type=float, default=DEFAULT_CROSSFADE,
                    help=f"seconds between loop repeats (default {DEFAULT_CROSSFADE:g}; at most a quarter of the file)")
    lp.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    lp.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    lp.add_argument("--timeout", type=float, default=LIBRARY_TIMEOUT, help=f"per ffmpeg-skill tool (default {LIBRARY_TIMEOUT:g} s)")
    lp.add_argument("--json", action="store_true")
    lb = sub.add_parser("loop-bed", help="a bed whose end runs seamlessly into its start (loop point searched, wrap "
                                         "measured); no engine")
    lb.add_argument("input", help="the source bed (at least --duration + --crossfade long; never modified)")
    lb.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    lb.add_argument("--duration", type=float, required=True, help="loop length in seconds (1-600)")
    lb.add_argument("--search", type=float, default=DEFAULT_SEARCH,
                    help=f"look for the best loop point within +/- this many seconds (default {DEFAULT_SEARCH:g}; 0 = exact)")
    lb.add_argument("--crossfade", type=float, default=DEFAULT_LOOP_CROSSFADE,
                    help=f"seconds of the source's continuation blended into its start (default {DEFAULT_LOOP_CROSSFADE:g})")
    lb.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    lb.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    lb.add_argument("--timeout", type=float, default=SHAPING_TIMEOUT, help=f"per ffmpeg-skill tool (default {SHAPING_TIMEOUT:g} s)")
    lb.add_argument("--json", action="store_true")
    st = sub.add_parser("stinger", help="a transition sound of an exact length with its peak at a stated time; no engine")
    st.add_argument("input", help="the source sound (a whoosh, hit or rise; never modified)")
    st.add_argument("-o", "--output", required=True, help="output .wav (refused if it exists, unless --overwrite)")
    st.add_argument("--duration", type=float, required=True, help="exact length in seconds (up to 30)")
    st.add_argument("--peak-at", type=float, required=True, help="seconds from the start where the peak lands (the cut)")
    st.add_argument("--type", choices=STINGER_TYPES, default="other", help="what it is (recorded)")
    st.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    st.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    st.add_argument("--timeout", type=float, default=SHAPING_TIMEOUT, help=f"per ffmpeg-skill tool (default {SHAPING_TIMEOUT:g} s)")
    st.add_argument("--json", action="store_true")
    sc = sub.add_parser("seam-check", help="click, level and spectrum across each seam of a WAV (read-only)")
    sc.add_argument("input", help="the file (e.g. a loop repeated end to end)")
    sc.add_argument("--period", type=float, help="a seam every this many seconds")
    sc.add_argument("--seams", type=_time_list, default=[], help="comma-separated seam times in seconds")
    sc.add_argument("--window", type=float, default=SEAM_WINDOW, help=f"seconds compared on each side (default {SEAM_WINDOW:g})")
    sc.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    sc.add_argument("--timeout", type=float, default=SHAPING_TIMEOUT, help=f"per ffmpeg-skill tool (default {SHAPING_TIMEOUT:g} s)")
    sc.add_argument("--json", action="store_true")
    cu = sub.add_parser("sfx-cues", help="a cue sheet of sound effects at the cut points / emphasis times you give, for MIX "
                                         "to place (nothing is mixed; no cue is added)")
    cu.add_argument("--cuts", type=_time_list, default=[], help="comma-separated cut times in seconds")
    cu.add_argument("--emphasis", type=_time_list, default=[], help="comma-separated emphasis times in seconds")
    cu.add_argument("--events", dest="events_path", help="JSON: [{\"time\", \"kind\": cut|emphasis, \"effect\", \"level\", "
                                                         "\"align\": peak|onset, \"character\": [words], \"label\"}] or "
                                                         "{\"events\": [...]}")
    cu.add_argument("--library", help="library folder to take the effects from (kind sfx, tagged with the effect name)")
    cu.add_argument("--effect", action="append", default=[], type=_pair, metavar="KIND=NAME",
                    help="effect for an event kind (default cut=whoosh, emphasis=pop)")
    cu.add_argument("--file", action="append", default=[], type=_pair, metavar="NAME=WAV",
                    help="use this WAV for the effect NAME (e.g. a generate-sfx or stinger result)")
    cu.add_argument("--min-spacing", type=float, default=DEFAULT_SPACING,
                    help=f"seconds; an event closer than this to the last kept cue is dropped (default {DEFAULT_SPACING:g})")
    cu.add_argument("--level", type=float, default=DEFAULT_LEVEL,
                    help=f"sample peak of every cue in dBFS, unless the event gives its own (default {DEFAULT_LEVEL:g})")
    cu.add_argument("--min-reuse-spacing", type=float, default=None, metavar="SECONDS",
                    help="off by default; an event whose effect would play the same sound file as a cue kept less than "
                         "this many seconds earlier takes the next library file for that effect in rank order, or is "
                         "dropped when there is none (a named --file is never replaced)")
    cu.add_argument("--character", action="append", default=[], type=_pair, metavar="EFFECT=WORD[,WORD]",
                    help="only library files for EFFECT whose tags or mood carry every WORD (e.g. pop=bright); an "
                         "event's own \"character\" replaces it")
    cu.add_argument("-o", "--output", help="write the cue sheet here too (.json)")
    cu.add_argument("--overwrite", action="store_true", help="allow replacing an existing cue sheet")
    cu.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    cu.add_argument("--timeout", type=float, default=CUES_TIMEOUT, help=f"per ffmpeg-skill tool (default {CUES_TIMEOUT:g} s)")
    cu.add_argument("--json", action="store_true")
    return ap


def _pair(text: str) -> tuple:
    key, sep, value = text.partition("=")
    if not sep or not key.strip() or not value.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return key.strip(), value.strip()


def _time_list(text: str) -> list:
    try:
        return [float(t) for t in text.split(",") if t.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated seconds, got {text!r}")


def _tag_list(text: str) -> list:
    return [t.strip() for t in text.split(",") if t.strip()]


def _emit(doc: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    else:
        _human(doc)
    sys.stdout.flush()


def _human(doc: Dict[str, Any]) -> None:
    if doc.get("error"):
        err = doc["error"]
        sys.stderr.write(f"error [{err['code']}]: {err['message']}\n")
    elif "checks" in doc:
        print(f"{PACKAGE_NAME} {__version__}: {doc['status']}")
        fs = doc["checks"]["ffmpeg_skill"]
        dev = doc["checks"]["device"]
        print(f"  ffmpeg-skill  {fs['status']:12} {fs.get('version') or fs.get('detail') or ''}")
        print(f"  device        {dev['accelerator']:12} {dev.get('detail') or ''}")
        if dev.get("acestep_recommendation"):
            print(f"    ACE-Step tier: {dev['acestep_recommendation']}")
        for eid, e in doc["checks"]["engines"].items():
            print(f"  engine {eid:17} {e['status']:12} {e.get('detail') or ''}")
            if e.get("install"):
                print(f"    install: {e['install']}")
        for sid, st in (doc["checks"].get("stock_sources") or {}).items():
            print(f"  stock {sid:18} {st['status']:12} {st['needs'] or 'download by hand'}")
        for op, o in doc["checks"]["operations"].items():
            print(f"  op {op:14} implemented={o['implemented']} engine={o['engine']} ready={o['ready']}")
            if o.get("confirmation_required"):
                print(f"    confirm first: tell the user, then pass {o['confirmation']['flag']} (ADR-007)")
        for p in doc["problems"]:
            print(f"  problem: {p}")
        for w in doc["warnings"]:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith(("/generate-bgm@1", "/generate-sfx@1")):
        out = doc["output"]
        print(f"{out['path']}: {out['duration']:.2f}s, {out['sample_rate']} Hz, {out['channels']} ch; seed "
              f"{doc['request']['seed']} ({doc['request']['seed_source']}); engine {doc['engine']['id']} {doc['engine']['version']}")
        if doc.get("tempo_map"):
            t = doc["tempo_map"]
            print(f"  {t['bpm']} BPM {t['time_signature']}, {t['bars']} bars; beat grid {t['path']}")
        if doc.get("confirmation"):
            c = doc["confirmation"]
            print(f"  acknowledged: {c['flag']} at {c['acknowledged_at']}")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/library-import@1"):
        lic = doc["licence"]
        print(f"{doc['file']['key']}: {doc['file']['measured']['duration']:.2f}s; licence {lic['licence']} ({lic['status']}, "
              f"not verified); {'copied into' if doc['file']['copied'] else 'registered in'} {doc['library']['path']}")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/library-index@1"):
        for f in doc["files"]:
            m = f.get("measured") or {}
            dur = f"{m['duration']:.2f}s" if m.get("duration") else "unreadable"
            print(f"{f['file']}: {dur}; {f.get('kind') or '-'}; licence {f['licence']['licence']}")
        sm = doc["summary"]
        print(f"  {sm['files']} files, {sm['declared']} with a declared licence, {sm['unknown_licence']} unknown (not used)")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/library-pick@1"):
        out = doc["output"]
        print(f"{out['path']}: {out['duration']:.2f}s from {doc['source']['file']} ({doc['fit']['mode']}); licence "
              f"{doc['licence']['licence']} (declared, not verified)")
        if doc["licence"].get("attribution"):
            print(f"  credit: {doc['licence']['attribution']}")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/loop-bed@1"):
        lp, sm = doc["loop"], doc["seam"]
        print(f"{doc['output']['path']}: loop {lp['length']:.3f}s (similarity {lp['similarity']:.2f}); wrap click "
              f"{sm['click_ratio']:.2f}, level {sm['level_delta_db']:.2f} dB, spectrum {sm['spectrum_delta_db']:.2f} dB")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/stinger@1"):
        t = doc["timing"]
        print(f"{doc['output']['path']}: {t['duration']:.3f}s, peak at {t['peak']['time']:.3f}s "
              f"(asked {t['peak_at']:.3f}s, error {t['peak_error'] * 1000:+.1f} ms)")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/seam-check@1"):
        print(f"{doc['file']['name']}: {'passed' if doc['passed'] else 'FAILED'} ({len(doc['seams'])} seams)")
        for s in doc["seams"]:
            print(f"  {s['time']:.3f}s: {'ok' if s['passed'] else '; '.join(s['reasons'])}")
    elif doc.get("schema", "").endswith("/sfx-cues@1"):
        for c in doc["cues"]:
            instead = f"  (instead of {c['substituted']['for']})" if c.get("substituted") else ""
            print(f"{c['time']:9.3f}s  {c['kind']:8} {c['effect']:10} start {c['start']:.3f}s  gain {c['gain_db']:+.1f} dB  "
                  f"{c['file']}{instead}")
        for d in doc["dropped"]:
            print(f"{d['time']:9.3f}s  dropped: {d['reason']}")
        for w in doc.get("warnings") or []:
            print(f"  warning: {w}")
    elif "engines" in doc and "skill_id" not in doc:
        for eid, e in doc["engines"].items():
            print(f"{eid:17} {e['status']:12} {e.get('version') or '-':10} code {e['code_license']}; "
                  f"weights {e['weights_license']}")
    else:
        print(f"{doc['skill_id']} {doc['version']}: " + ", ".join(f"{p['id']} ({p['lifecycle']})" for p in doc["provides"]))


def _human_streams() -> None:
    """Human-readable output never crashes on a legacy code page (Windows cp1252): characters the console cannot
    encode are written as backslash escapes. JSON needs no such help, it is ASCII (ADR-003)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except (ValueError, OSError):  # pragma: no cover - a stream that cannot be reconfigured
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _human_streams()
    args = build_parser().parse_args(argv)
    as_json = bool(getattr(args, "json", False))
    try:
        if args.cmd in ("contract", "skill"):
            _emit(contract(), as_json)
            return 0
        if args.cmd == "doctor":
            doc = doctor(args.ffmpeg_skill, offline=args.offline)
            _emit(doc, as_json)
            return 1 if doc["status"] == "fail" else 0
        if args.cmd == "engines":
            _emit({"ok": True, "skill": {"id": SKILL_ID, "version": __version__}, "offline": bool(args.offline),
                   "engines": engines_report(offline=args.offline)}, as_json)
            return 0
        if args.cmd == "generate-bgm":
            _emit(generate_bgm(BgmRequest(output_path=args.output, duration=args.duration, prompt=args.prompt, tags=args.tags,
                                          seed=args.seed, lyrics_path=args.lyrics, engine=args.engine, device=args.device,
                                          timeout=args.timeout, overwrite=args.overwrite, ffmpeg_skill_dir=args.ffmpeg_skill,
                                          offline=args.offline, bpm=args.bpm, time_signature=args.time_signature)), as_json)
            return 0
        if args.cmd == "generate-sfx":
            _emit(generate_sfx(SfxRequest(output_path=args.output, duration=args.duration, prompt=args.prompt, tags=args.tags,
                                          seed=args.seed, engine=args.engine, device=args.device, timeout=args.timeout,
                                          overwrite=args.overwrite, ffmpeg_skill_dir=args.ffmpeg_skill,
                                          offline=args.offline,
                                          acknowledge_ezaudio_training_data=args.acknowledge_ezaudio_training_data)), as_json)
            return 0
        if args.cmd == "library-import":
            _emit(library_import(LibraryImportRequest(input_path=args.file, library_dir=args.library, licence=args.licence,
                                                      kind=args.kind, tags=args.tags, mood=args.mood, loopable=args.loopable,
                                                      source=args.source, author=args.author, credit=args.credit,
                                                      name=args.name, loudness=args.loudness, overwrite=args.overwrite,
                                                      title=args.title, provider=args.provider, licence_url=args.licence_url,
                                                      licence_file=args.licence_file, attribution=args.attribution,
                                                      content_id=args.content_id,
                                                      ffmpeg_skill_dir=args.ffmpeg_skill, timeout=args.timeout)), as_json)
            return 0
        if args.cmd == "library-index":
            _emit(library_index(LibraryIndexRequest(library_dir=args.library, loudness=args.loudness,
                                                    ffmpeg_skill_dir=args.ffmpeg_skill, timeout=args.timeout)), as_json)
            return 0
        if args.cmd == "library-pick":
            _emit(library_pick(LibraryPickRequest(library_dir=args.library, output_path=args.output, kind=args.kind,
                                                  tags=args.tags, mood=args.mood, duration=args.duration, file=args.file,
                                                  crossfade=args.crossfade, overwrite=args.overwrite,
                                                  ffmpeg_skill_dir=args.ffmpeg_skill, timeout=args.timeout)), as_json)
            return 0
        if args.cmd == "loop-bed":
            _emit(loop_bed(LoopBedRequest(input_path=args.input, output_path=args.output, duration=args.duration,
                                          search=args.search, crossfade=args.crossfade, overwrite=args.overwrite,
                                          ffmpeg_skill_dir=args.ffmpeg_skill, timeout=args.timeout)), as_json)
            return 0
        if args.cmd == "stinger":
            _emit(stinger(StingerRequest(input_path=args.input, output_path=args.output, duration=args.duration,
                                         peak_at=args.peak_at, type=args.type, overwrite=args.overwrite,
                                         ffmpeg_skill_dir=args.ffmpeg_skill, timeout=args.timeout)), as_json)
            return 0
        if args.cmd == "seam-check":
            doc = seam_check(SeamCheckRequest(input_path=args.input, period=args.period, seams=args.seams, window=args.window,
                                              ffmpeg_skill_dir=args.ffmpeg_skill, timeout=args.timeout))
            _emit(doc, as_json)
            return 0 if doc["passed"] else 1
        if args.cmd == "sfx-cues":
            _emit(sfx_cues(CuesRequest(cuts=args.cuts, emphasis=args.emphasis, events_path=args.events_path,
                                       library_dir=args.library, effects=dict(args.effect), files=dict(args.file),
                                       min_spacing=args.min_spacing, level=args.level, output_path=args.output,
                                       overwrite=args.overwrite, ffmpeg_skill_dir=args.ffmpeg_skill,
                                       timeout=args.timeout, min_reuse_spacing=args.min_reuse_spacing,
                                       characters={k: _tag_list(v) for k, v in args.character})), as_json)
            return 0
    except SoundDesignError as e:
        _emit({"ok": False, "skill": {"id": SKILL_ID, "version": __version__}, "error": e.to_dict()}, as_json)
        return e.exit_code
    except Exception as e:  # pragma: no cover - a bug; still one JSON document
        err = SoundDesignError("INTERNAL_ERROR", f"{type(e).__name__}: {e}")
        _emit({"ok": False, "skill": {"id": SKILL_ID, "version": __version__}, "error": err.to_dict()}, as_json)
        return err.exit_code
    return 2  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
