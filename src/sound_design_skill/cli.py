"""sound-design CLI.

stdout: with --json exactly one JSON document, on success and failure alike, ASCII only (non-ASCII as \\u escapes,
ADR-003), so it parses the same on any console code page. stderr: diagnostics only.
Exit code: 0 on success; doctor exits 1 when its status is `fail`; an error exits errors.EXIT_CODES[code].

  sound-design contract --json     the contract (alias: skill --json)
  sound-design doctor --json       engine, weights, device / GPU memory, ffmpeg-skill, ops [--offline] [--ffmpeg-skill DIR]
  sound-design engines --json      registered engines with status and licences [--offline]
  sound-design generate-bgm --prompt <text> [--tags a,b] --duration <s> -o <out.wav> [--seed N] --json   (ADR-004)

generate-sfx and library-import are declared PROPOSED and arrive in roadmap steps 3 and 4."""
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
from .ops import BgmRequest, generate_bgm
from .service import DEFAULT_TIMEOUT


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
    b.add_argument("--lyrics", help="UTF-8 lyrics file (default: instrumental)")
    b.add_argument("--engine", help="engine id (default: acestep)")
    b.add_argument("--device", choices=DEVICES, default="auto", help="compute device passed to the engine (default auto)")
    b.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"engine time limit in seconds (default {DEFAULT_TIMEOUT:g})")
    b.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    b.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout directory")
    b.add_argument("--offline", action="store_true", help="no network (every registered engine already needs none)")
    b.add_argument("--json", action="store_true")
    return ap


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
        for op, o in doc["checks"]["operations"].items():
            print(f"  op {op:14} implemented={o['implemented']} engine={o['engine']} ready={o['ready']}")
        for p in doc["problems"]:
            print(f"  problem: {p}")
        for w in doc["warnings"]:
            print(f"  warning: {w}")
    elif doc.get("schema", "").endswith("/generate-bgm@1"):
        out = doc["output"]
        print(f"{out['path']}: {out['duration']:.2f}s, {out['sample_rate']} Hz, {out['channels']} ch; seed "
              f"{doc['request']['seed']} ({doc['request']['seed_source']}); engine {doc['engine']['id']} {doc['engine']['version']}")
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
                                          offline=args.offline)), as_json)
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
