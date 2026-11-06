"""Loopable beds, timed stingers and the seam check (roadmap step 7, ADR-011). No engine runs here: each tool takes a
WAV the user already has (a generate-bgm / generate-sfx / library-pick result, or their own file), edits it only
through ffmpeg-skill (cut, join, audio fades) and measures the result from its samples (pcm.py).

  loop-bed    a bed whose end runs seamlessly into its start, so an editor can repeat it under a scene of any length.
              From a source at least --duration + --crossfade long: the loop length L is searched within +/- --search
              seconds of --duration for the point whose next --crossfade seconds look most like the source's first
              --crossfade seconds; then
                A = source[L - 2x, L + x], B = source[0, L], J = join(A, B) with an x-second crossfade,
                loop = J[2x, 2x + L]  =  crossfade(source[L, L + x] -> source[0, x]) ++ source[x, L]
              so the loop's last sample is source[L - 1] and its first is source[L]: the wrap is the source's own
              continuity. The wrap is measured (seam check) and refused if it fails.
  stinger     a transition sound (whoosh, hit, rise) of exactly --duration seconds whose peak (the loudest 10 ms) is
              at --peak-at seconds, so it lands on the cut. The source is cut (and faded for a few milliseconds where
              it was cut short) and padded with digital silence; the result is re-measured and refused unless its peak
              is within 10 ms of --peak-at and its length exact.
  seam-check  click, level and spectrum across each seam of a WAV (every --period seconds and / or at --seams); the
              loop check qc-skill does not have yet (ADR-011). Read-only.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import SKILL_ID, __version__
from . import pcm as P
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill
from .library import measure
from .ops import check_output, usable_skill
from .provenance import build_shaping_provenance, sha256_file

LOOP_SCHEMA = f"{SKILL_ID}/loop-bed@1"
STINGER_SCHEMA = f"{SKILL_ID}/stinger@1"
SEAM_SCHEMA = f"{SKILL_ID}/seam-check@1"
DEFAULT_TIMEOUT = 600.0
DEFAULT_LOOP_CROSSFADE = 0.25
DEFAULT_SEARCH = 1.0
MAX_LOOP = 600.0
MAX_STINGER = 30.0
PEAK_TOLERANCE = 0.010        # roadmap step 7: "a stinger's peak is within 10 ms of the stated time"
DECLICK_IN = 0.005            # fade where a sound's head was cut away
DECLICK_OUT = 0.010           # fade where a sound's tail was cut away
PEAK_MARGIN = 0.020           # --peak-at at least this far from either end
STINGER_TYPES = ("whoosh", "hit", "rise", "other")
LOOP_SIMILARITY_WARN = 0.5
SEAM_METHOD = ("click: the largest 2nd difference within +/- 1 ms of the seam over the 99th percentile within +/- 50 ms; "
               "level: RMS of --window seconds on each side; spectrum: octave bands (Welch, 512-point Hann frames of the "
               "signal averaged to about 12 kHz) of --window seconds on each side, over bands within 30 dB of the "
               "loudest (ADR-011)")


def _number(value: Any, what: str, lo: float, hi: float, lo_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be a number", {what: value})
    if value > hi or value < lo or (lo_open and value == lo):
        raise SoundDesignError("INVALID_ARGUMENT", f"{what} must be {'>' if lo_open else '>='} {lo:g} and <= {hi:g}",
                               {what: value})
    return float(value)


def _input(path: Any) -> str:
    if not isinstance(path, str) or not os.path.isfile(path):
        raise SoundDesignError("INPUT_NOT_FOUND", f"{path} is not a file", {"input": path})
    return os.path.abspath(path)


def _skill_doc(info: Any) -> Dict[str, Any]:
    return {"version": info.version, "contract_version": info.contract_version}


def _as_pcm(skill: FfmpegSkill, src: str, facts: Dict[str, Any], run_dir: str, name: str = "source.wav"
            ) -> Tuple[str, List[Dict[str, Any]]]:
    """The source as PCM WAV in `run_dir`, through ffmpeg-skill/cut --accurate over its whole length."""
    out = os.path.join(run_dir, name)
    skill.cut(src, out, duration=facts["duration"])
    return out, [{"tool": "ffmpeg-skill/cut", "start": 0.0, "duration": facts["duration"], "accurate": True,
                  "note": "the source as PCM WAV"}]


def _validate(skill: FfmpegSkill, path: str, rate: int, frames: int, what: str) -> Tuple[P.Pcm, Dict[str, Any]]:
    """ffmpeg-skill/probe and the samples: one audio stream, the source's rate, exactly `frames` samples (+/- 1)."""
    try:
        facts = skill.probe(path)
    except SoundDesignError as e:
        raise SoundDesignError("INVALID_RESULT", f"the {what} is not readable: {e.message}")
    pcm = P.read_wav(path)
    problems = []
    a = facts.get("audio") if isinstance(facts.get("audio"), dict) else {}
    if not a or facts.get("video"):
        problems.append("expected one audio stream and no video")
    if pcm.rate != rate:
        problems.append(f"sample rate {pcm.rate}, the source's is {rate}")
    if abs(pcm.frames - frames) > 1:
        problems.append(f"{pcm.frames} samples, expected {frames}")
    if problems:
        raise SoundDesignError("INVALID_RESULT", f"the {what} failed validation: " + "; ".join(problems), {"problems": problems})
    return pcm, facts


def _place(path: str, dst: str, overwrite: bool) -> None:
    if os.path.lexists(dst) and not overwrite:
        raise SoundDesignError("OUTPUT_EXISTS", f"{dst} appeared while working; pass --overwrite to replace it", {"output": dst})
    os.replace(path, dst)


def _output_doc(dst: str, pcm: P.Pcm, facts: Dict[str, Any], sha: str) -> Dict[str, Any]:
    a = facts.get("audio") or {}
    return {"path": dst, "format": "wav", "codec": a.get("codec"), "sample_rate": pcm.rate, "channels": pcm.channels,
            "frames": pcm.frames, "duration": round(pcm.duration, 6), "sha256": sha}


# ------------------------------------------------------------------------------------------------------ loop-bed

@dataclass
class LoopBedRequest:
    input_path: str
    output_path: str
    duration: float
    search: float = DEFAULT_SEARCH
    crossfade: float = DEFAULT_LOOP_CROSSFADE
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def loop_bed(req: LoopBedRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """loop-bed: a seamless loop of about --duration seconds (the module docstring has the construction)."""
    duration = _number(req.duration, "--duration", 1.0, MAX_LOOP)
    search = _number(req.search, "--search", 0.0, 10.0)
    crossfade = _number(req.crossfade, "--crossfade", 0.01, 2.0)
    if crossfade > duration / 4:
        raise SoundDesignError("INVALID_ARGUMENT", "--crossfade may be at most a quarter of --duration",
                               {"crossfade": crossfade, "duration": duration})
    timeout = _number(req.timeout, "--timeout", 0.0, 1e9, lo_open=True)
    src = _input(req.input_path)
    dst = check_output(req.output_path, req.overwrite, (src,))
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    facts = measure(skill, src)
    if facts["duration"] < duration - search + crossfade:
        raise SoundDesignError("INVALID_ARGUMENT", f"the source is {facts['duration']:.3f}s; a loop of {duration:g}s with a "
                               f"{crossfade:g}s crossfade needs at least {duration - search + crossfade:.3f}s",
                               {"source_duration": facts["duration"]})
    run_dir = tempfile.mkdtemp(prefix=".sound-design-", dir=os.path.dirname(dst))
    try:
        source, steps = _as_pcm(skill, src, facts, run_dir)
        pcm = P.read_wav(source)
        r = pcm.rate
        x = P.frames_for(crossfade, r)
        y = 2 * x
        L, similarity = P.find_loop_point(pcm.mono, r, P.frames_for(duration, r), P.frames_for(search, r), x,
                                          lowest=max(y, 2 * x + 1), highest=pcm.frames - x - 1)
        a, b = os.path.join(run_dir, "a.wav"), os.path.join(run_dir, "b.wav")
        joined, out = os.path.join(run_dir, "joined.wav"), os.path.join(run_dir, "loop.wav")
        skill.cut(source, a, start=(L - y) / r, duration=(x + y) / r)
        skill.cut(source, b, start=0.0, duration=L / r)
        skill.join([a, b], joined, x / r)
        skill.cut(joined, out, start=y / r, duration=L / r)
        steps += [{"tool": "ffmpeg-skill/cut", "start": round((L - y) / r, 6), "duration": round((x + y) / r, 6),
                   "accurate": True, "part": "A: source[L - 2x, L + x]"},
                  {"tool": "ffmpeg-skill/cut", "start": 0.0, "duration": round(L / r, 6), "accurate": True,
                   "part": "B: source[0, L]"},
                  {"tool": "ffmpeg-skill/join", "inputs": 2, "crossfade": round(x / r, 6), "transition": "fade"},
                  {"tool": "ffmpeg-skill/cut", "start": round(y / r, 6), "duration": round(L / r, 6), "accurate": True,
                   "part": "loop = J[2x, 2x + L]"}]
        res, ofacts = _validate(skill, out, r, L, "loop")
        w = min(P.frames_for(P.SEAM_WINDOW, r), res.frames // 2)
        wrap = P.seam_check(res.mono[-w:] + res.mono[:w], r, [w], window=w / r)
        seam = dict(wrap["seams"][0], time=round(res.duration, 6), sample=res.frames, where="end -> start (the wrap)")
        if not seam["passed"]:
            raise SoundDesignError("INVALID_RESULT", "the loop's wrap failed the seam check: " + "; ".join(seam["reasons"]),
                                   {"seam": seam})
        steps.append({"tool": f"{SKILL_ID}/seam-check", "result": {"passed": True, "click_ratio": seam["click_ratio"],
                                                                     "level_delta_db": seam["level_delta_db"],
                                                                     "spectrum_delta_db": seam["spectrum_delta_db"]}})
        loop = {"length": round(L / r, 6), "frames": L, "requested": duration, "search": search,
                "loop_point": round(L / r, 6), "crossfade": round(x / r, 6), "similarity": similarity,
                "construction": "crossfade(source[L, L + x] -> source[0, x]) ++ source[x, L]"}
        sha = sha256_file(out)
        provenance = build_shaping_provenance(
            op="loop-bed", source={"file": os.path.basename(src), "sha256": sha256_file(src), "measured": facts},
            parameters={"duration": duration, "search": search, "crossfade": crossfade}, result={"loop": loop, "seam": seam},
            output_sha256=sha, ffmpeg_skill=_skill_doc(sinfo))
        _place(out, dst, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    warnings = []
    if similarity < LOOP_SIMILARITY_WARN:
        warnings.append(f"the crossfade blends material that differs (similarity {similarity:.2f}); the music does not "
                        "repeat near this length: listen, or try another --duration (a whole number of bars loops best)")
    if abs(L / r - duration) > 0.0005:
        warnings.append(f"the loop is {L / r:.3f}s, not {duration:g}s: the best loop point within +/- {search:g}s")
    return {
        "schema": LOOP_SCHEMA, "ok": True, "op": "loop-bed", "skill": {"id": SKILL_ID, "version": __version__},
        "output": _output_doc(dst, res, ofacts, sha), "loop": loop, "seam": seam, "source": provenance["source"],
        "engine": None, "steps": steps, "provenance": provenance,
        "handoff": {"to": "audio-production-skill MIX / the editor", "path": dst,
                    "note": "repeat it end to end (no crossfade needed); the source's own provenance carries its "
                            "engine or licence"},
        "warnings": warnings,
    }


# ------------------------------------------------------------------------------------------------------- stinger

@dataclass
class StingerRequest:
    input_path: str
    output_path: str
    duration: float
    peak_at: float
    type: str = "other"
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def plan_stinger(source_frames: int, source_peak: int, frames: int, peak_at: int) -> Dict[str, int]:
    """Samples: `head` cut from the source's start, `pre` silence before it, `body` source samples kept, `post`
    silence after, so that pre + body + post = frames and the source's peak lands on `peak_at`. Pure."""
    head = max(0, source_peak - peak_at)
    pre = max(0, peak_at - source_peak)
    body = max(0, min(source_frames - head, frames - pre))
    return {"head": head, "pre": pre, "body": body, "post": frames - pre - body}


def stinger(req: StingerRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """stinger: the source placed so its peak is at --peak-at in exactly --duration seconds."""
    duration = _number(req.duration, "--duration", 0.0, MAX_STINGER, lo_open=True)
    peak_at = _number(req.peak_at, "--peak-at", 0.0, duration)
    if not PEAK_MARGIN <= peak_at <= duration - PEAK_MARGIN:
        raise SoundDesignError("INVALID_ARGUMENT", f"--peak-at must be at least {PEAK_MARGIN * 1000:g} ms from either end "
                               "(a peak is read from a 10 ms window, and a cut end gets a 10 ms fade)",
                               {"peak_at": peak_at, "duration": duration})
    if req.type not in STINGER_TYPES:
        raise SoundDesignError("INVALID_ARGUMENT", f"--type must be one of {', '.join(STINGER_TYPES)}", {"type": req.type})
    timeout = _number(req.timeout, "--timeout", 0.0, 1e9, lo_open=True)
    src = _input(req.input_path)
    dst = check_output(req.output_path, req.overwrite, (src,))
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    facts = measure(skill, src)
    run_dir = tempfile.mkdtemp(prefix=".sound-design-", dir=os.path.dirname(dst))
    try:
        source, steps = _as_pcm(skill, src, facts, run_dir)
        pcm = P.read_wav(source)
        r = pcm.rate
        src_peak = P.peak(pcm)
        frames, at = P.frames_for(duration, r), P.frames_for(peak_at, r)
        plan = plan_stinger(pcm.frames, P.frames_for(src_peak["time"], r), frames, at)
        if plan["body"] <= 0:
            raise SoundDesignError("INVALID_ARGUMENT", "nothing of the source would be left in the stinger",
                                   {"plan": plan})
        body = os.path.join(run_dir, "body.wav")
        skill.cut(source, body, start=plan["head"] / r, duration=plan["body"] / r)
        steps.append({"tool": "ffmpeg-skill/cut", "start": round(plan["head"] / r, 6), "duration": round(plan["body"] / r, 6),
                      "accurate": True})
        fade_in = DECLICK_IN if plan["head"] > 0 else 0.0
        fade_out = DECLICK_OUT if plan["head"] + plan["body"] < pcm.frames - 1 else 0.0
        if fade_in or fade_out:
            faded = os.path.join(run_dir, "body-faded.wav")
            skill.fade(body, faded, fade_in=fade_in, fade_out=fade_out)
            steps.append({"tool": "ffmpeg-skill/audio", "fade_in": fade_in, "fade_out": fade_out,
                          "note": "declick where the sound was cut short"})
            body = faded
        body_frames = P.read_wav(body).frames
        pieces = []
        if plan["pre"] > 0:
            pieces.append(P.write_silence(os.path.join(run_dir, "pre.wav"), r, pcm.channels, plan["pre"]))
        pieces.append(body)
        post = frames - plan["pre"] - body_frames
        if post > 0:
            pieces.append(P.write_silence(os.path.join(run_dir, "post.wav"), r, pcm.channels, post))
        joined = body
        if len(pieces) > 1:
            joined = os.path.join(run_dir, "joined.wav")
            skill.join(pieces, joined, 0.0)
            steps.append({"tool": "ffmpeg-skill/join", "inputs": len(pieces), "transition": "none",
                          "silence": {"before": round(plan["pre"] / r, 6), "after": round(max(0, post) / r, 6)}})
        out = os.path.join(run_dir, "stinger.wav")
        if P.read_wav(joined).frames != frames:
            skill.cut(joined, out, start=0.0, duration=frames / r)
            steps.append({"tool": "ffmpeg-skill/cut", "start": 0.0, "duration": round(frames / r, 6), "accurate": True})
        else:
            shutil.copyfile(joined, out)
        res, ofacts = _validate(skill, out, r, frames, "stinger")
        got = P.peak(res)
        error = round(got["time"] - peak_at, 6)
        if abs(error) > PEAK_TOLERANCE:
            raise SoundDesignError("INVALID_RESULT", f"the stinger peaks at {got['time']:.3f}s, not {peak_at:.3f}s "
                                   f"(+/- {PEAK_TOLERANCE * 1000:g} ms)", {"peak": got, "peak_at": peak_at})
        timing = {"duration": round(res.duration, 6), "frames": res.frames, "peak_at": peak_at, "peak": got,
                  "peak_error": error, "tolerance": PEAK_TOLERANCE, "source_peak": src_peak,
                  "plan": {k: round(v / r, 6) for k, v in plan.items()}, "type": req.type}
        sha = sha256_file(out)
        provenance = build_shaping_provenance(
            op="stinger", source={"file": os.path.basename(src), "sha256": sha256_file(src), "measured": facts},
            parameters={"duration": duration, "peak_at": peak_at, "type": req.type}, result={"timing": timing},
            output_sha256=sha, ffmpeg_skill=_skill_doc(sinfo))
        _place(out, dst, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    warnings = []
    if plan["head"] > 0:
        warnings.append(f"the source's first {plan['head'] / r:.3f}s were cut away to put its peak at {peak_at:g}s")
    if fade_out:
        warnings.append(f"the source was cut {duration - peak_at:.3f}s after its peak, with a {DECLICK_OUT * 1000:g} ms fade")
    return {
        "schema": STINGER_SCHEMA, "ok": True, "op": "stinger", "skill": {"id": SKILL_ID, "version": __version__},
        "output": _output_doc(dst, res, ofacts, sha), "timing": timing, "source": provenance["source"], "engine": None,
        "steps": steps, "provenance": provenance,
        "handoff": {"to": "audio-production-skill MIX", "path": dst,
                    "note": f"place it so that it starts {peak_at:g}s before the cut: its peak then lands on the cut"},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------------------------------- seam-check

@dataclass
class SeamCheckRequest:
    input_path: str
    period: Optional[float] = None
    seams: List[float] = field(default_factory=list)
    window: float = P.SEAM_WINDOW
    ffmpeg_skill_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT


def seam_check(req: SeamCheckRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """seam-check: every seam of a WAV measured (click, level, spectrum). Read-only."""
    window = _number(req.window, "--window", 0.05, 2.0)
    timeout = _number(req.timeout, "--timeout", 0.0, 1e9, lo_open=True)
    if req.period is None and not req.seams:
        raise SoundDesignError("INVALID_ARGUMENT", "give --period (a looped file) and / or --seams")
    period = None if req.period is None else _number(req.period, "--period", 0.1, 3600.0)
    if not isinstance(req.seams, list):
        raise SoundDesignError("INVALID_ARGUMENT", "--seams must be a list of times", {"seams": req.seams})
    times = [_number(t, "--seams", 0.0, 1e6, lo_open=True) for t in req.seams]
    src = _input(req.input_path)
    skill, sinfo = usable_skill(skill, req.ffmpeg_skill_dir, timeout)
    facts = measure(skill, src)
    run_dir = tempfile.mkdtemp(prefix="sound-design-seam-")
    try:
        try:
            pcm = P.read_wav(src)
            steps: List[Dict[str, Any]] = [{"tool": "ffmpeg-skill/probe", "result": facts}]
        except SoundDesignError:
            path, steps = _as_pcm(skill, src, facts, run_dir)
            pcm = P.read_wav(path)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    r = pcm.rate
    seams = set(P.frames_for(t, r) for t in times)
    if period is not None:
        n = P.frames_for(period, r)
        seams.update(k * n for k in range(1, pcm.frames // n + 1) if k * n < pcm.frames - 1)
    if not seams:
        raise SoundDesignError("INVALID_ARGUMENT", "no seam falls inside the file", {"duration": pcm.duration})
    result = P.seam_check(pcm.mono, r, sorted(seams), window=window)
    return {
        "schema": SEAM_SCHEMA, "ok": True, "op": "seam-check", "skill": {"id": SKILL_ID, "version": __version__},
        "file": {"name": os.path.basename(src), "sha256": sha256_file(src), **P.describe(pcm)},
        "request": {"period": period, "seams": times, "window": window},
        "passed": result["passed"], "seams": result["seams"], "limits": result["limits"],
        "method": SEAM_METHOD,
        "engine": None, "steps": steps, "ffmpeg_skill": _skill_doc(sinfo),
        "warnings": [] if result["passed"] else [f"seam at {s['time']:.3f}s: " + "; ".join(s["reasons"])
                                                 for s in result["seams"] if not s["passed"]],
    }
