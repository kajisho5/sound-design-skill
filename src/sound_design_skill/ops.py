"""The ops. GENERATE_BGM (roadmap step 2, ADR-004): a music bed from a prompt and / or tags, a target duration and a
seed, written as a WAV with its provenance. GENERATE_SFX (step 3, ADR-006): the same frame for a sound effect, on
EzAudio (ADR-005). Library mode (step 4, ADR-008) is library.py: no engine, the user's own files.

GENERATE_BGM, step by step (the frame of voice-enhance-skill's ops: checks first, the engine in the worker, the result
validated through ffmpeg-skill, then one rename):
  1. arguments  the prompt (text and tags joined) is non-empty and < 512 characters; duration inside the engine's
                range; seed in [0, 2**32) or omitted; device one of auto / cuda / mps / xpu / cpu; a lyrics file, when
                given, is a readable UTF-8 file of < 4096 characters; --timeout > 0
  2. output     a .wav that is not a directory, does not exist (unless --overwrite), is not the lyrics file, and whose
                directory exists
  3. engine     the selected engine does BGM and reports OK (else ENGINE_UNAVAILABLE / MODEL_NOT_FOUND)
  4. ffmpeg     ffmpeg-skill is located and declares probe
  5. seed       an omitted seed is drawn here, so the engine always runs with a fixed seed and the provenance records
                the seed that was used
  6. engine     the worker runs the engine on the request in a private run directory inside the output's own
                directory, killed with its process group on --timeout
  7. validate   ffmpeg-skill/probe on the WAV: readable, one audio stream at the engine's sample rate, duration within
                max(0.1 s, 1 %) of the request (else INVALID_RESULT)
  8. place      the WAV is renamed into place (it replaces nothing unless --overwrite); the run directory is removed
Nothing is written to the output path unless every step passes.

GENERATE_SFX runs the same eight steps with these differences: no lyrics; the prompt is < 512 characters (EzAudio's
T5 encoder reads at most 100 tokens and the result warns when the prompt is certainly longer); the duration is a
whole number of seconds inside the engine's range (EzAudio: 1-10 s); the device is auto / cuda / cpu; the engine's
output warning (EzAudio: its training audio) is copied into every result. Between steps 2 and 3 an engine that needs
the user's confirmation (EzAudio, ADR-007) is refused with CONFIRMATION_REQUIRED unless the caller passed its flag
(--acknowledge-ezaudio-training-data); the acknowledgement and its time go into the provenance."""
from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import SKILL_ID, __version__
from .engines import DEFAULT_ENGINE, confirmation_for, engines_for, get_engine
from .engines import ezaudio
from .engines.acestep import DEVICES, MAX_CAPTION, MAX_LYRICS
from .engines.base import OK, Engine, EngineRequest
from .errors import SoundDesignError
from .ffmpeg_skill import FfmpegSkill, SkillInfo
from .provenance import build_provenance, sha256_text
from .service import DEFAULT_TIMEOUT, run_engine

BGM_SCHEMA = f"{SKILL_ID}/generate-bgm@1"
SFX_SCHEMA = f"{SKILL_ID}/generate-sfx@1"
DURATION_TOLERANCE = 0.1             # seconds, or 1 % of the requested duration when that is larger
OUTPUT_NAME = "bgm.wav"
SFX_OUTPUT_NAME = "sfx.wav"
SFX_DEVICES = ezaudio.DEVICES
SFX_MAX_PROMPT = ezaudio.MAX_PROMPT
SFX_TIMEOUT = 600.0


@dataclass
class BgmRequest:
    output_path: str
    duration: float
    prompt: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    lyrics_path: Optional[str] = None
    engine: Optional[str] = None
    device: str = "auto"
    timeout: float = DEFAULT_TIMEOUT
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    offline: bool = False


@dataclass
class SfxRequest:
    output_path: str
    duration: float
    prompt: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    engine: Optional[str] = None
    device: str = "auto"
    timeout: float = SFX_TIMEOUT
    overwrite: bool = False
    ffmpeg_skill_dir: Optional[str] = None
    offline: bool = False
    acknowledge_ezaudio_training_data: bool = False   # --acknowledge-ezaudio-training-data (ADR-007)


def caption_from(prompt: Optional[str], tags: List[str]) -> str:
    """The engine's caption: the prompt text, then the tags, comma-separated (ACE-Step takes one caption string)."""
    parts = [prompt.strip()] if isinstance(prompt, str) and prompt.strip() else []
    parts += [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    return ", ".join(parts)


def tolerance_for(duration: float) -> float:
    return max(DURATION_TOLERANCE, 0.01 * duration)


def _check_common(req: Any, what: str, max_caption: int, devices: Tuple[str, ...]) -> str:
    """Timeout, tags, caption, seed and device, shared by the ops. Returns the caption. Pure checks."""
    if isinstance(req.timeout, bool) or not isinstance(req.timeout, (int, float)) or req.timeout <= 0:
        raise SoundDesignError("INVALID_ARGUMENT", "--timeout must be > 0", {"timeout": req.timeout})
    if not isinstance(req.tags, list) or not all(isinstance(t, str) for t in req.tags):
        raise SoundDesignError("INVALID_ARGUMENT", "tags must be a list of strings", {"tags": req.tags})
    caption = caption_from(req.prompt, req.tags)
    if not caption:
        raise SoundDesignError("INVALID_ARGUMENT", f"give --prompt and / or --tags: the {what} needs a description")
    if len(caption) > max_caption:
        raise SoundDesignError("INVALID_ARGUMENT", f"prompt and tags are {len(caption)} characters together; the engine takes "
                               f"< {max_caption}", {"length": len(caption)})
    if req.seed is not None and (isinstance(req.seed, bool) or not isinstance(req.seed, int) or not 0 <= req.seed < 2 ** 32):
        raise SoundDesignError("INVALID_ARGUMENT", "--seed must be an integer in [0, 2**32)", {"seed": req.seed})
    if req.device not in devices:
        raise SoundDesignError("INVALID_ARGUMENT", f"--device must be one of {', '.join(devices)}", {"device": req.device})
    return caption


def _check_args(req: BgmRequest) -> Tuple[str, Optional[str]]:
    """Returns (caption, lyrics text or None). Pure checks; nothing is written."""
    caption = _check_common(req, "music", MAX_CAPTION, DEVICES)
    lyrics = None
    if req.lyrics_path is not None:
        if not os.path.isfile(req.lyrics_path):
            raise SoundDesignError("INPUT_NOT_FOUND", f"{req.lyrics_path} is not a file", {"lyrics": req.lyrics_path})
        try:
            with open(req.lyrics_path, encoding="utf-8") as fh:
                lyrics = fh.read()
        except (OSError, UnicodeDecodeError) as e:
            raise SoundDesignError("INVALID_ARGUMENT", f"cannot read the lyrics as UTF-8 text: {e}", {"lyrics": req.lyrics_path})
        if not lyrics.strip() or len(lyrics) > MAX_LYRICS:
            raise SoundDesignError("INVALID_ARGUMENT", f"the lyrics must be non-empty and < {MAX_LYRICS} characters",
                                   {"lyrics": req.lyrics_path, "length": len(lyrics)})
    return caption, lyrics


def check_output(output_path: str, overwrite: bool, inputs: Tuple[Optional[str], ...] = ()) -> str:
    """Returns the absolute output path. Pure checks on the file system; nothing is written."""
    dst = os.path.abspath(output_path)
    if not dst.lower().endswith(".wav"):
        raise SoundDesignError("INVALID_ARGUMENT", "the output must be a .wav file (the engine writes PCM WAV; it is not "
                               "re-encoded)", {"output": output_path})
    for src in inputs:
        if src and (os.path.realpath(src) == os.path.realpath(dst) or (os.path.exists(dst) and os.path.samefile(dst, src))):
            raise SoundDesignError("OUTPUT_IS_INPUT", "the output path is an input; inputs are never modified", {"output": output_path})
    if os.path.isdir(dst):
        raise SoundDesignError("INVALID_ARGUMENT", f"{output_path} is a directory", {"output": output_path})
    if os.path.lexists(dst) and not overwrite:
        raise SoundDesignError("OUTPUT_EXISTS", f"{output_path} exists; pass --overwrite to replace it", {"output": output_path})
    if not os.path.isdir(os.path.dirname(dst)):
        raise SoundDesignError("INVALID_ARGUMENT", f"the output directory {os.path.dirname(dst)} does not exist", {"output": output_path})
    return dst


def select_engine(op: str, engine_id: Optional[str], offline: bool = False) -> Tuple[Engine, Dict[str, Any]]:
    """The engine for `op` and its report; refuses an engine that is not OK before any work."""
    eid = engine_id or DEFAULT_ENGINE.get(op)
    if eid is None:
        raise SoundDesignError("ENGINE_UNAVAILABLE", f"no engine is registered for {op}", {"op": op})
    engine = get_engine(eid)
    if op not in engine.spec.ops:
        raise SoundDesignError("INVALID_ARGUMENT", f"engine {eid!r} does not do {op}; engines for {op}: {engines_for(op)}",
                               {"engine": eid, "op": op, "engines": engines_for(op)})
    report = engine.report(offline=offline)
    if report["status"] != OK:
        code = "MODEL_NOT_FOUND" if report.get("missing") == "model" else "ENGINE_UNAVAILABLE"
        raise SoundDesignError(code, f"engine {eid!r} is {report['status']}: {report.get('detail')}",
                               {"engine": eid, "status": report["status"], "install": engine.spec.install})
    return engine, report


def require_confirmation(op: str, engine_id: Optional[str], acknowledged: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """ADR-007: an engine whose spec names a confirmation flag runs only when the caller passed that flag, i.e. the
    user was told first. Refuses with CONFIRMATION_REQUIRED, in English and Japanese, before any engine inspection or
    ffmpeg-skill call. Returns the acknowledgement record for the provenance, or None when nothing was needed."""
    need = confirmation_for(op, engine_id)
    if need is None:
        return None
    flag = need["flag"]
    if flag not in acknowledged:
        en = f"{need['reason']} Tell the user, then re-run with {flag} to generate with the {need['engine']} engine."
        ja = f"{need['reason_ja']}ユーザーに伝えてから、{flag} を付けて再実行してください。"
        raise SoundDesignError("CONFIRMATION_REQUIRED", f"{en}\n{ja}",
                               {"engine": need["engine"], "op": op, "flag": flag, "message_en": en, "message_ja": ja,
                                "training_data": get_engine(need["engine"]).spec.training_data, "approval": "CONFIRM",
                                "adr": need["adr"]})
    return {"required": True, "engine": need["engine"], "flag": flag, "acknowledged": True,
            "acknowledged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": need["reason"],
            "adr": need["adr"]}


def usable_skill(skill: Optional[FfmpegSkill], directory: Optional[str], timeout: float) -> Tuple[FfmpegSkill, SkillInfo]:
    if skill is None:
        skill = FfmpegSkill.locate(directory, timeout=timeout)
    info = skill.info()
    if not info.supported:
        raise SoundDesignError("FFMPEG_SKILL_UNAVAILABLE", "ffmpeg-skill is not usable: " + "; ".join(info.problems),
                               {"directory": str(info.directory), "problems": info.problems})
    return skill, info


def validate_output(skill: FfmpegSkill, path: str, rate: int, requested: float, engine_id: str) -> Dict[str, Any]:
    """The engine's WAV through ffmpeg-skill/probe: readable, exactly one audio stream at `rate` Hz, no picture, and as
    long as requested within the tolerance."""
    try:
        facts = skill.probe(path)
    except SoundDesignError as e:
        raise SoundDesignError("INVALID_RESULT", f"engine {engine_id!r} wrote a file ffmpeg-skill cannot read: {e.message}",
                               {"engine": engine_id})
    a = facts.get("audio") if isinstance(facts.get("audio"), dict) else {}
    problems = []
    if not a:
        problems.append("no audio stream")
    counts = facts.get("stream_counts") if isinstance(facts.get("stream_counts"), dict) else {}
    if int(counts.get("audio") or (1 if a else 0)) != 1 or facts.get("video"):
        problems.append(f"streams {counts or 'unknown'}, expected one audio stream and no video")
    if a and int(a.get("sample_rate") or 0) != rate:
        problems.append(f"sample rate {a.get('sample_rate')}, expected {rate}")
    if a and int(a.get("channels") or 0) < 1:
        problems.append(f"{a.get('channels')} channels")
    measured = float(facts.get("duration") or 0.0)
    tol = tolerance_for(requested)
    if abs(measured - requested) > tol:
        problems.append(f"duration {measured:.3f}s, requested {requested:.3f}s (+/- {tol:.3f}s)")
    if problems:
        raise SoundDesignError("INVALID_RESULT", f"engine {engine_id!r} output failed validation: " + "; ".join(problems),
                               {"engine": engine_id, "problems": problems})
    return facts


def _run_dir(directory: str, output_path: str) -> str:
    """A private run directory in the output's own directory, so the result is placed with one rename."""
    try:
        return tempfile.mkdtemp(prefix=".sound-design-", dir=directory)
    except OSError as e:
        raise SoundDesignError("INVALID_ARGUMENT", f"cannot write to the output directory: {e}", {"output": output_path})


def _place(path: str, dst: str, overwrite: bool) -> None:
    if os.path.lexists(dst) and not overwrite:      # appeared while the engine ran
        raise SoundDesignError("OUTPUT_EXISTS", f"{dst} appeared while the engine ran; pass --overwrite to replace it", {"output": dst})
    os.replace(path, dst)


def _run_validated(engine: Engine, request: EngineRequest, timeout: float, run_dir: str,
                   skill: FfmpegSkill) -> Tuple[Any, Dict[str, Any]]:
    """Steps 6-7: the worker writes `request.output_path` in `run_dir`; the file must be the one requested, made with
    the requested seed, and pass ffmpeg-skill/probe. Returns (EngineResult, probe facts)."""
    wav = request.output_path
    result = run_engine(engine.id, request, timeout, run_dir)
    if os.path.realpath(result.output_path) != os.path.realpath(wav) or not os.path.isfile(wav):
        raise SoundDesignError("INVALID_RESULT", f"engine {engine.id!r} did not write the requested file", {"engine": engine.id})
    if result.seed != request.seed:
        raise SoundDesignError("INVALID_RESULT", f"engine {engine.id!r} ran with seed {result.seed}, not {request.seed}",
                               {"engine": engine.id, "seed": request.seed, "used_seed": result.seed})
    facts = validate_output(skill, wav, result.sample_rate, float(request.duration), engine.id)
    return result, facts


def generate_bgm(req: BgmRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """GENERATE_BGM: one music bed as a WAV at `req.output_path`, with its provenance. `skill` is the ffmpeg-skill
    handle (located from `req.ffmpeg_skill_dir` when None)."""
    caption, lyrics = _check_args(req)
    dst = check_output(req.output_path, req.overwrite, (req.lyrics_path,))
    engine, report = select_engine("bgm", req.engine, req.offline)
    duration = req.duration
    probe_req = EngineRequest(op="bgm", prompt=caption, duration=duration, output_path=dst, seed=req.seed, lyrics=lyrics)
    engine.validate(probe_req)                       # the engine's duration window, before any process starts
    duration = float(duration)
    skill, info = usable_skill(skill, req.ffmpeg_skill_dir, float(req.timeout))
    seed_source = "given" if req.seed is not None else "drawn"
    seed = req.seed if req.seed is not None else secrets.randbelow(2 ** 32)
    run_dir = _run_dir(os.path.dirname(dst), req.output_path)
    try:
        wav = os.path.join(run_dir, OUTPUT_NAME)
        result, facts = _run_validated(engine, EngineRequest(op="bgm", prompt=caption, duration=duration, output_path=wav,
                                                             seed=seed, lyrics=lyrics, params={"device": req.device}),
                                       float(req.timeout), run_dir, skill)
        measured = float(facts.get("duration") or 0.0)
        engine_doc = dict(report, version=result.engine_version, model=result.model)
        parameters = {"tags": [t.strip() for t in req.tags if t.strip()], "prompt_given": caption_from(req.prompt, []) or None,
                      "seed_source": seed_source, "instrumental": lyrics is None,
                      "lyrics_sha256": None if lyrics is None else sha256_text(lyrics),
                      "device": result.extra.get("device"), "requested_device": req.device, "lm": result.extra.get("lm"),
                      "sample_rate": result.sample_rate, "engine_api": result.extra.get("api"),
                      "generation_params": result.extra.get("generation_params"),
                      "generation_config": result.extra.get("generation_config"), "timeout": float(req.timeout)}
        provenance = build_provenance(op="bgm", engine=engine_doc, prompt=caption, seed=seed,
                                      duration={"requested": duration, "measured": measured,
                                                "tolerance": tolerance_for(duration)},
                                      output_path=wav, parameters=parameters, ffmpeg_skill=info.to_dict())
        _place(wav, dst, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    a = facts.get("audio") or {}
    return {
        "schema": BGM_SCHEMA,
        "ok": True,
        "op": "bgm",
        "capability": "audio.bgm_generate",
        "skill": {"id": SKILL_ID, "version": __version__},
        "output": {"path": dst, "format": "wav", "codec": a.get("codec"), "sample_rate": a.get("sample_rate"),
                   "channels": a.get("channels"), "duration": measured, "sha256": provenance["output"]["sha256"]},
        "request": {"prompt": caption, "duration": duration, "seed": seed, "seed_source": seed_source},
        "engine": {k: engine_doc.get(k) for k in ("id", "version", "model", "code_license", "weights_license", "output_use",
                                                   "lm_status")},
        "steps": [{"tool": f"{SKILL_ID}/worker", "engine": engine.id}, {"tool": "ffmpeg-skill/probe", "result":
                  {"duration": measured, "sample_rate": a.get("sample_rate"), "channels": a.get("channels")}}],
        "provenance": provenance,
        "handoff": {"to": "audio-production-skill MIX", "path": dst, "note": "placed under the programme there; nothing is "
                                                                              "mixed here"},
        "warnings": [],
    }


def generate_sfx(req: SfxRequest, skill: Optional[FfmpegSkill] = None) -> Dict[str, Any]:
    """GENERATE_SFX (ADR-006): one sound effect as a WAV at `req.output_path`, with its provenance. Same frame as
    GENERATE_BGM: every check first, the engine in the worker, ffmpeg-skill/probe, one rename."""
    caption = _check_common(req, "effect", SFX_MAX_PROMPT, SFX_DEVICES)
    dst = check_output(req.output_path, req.overwrite)
    confirmation = require_confirmation("sfx", req.engine,
                                        (ezaudio.ACK_FLAG,) if req.acknowledge_ezaudio_training_data is True else ())
    engine, report = select_engine("sfx", req.engine, req.offline)
    engine.validate(EngineRequest(op="sfx", prompt=caption, duration=req.duration, output_path=dst, seed=req.seed))
    duration = float(req.duration)
    skill, info = usable_skill(skill, req.ffmpeg_skill_dir, float(req.timeout))
    seed_source = "given" if req.seed is not None else "drawn"
    seed = req.seed if req.seed is not None else secrets.randbelow(2 ** 32)
    warnings: List[str] = []
    if engine.spec.output_warning:
        warnings.append(engine.spec.output_warning)
    if len(caption.split()) > ezaudio.T5_MAX_TOKENS:
        warnings.append(f"the prompt has more than {ezaudio.T5_MAX_TOKENS} words; EzAudio's text encoder reads at most "
                        f"{ezaudio.T5_MAX_TOKENS} tokens and drops the rest")
    run_dir = _run_dir(os.path.dirname(dst), req.output_path)
    try:
        wav = os.path.join(run_dir, SFX_OUTPUT_NAME)
        result, facts = _run_validated(engine, EngineRequest(op="sfx", prompt=caption, duration=duration, output_path=wav,
                                                             seed=seed, params={"device": req.device}),
                                       float(req.timeout), run_dir, skill)
        measured = float(facts.get("duration") or 0.0)
        engine_doc = dict(report, version=result.engine_version, model=result.model)
        parameters = {"tags": [t.strip() for t in req.tags if t.strip()], "prompt_given": caption_from(req.prompt, []) or None,
                      "seed_source": seed_source, "device": result.extra.get("device"), "requested_device": req.device,
                      "sample_rate": result.sample_rate, "engine_api": result.extra.get("api"),
                      "generate_audio": result.extra.get("generate_audio"), "text_encoder": result.extra.get("text_encoder"),
                      "timeout": float(req.timeout)}
        provenance = build_provenance(op="sfx", engine=engine_doc, prompt=caption, seed=seed,
                                      duration={"requested": duration, "measured": measured,
                                                "tolerance": tolerance_for(duration)},
                                      output_path=wav, parameters=parameters, ffmpeg_skill=info.to_dict(),
                                      confirmation=confirmation)
        _place(wav, dst, req.overwrite)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    a = facts.get("audio") or {}
    return {
        "schema": SFX_SCHEMA,
        "ok": True,
        "op": "sfx",
        "capability": "audio.sfx_generate",
        "skill": {"id": SKILL_ID, "version": __version__},
        "output": {"path": dst, "format": "wav", "codec": a.get("codec"), "sample_rate": a.get("sample_rate"),
                   "channels": a.get("channels"), "duration": measured, "sha256": provenance["output"]["sha256"]},
        "request": {"prompt": caption, "duration": duration, "seed": seed, "seed_source": seed_source},
        "engine": {k: engine_doc.get(k) for k in ("id", "version", "model", "code_license", "weights_license", "output_use",
                                                   "training_data")},
        "steps": [{"tool": f"{SKILL_ID}/worker", "engine": engine.id}, {"tool": "ffmpeg-skill/probe", "result":
                  {"duration": measured, "sample_rate": a.get("sample_rate"), "channels": a.get("channels")}}],
        "confirmation": confirmation,
        "provenance": provenance,
        "handoff": {"to": "audio-production-skill MIX", "path": dst, "note": "placed in the programme there; nothing is "
                                                                              "mixed here"},
        "warnings": warnings,
    }
