"""Engine contract. An engine turns one request (a text prompt, a target duration, a seed) into one WAV file; probing,
duration checks, provenance and handing the file to audio-production-skill MIX are the service's job (through
ffmpeg-skill), so every engine yields the same kind of result.

Every engine carries an EngineSpec: the static facts ADR-001 verified (code and weights licences, what the model card
says about commercial use of the output and about the training data, credit and watermark obligations, the devices
and memory it needs, how it is isolated). The spec is what the contract publishes and what goes into the provenance
of every generated file.

Statuses (doctor vocabulary): OK (installed and usable), MISSING (not installed, or its weights are absent),
INCOMPATIBLE (installed, but a version outside the supported range). An engine never downloads anything."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..errors import SoundDesignError

OK, MISSING, INCOMPATIBLE = "OK", "MISSING", "INCOMPATIBLE"
STATUSES = (OK, MISSING, INCOMPATIBLE)
OPS = ("bgm", "sfx")
KINDS = ("python",)


@dataclass(frozen=True)
class EngineSpec:
    """Static, publishable facts about one engine (no local paths). Licences are as checked in docs/decisions.md
    ADR-001 on `licence_checked`; they must be re-checked before the supported version range moves."""
    id: str
    ops: Tuple[str, ...]
    description: str
    kind: str                                 # "python": run in the worker, ideally under the engine's own interpreter
    code_license: str
    weights_license: str
    output_use: str                           # what the licence / model card says about using the generated audio
    training_data: str                        # what the model card says it was trained on (a claim, not a check)
    credit: Optional[str]                     # text the published output must carry; None = no credit obligation
    credit_required: bool
    watermark: Optional[str]                  # watermark the engine embeds in every file; None = none
    duration_range: Tuple[float, float]       # seconds the engine can generate (min, max)
    devices: Tuple[str, ...]                  # accelerators the engine documents ("cuda", "mps", "rocm", "xpu", "cpu")
    min_vram_gb: Optional[float]              # smallest documented GPU memory for the default model
    isolation: str                            # how the engine is kept apart from this MIT package
    version_range: str
    install: str
    located_by: Tuple[str, ...]
    roadmap_step: int                         # the step that implements process()
    licence_checked: str = "2026-10-31 (docs/decisions.md ADR-001)"
    licence_notes: Tuple[str, ...] = ()
    sources: Tuple[str, ...] = ()
    execution_mode: str = "local"
    network_required: bool = False            # at run time; no engine registered here needs it
    deterministic: bool = False               # the seed is fixed and recorded; bit-exact repeats on the real engine are unverified (ADR-004)
    lm_status: str = ""                       # whether a language-model stage is used, and why (ADR-004)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


@dataclass
class EngineRequest:
    op: str
    prompt: str                               # caption / tags describing the music or effect
    duration: float                           # target length in seconds
    output_path: str
    seed: Optional[int] = None                # None = the engine picks one and reports it
    lyrics: Optional[str] = None              # None = instrumental (a BGM bed has no vocals by default)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineRequest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EngineResult:
    output_path: str
    engine_version: str
    model: Optional[str]
    sample_rate: int
    seed: int
    duration: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineResult":
        return cls(output_path=str(d["output_path"]), engine_version=str(d["engine_version"]),
                   model=None if d.get("model") is None else str(d["model"]), sample_rate=int(d["sample_rate"]),
                   seed=int(d["seed"]), duration=float(d["duration"]), extra=dict(d.get("extra") or {}))


def parse_version(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not text:
        return None
    import re
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return tuple(int(x) for x in m.groups() if x is not None)


def in_range(version: Tuple[int, ...], lo: Tuple[int, ...], hi: Tuple[int, ...]) -> bool:
    v = (tuple(version) + (0, 0, 0))[:3]
    return (tuple(lo) + (0, 0, 0))[:3] <= v < (tuple(hi) + (0, 0, 0))[:3]


class Engine:
    """Subclass, set `spec`, implement `inspect()`, and register in `engines/__init__.py`. `process` may be slow and
    runs in the worker process."""
    spec: EngineSpec

    @property
    def id(self) -> str:
        return self.spec.id

    def inspect(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def report(self, offline: bool = False) -> Dict[str, Any]:
        """inspect() plus the static facts; with offline=True an engine that needs the network is MISSING."""
        info = dict(self.spec.to_dict(), **self.inspect())
        if offline and self.spec.network_required and info["status"] == OK:
            info.update(status=MISSING, detail="needs the network at run time; --offline was given")
        info["usable_offline"] = not self.spec.network_required
        if info["status"] == OK:
            info["install"] = None
        return info

    def require(self) -> Dict[str, Any]:
        info = self.inspect()
        if info["status"] != OK:
            code = "MODEL_NOT_FOUND" if info.get("missing") == "model" else "ENGINE_UNAVAILABLE"
            raise SoundDesignError(code, f"engine {self.id!r} is {info['status']}: {info.get('detail')}",
                                   {"engine": self.id, "install": self.spec.install})
        return info

    def validate(self, req: EngineRequest) -> None:
        """Request checks that need no engine: the op, the prompt, the duration window and the seed."""
        if req.op not in self.spec.ops:
            raise SoundDesignError("INVALID_ARGUMENT", f"engine {self.id!r} does not serve op {req.op!r}",
                                   {"engine": self.id, "ops": list(self.spec.ops)})
        if not isinstance(req.prompt, str) or not req.prompt.strip():
            raise SoundDesignError("INVALID_ARGUMENT", "prompt must be a non-empty string", {"engine": self.id})
        lo, hi = self.spec.duration_range
        if isinstance(req.duration, bool) or not isinstance(req.duration, (int, float)) or not lo <= req.duration <= hi:
            raise SoundDesignError("INVALID_ARGUMENT", f"duration must be between {lo:g} and {hi:g} seconds for {self.id!r}",
                                   {"engine": self.id, "duration": req.duration, "range": [lo, hi]})
        if req.seed is not None and (isinstance(req.seed, bool) or not isinstance(req.seed, int) or not 0 <= req.seed < 2 ** 32):
            raise SoundDesignError("INVALID_ARGUMENT", "seed must be an integer in [0, 2**32)", {"engine": self.id, "seed": req.seed})

    def process(self, req: EngineRequest) -> EngineResult:
        """Overridden by engines that generate (ACE-Step since roadmap step 2); the default is NOT_IMPLEMENTED."""
        self.validate(req)
        self.require()
        raise SoundDesignError("NOT_IMPLEMENTED", f"engine {self.id!r} is installed, but generation is not implemented yet "
                               f"(roadmap step {self.spec.roadmap_step})", {"engine": self.id, "op": req.op,
                                                                            "roadmap_step": self.spec.roadmap_step})


def missing(detail: str, what: str, **extra: Any) -> Dict[str, Any]:
    return dict({"status": MISSING, "version": None, "missing": what, "detail": detail, "model": None}, **extra)
