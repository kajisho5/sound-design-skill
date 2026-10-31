"""EzAudio (Jiarui Hai et al.), the SFX engine (ADR-005): text-to-audio for sound effects, a DiT over a 24 kHz
waveform-VAE latent with a Flan-T5 text encoder, 1-10 s. Code MIT (haidog-yaqub/EzAudio LICENSE), weights MIT
(Hugging Face OpenSound/EzAudio model card), text encoder google/flan-t5-xl Apache-2.0. No licence or card says
anything about the output; MIT does not restrict it. The training data is AudioSet, VGGSound and AudioCaps (YouTube
clips, arXiv 2409.10819 section I-II), third-party audio: recorded in every provenance (TRAINING_DATA, OUTPUT_WARNING),
and GENERATE_SFX refuses to run it until the caller passes ACK_FLAG (ADR-007, the owner's decision).

EzAudio is a git checkout, not a distribution: `api/ezaudio.py` is imported from the checkout root, and it reads its
model config relative to the working directory (`ckpts/ezaudio-xl.yml`). Looked up, never imported by doctor: the
checkout (SOUND_DESIGN_EZAUDIO_DIR) with the API this ADR read, its weights in `<checkout>/ckpts/`, the Hugging Face
cache snapshot of google/flan-t5-xl, and the Python modules EzAudio imports, in this interpreter or in EzAudio's own
environment (SOUND_DESIGN_EZAUDIO_PYTHON).

Generation (ADR-006) runs only inside the worker subprocess, through the Python API of haidog-yaqub/EzAudio at commit
86f18268 (api/ezaudio.py, src/inference.py, src/modules/stable_vae/__init__.py):

    ez = api.ezaudio.EzAudio(model_name="s3_xl", ckpt_path=<checkout>/ckpts/s3/ezaudio_s3_xl.pt,
                             vae_path=<checkout>/ckpts/vae/1m.pt, device=...)
    sr, audio = ez.generate_audio(prompt, length=<whole seconds>, random_seed=seed, randomize_seed=False)

Left alone, `EzAudio.__init__` downloads a missing checkpoint (`download_ckpt`, a urllib urlretrieve from
huggingface.co) and the text encoder comes from `T5EncoderModel.from_pretrained("google/flan-t5-xl")`. This engine
(1) checks every file first and refuses before anything is imported, (2) passes both checkpoint paths explicitly, so
`download_ckpt` is not called, (3) replaces `EzAudio.download_ckpt` with a refusal in the worker, (4) keeps the Hugging
Face libraries offline, so the T5 encoder can only come from the local cache, and (5) the worker blocks every
non-loopback socket (engines/netguard.py)."""
from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..errors import SoundDesignError
from .base import INCOMPATIBLE, OK, Engine, EngineRequest, EngineResult, EngineSpec, missing
from .lookup import python_packages, weights_dir

ENV_DIR = "SOUND_DESIGN_EZAUDIO_DIR"
ENV_PYTHON = "SOUND_DESIGN_EZAUDIO_PYTHON"
MODEL_NAME = "s3_xl"                             # api/ezaudio.py `configs` key; the README's usage example
CONFIG_FILE = "ckpts/ezaudio-xl.yml"             # configs["s3_xl"]["config"], opened relative to the working directory
CKPT_FILE = "ckpts/s3/ezaudio_s3_xl.pt"          # configs["s3_xl"]["path"]
VAE_FILE = "ckpts/vae/1m.pt"                     # configs["vae"]["path"]; load_vae reads config.json beside it
VAE_CONFIG = "ckpts/vae/config.json"
CHECKOUT_FILES = ("api/ezaudio.py", "src/inference.py", CONFIG_FILE, VAE_CONFIG)
WEIGHT_FILES = (CKPT_FILE, VAE_FILE)
T5_REPO = "google/flan-t5-xl"                    # ckpts/ezaudio-xl.yml text_encoder.model
T5_FILES = ("config.json", "spiece.model", "model.safetensors.index.json", "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors")
MODULES = ("torch", "transformers", "diffusers", "soundfile", "yaml", "librosa", "einops")
DISTS = ("torch", "transformers", "diffusers")
# what this adapter calls, as written in api/ezaudio.py at API_COMMIT; a checkout without them is INCOMPATIBLE
API_MARKERS = ("class EzAudio:", "def __init__(self, model_name, ckpt_path=None, vae_path=None, device=",
               "def download_ckpt(self, model_dict):", "def generate_audio(self, text, length=10,",
               "random_seed=None, randomize_seed=False")
API_COMMIT = "86f18268117dde73bb274401c610e639e4a8fae1"
API_SOURCE = f"haidog-yaqub/EzAudio@{API_COMMIT}"
DEVICES = ("auto", "cuda", "cpu")                # EzAudio passes `device` to torch .to() / torch.Generator(device=)
SAMPLE_RATE = 24000                              # ckpts/ezaudio-xl.yml autoencoder.sr
LATENT_RATE = 50                                 # autoencoder.latent_sr: generate_audio's length * 50 latent frames
MAX_PROMPT = 512
T5_MAX_TOKENS = 100                              # text_encoder.max_length; src/inference.py tokenizes with truncation=True
# generate_audio's defaults at API_COMMIT, passed explicitly so a changed default cannot change the result silently
GENERATION_DEFAULTS = {"guidance_scale": 5, "guidance_rescale": 0.75, "ddim_steps": 100, "eta": 1}
TRAINING_DATA = ("AudioSet, VGGSound and AudioCaps (arXiv 2409.10819: DiT stages and the waveform VAE); these "
                 "datasets are lists of YouTube clips whose audio belongs to the uploaders, not licensed by them for "
                 "training (risk, not verified either way)")
OUTPUT_WARNING = ("EzAudio was trained on YouTube audio (AudioSet / VGGSound / AudioCaps); its licences (MIT) do not "
                  "restrict the output, but the rights in the training audio were not licensed: listen to the effect "
                  "before publishing (ADR-005)")
# The owner's decision (ADR-007): EzAudio never runs until the caller says the user was told about its training data.
ACK_FLAG = "--acknowledge-ezaudio-training-data"
CONFIRMATION_REASON = ("EzAudio was trained on AudioSet, VGGSound and AudioCaps, YouTube clips whose audio was not "
                       "licensed for training by its owners. Its MIT licence does not restrict the output, but the "
                       "owner decided to be told before every use of EzAudio (ADR-007).")
CONFIRMATION_REASON_JA = ("EzAudio は AudioSet・VGGSound・AudioCaps で学習されています。これらは YouTube の音声で、"
                          "持ち主から学習用の許諾を得ていません。MIT ライセンスは出力を制限しませんが、EzAudio を使う前に"
                          "毎回ユーザーに伝えると決めています（ADR-007）。")

SPEC = EngineSpec(
    id="ezaudio",
    ops=("sfx",),
    description="EzAudio-XL (DiT over a 24 kHz waveform-VAE latent, Flan-T5-XL text encoder): text-to-audio sound effects, 1-10 s, mono",
    kind="python",
    code_license="MIT (haidog-yaqub/EzAudio LICENSE, Copyright (c) 2025 Jiarui Hai)",
    weights_license="MIT (Hugging Face OpenSound/EzAudio model card: DiT checkpoints and the waveform VAE); text "
                    "encoder google/flan-t5-xl Apache-2.0",
    output_use="no statement on the output in the licence, model card or README; MIT places no restriction on use, "
               "commercial use included, and no licence read restricts the output",
    training_data=TRAINING_DATA,
    credit=None,
    credit_required=False,
    watermark=None,
    duration_range=(1.0, 10.0),
    devices=("cuda", "cpu"),
    min_vram_gb=None,
    isolation="worker subprocess under EzAudio's own environment (SOUND_DESIGN_EZAUDIO_PYTHON): torch, diffusers, "
              "transformers, librosa and its other requirements.txt entries are unpinned; this package never imports it",
    version_range=f"a haidog-yaqub/EzAudio checkout exposing the API read at {API_COMMIT[:8]} (capability detection "
                  "on api/ezaudio.py; EzAudio publishes no version number)",
    install=(f"git clone https://github.com/haidog-yaqub/EzAudio ~/EzAudio && python -m venv ~/EzAudio/.venv && "
             f"~/EzAudio/.venv/bin/pip install -r ~/EzAudio/requirements.txt (plus torch for your platform), then fetch "
             f"the weights once: huggingface-cli download OpenSound/EzAudio {CKPT_FILE} {VAE_FILE} --local-dir ~/EzAudio "
             f"and huggingface-cli download {T5_REPO} (into the Hugging Face cache); set {ENV_DIR}=~/EzAudio and "
             f"{ENV_PYTHON}=~/EzAudio/.venv/bin/python"),
    located_by=(f"${ENV_DIR} (the checkout, weights in its ckpts/)", f"${ENV_PYTHON} (else this Python)",
                f"the Hugging Face cache ({T5_REPO})"),
    roadmap_step=3,
    licence_checked="2026-10-31 (docs/decisions.md ADR-005)",
    output_warning=OUTPUT_WARNING,
    confirmation_flag=ACK_FLAG,
    confirmation_reason=CONFIRMATION_REASON,
    confirmation_reason_ja=CONFIRMATION_REASON_JA,
    licence_notes=(
        "EzAudio downloads a missing checkpoint itself (EzAudio.download_ckpt); this skill passes both checkpoint "
        "paths, replaces download_ckpt with a refusal and blocks every non-loopback socket in the worker (ADR-006)",
        "the model card and README say nothing about commercial use of the output; the licences (MIT, Apache-2.0) "
        "do not restrict it",
        "the work was done while the first author was an intern at Tencent AI Lab (paper footnote); the weights are "
        "published under MIT by the OpenSound organisation",
        "speed, memory use and seed determinism on a real install are unverified (no install here)",
    ),
    sources=("https://github.com/haidog-yaqub/EzAudio", "https://github.com/haidog-yaqub/EzAudio/blob/main/LICENSE",
             "https://huggingface.co/OpenSound/EzAudio", "https://huggingface.co/google/flan-t5-xl",
             "https://arxiv.org/abs/2409.10819"),
)


class EzAudioEngine(Engine):
    spec = SPEC

    def inspect(self) -> Dict[str, Any]:
        root = checkout_dir()
        if root is None:
            return missing(f"{ENV_DIR} is not set (the EzAudio checkout)", "package", interpreter=None)
        absent = [f for f in CHECKOUT_FILES if not (root / f).is_file()]
        if absent:
            return missing(f"${ENV_DIR} is not an EzAudio checkout: no {', '.join(absent)}", "package", interpreter=None)
        commit = checkout_commit(root)
        gaps = api_gaps(root)
        if gaps:
            return {"status": INCOMPATIBLE, "version": commit, "model": None, "interpreter": None,
                    "detail": f"api/ezaudio.py lacks the API this skill calls ({API_SOURCE}): {'; '.join(gaps)}"}
        found = python_packages(list(DISTS), list(MODULES), ENV_PYTHON)
        where = found["interpreter"]
        if found["error"]:
            return dict(missing(found["error"], "package", interpreter=where), version=commit)
        lacking = [m for m in MODULES if not found["modules"].get(m)]
        if lacking:
            return dict(missing(f"EzAudio's requirements are not installed in {where}: {', '.join(lacking)}", "package",
                                interpreter=where), version=commit)
        no_weights = [f for f in WEIGHT_FILES if not (root / f).is_file()]
        if no_weights:
            return dict(missing(f"weights missing in ${ENV_DIR}: {', '.join(no_weights)} (EzAudio would download them)",
                                "model", interpreter=where), version=commit)
        t5 = weights_dir(T5_REPO, T5_FILES)
        if t5["dir"] is None:
            return dict(missing(f"text encoder {T5_REPO} missing in the {t5['source']}: {', '.join(t5['missing'])}", "model",
                                interpreter=where), version=commit)
        return {"status": OK, "version": commit, "detail": f"EzAudio checkout {commit or 'unknown commit'}, {MODEL_NAME}, "
                f"{T5_REPO} ({t5['source']})", "model": MODEL_NAME, "interpreter": where,
                "packages": {d: found["dists"].get(d) for d in DISTS}}

    def validate(self, req: EngineRequest) -> None:
        super().validate(req)
        # generate_audio computes `length * latent_sr` latent frames and hands it to torch.randn as a size, which
        # takes integers only: whole seconds (src/inference.py, api/ezaudio.py at API_COMMIT)
        if float(req.duration) != int(req.duration):
            raise SoundDesignError("INVALID_ARGUMENT", f"duration must be a whole number of seconds for {self.id!r} (EzAudio "
                                   "generates length * 50 latent frames)", {"engine": self.id, "duration": req.duration})

    def process(self, req: EngineRequest) -> EngineResult:
        """Runs in the worker only (under SOUND_DESIGN_EZAUDIO_PYTHON). Checks first, imports EzAudio last."""
        self.validate(req)
        prompt = req.prompt.strip()
        if len(prompt) > MAX_PROMPT:
            raise SoundDesignError("INVALID_ARGUMENT", f"the prompt is {len(prompt)} characters; this skill takes < {MAX_PROMPT}",
                                   {"engine": self.id, "length": len(prompt)})
        if req.lyrics is not None:
            raise SoundDesignError("INVALID_ARGUMENT", "EzAudio takes no lyrics", {"engine": self.id})
        device = req.params.get("device", "auto")
        if device not in DEVICES:
            raise SoundDesignError("INVALID_ARGUMENT", f"device must be one of {', '.join(DEVICES)}", {"device": device})
        info = self.require()
        root = checkout_dir()
        assert root is not None                  # require() passed
        root = root.resolve()
        seed = req.seed if req.seed is not None else random.SystemRandom().randrange(2 ** 32)
        length = int(req.duration)
        output = os.path.abspath(req.output_path)

        # EzAudio imports `src.*` from the checkout root and opens ckpts/ezaudio-xl.yml relative to the working directory
        os.chdir(str(root))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if device == "auto":
            device = "cuda" if importlib.import_module("torch").cuda.is_available() else "cpu"
        api = importlib.import_module("api.ezaudio")
        attempts = refuse_downloads(api)
        try:
            model = api.EzAudio(model_name=MODEL_NAME, ckpt_path=str(root / CKPT_FILE), vae_path=str(root / VAE_FILE),
                                device=device)
        except Exception as exc:
            if attempts:
                raise SoundDesignError("MODEL_NOT_FOUND", f"EzAudio tried to download ({attempts[0]}); refused, nothing was run",
                                       {"engine": self.id, "install": self.spec.install, "attempts": attempts})
            raise SoundDesignError("ENGINE_FAILED", f"EzAudio failed to load: {type(exc).__name__}: {_tail(str(exc))}",
                                   {"engine": self.id})
        if attempts:
            raise SoundDesignError("MODEL_NOT_FOUND", f"EzAudio tried to download ({attempts[0]}); refused, nothing was run",
                                   {"engine": self.id, "install": self.spec.install, "attempts": attempts})
        call = dict(GENERATION_DEFAULTS, length=length, random_seed=seed, randomize_seed=False)
        try:
            sr, audio = model.generate_audio(prompt, **call)
        except Exception as exc:
            raise SoundDesignError("ENGINE_FAILED", f"EzAudio generate_audio failed: {type(exc).__name__}: {_tail(str(exc))}",
                                   {"engine": self.id})
        if int(sr) != SAMPLE_RATE:
            raise SoundDesignError("INVALID_RESULT", f"EzAudio returned {sr} Hz, expected {SAMPLE_RATE}", {"engine": self.id})
        # the README's own way of saving the result (soundfile); 16-bit PCM, mono, never re-encoded afterwards
        importlib.import_module("soundfile").write(output, audio, int(sr), subtype="PCM_16")
        if not os.path.isfile(output):
            raise SoundDesignError("INVALID_RESULT", "EzAudio's audio could not be written", {"engine": self.id})
        return EngineResult(output_path=req.output_path, engine_version=str(info.get("version") or "unknown"), model=MODEL_NAME,
                            sample_rate=int(sr), seed=seed, duration=float(length),
                            extra={"device": device, "requested_device": req.params.get("device", "auto"), "api": API_SOURCE,
                                   "generate_audio": call, "text_encoder": T5_REPO, "t5_max_tokens": T5_MAX_TOKENS,
                                   "subtype": "PCM_16"})


def checkout_dir() -> Optional[Path]:
    explicit = os.environ.get(ENV_DIR)
    return Path(explicit) if explicit else None


def checkout_commit(root: Path) -> Optional[str]:
    """The checkout's HEAD commit, read from .git without running git; None when it cannot be read."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head[4:].strip()
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip() or None
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        return None
    return None


def api_gaps(root: Path) -> List[str]:
    try:
        text = (root / "api" / "ezaudio.py").read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [str(e)]
    return [f"no {m!r}" for m in API_MARKERS if m not in text]


def refuse_downloads(api: Any) -> List[str]:
    """Replace EzAudio.download_ckpt (api/ezaudio.py, the only place it fetches a checkpoint) with one that refuses
    and records the attempt. Both paths are passed explicitly, so it is never called unless upstream changes."""
    attempts: List[str] = []

    def refuse(name: str) -> Callable[..., Any]:
        def _refused(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{name}{tuple(str(a) for a in args[1:])}")
            raise RuntimeError(f"sound-design-skill: {name} refused (no download at run time)")
        return _refused

    if hasattr(api, "EzAudio") and hasattr(api.EzAudio, "download_ckpt"):
        api.EzAudio.download_ckpt = refuse("EzAudio.download_ckpt")
    return attempts


def _tail(text: str, lines: int = 3) -> str:
    rows = [r for r in str(text).strip().splitlines() if r.strip()]
    return " | ".join(rows[-lines:]) if rows else "no message"
