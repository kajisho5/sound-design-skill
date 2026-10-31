"""Shared test helpers: paths, the CLI runner, environment patching, a fake ACE-Step install (distribution metadata
and an empty `acestep` package on a temporary import path, never imported), a fake ACE-Step *engine* (the same
metadata plus a stdlib-only `acestep` package with the API surface ADR-004 verified, writing a deterministic WAV for a
seed and a duration), and a fake checkpoints directory. Nothing here is part of the package; no engine, no weights,
no GPU and no network are needed."""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_design_skill import cli  # noqa: E402
from sound_design_skill.engines.acestep import REQUIRED_FILES  # noqa: E402

# what `acestep-download` leaves in a checkpoints directory, as far as this skill and ACE-Step's own
# check_main_model_exists look: the required files plus the 1.7B LM that ACE-Step insists on (never loaded here)
CHECKPOINT_FILES = REQUIRED_FILES + ("acestep-5Hz-lm-1.7B/model.safetensors",)


def _dir_from_env(key: str, sibling: str, marker: str) -> Optional[Path]:
    for c in ([Path(os.environ[key])] if os.environ.get(key) else []) + [ROOT.parent / sibling]:
        if (c / marker).is_file():
            return c
    return None


OS_DIR = _dir_from_env("AI_VIDEO_PRODUCTION_OS_DIR", "AI-video-production-OS", "registry/conformance.py")
FFMPEG_SKILL_DIR = _dir_from_env("SOUND_DESIGN_FFMPEG_SKILL_DIR", "ffmpeg-skill", "scripts/_contract.py")


def run_cli(*argv: str) -> Tuple[int, Dict[str, Any]]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    return rc, json.loads(buf.getvalue())


@contextmanager
def env(**values: Optional[str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def no_engines_env(cache: str) -> Dict[str, Optional[str]]:
    """No engine environment and an empty Hugging Face cache, whatever this machine has installed."""
    return {"SOUND_DESIGN_ACESTEP_PYTHON": None, "SOUND_DESIGN_ACESTEP_CHECKPOINTS": None,
            "HF_HUB_CACHE": cache, "HF_HOME": cache, "XDG_CACHE_HOME": cache}


def fake_checkpoints(root: str, files: Sequence[str] = CHECKPOINT_FILES) -> str:
    """Empty weights files laid out like ACE-Step's checkpoints directory (= the HF repo layout)."""
    for rel in files:
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


def fake_hf_snapshot(cache: str, repo_id: str, files: Sequence[str] = CHECKPOINT_FILES) -> str:
    d = Path(cache) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "0123abcd"
    return fake_checkpoints(str(d), files)


# The fake engine: module and class names, signatures and dataclass fields as read from ace-step/ACE-Step-1.5 at
# ca1e85fe (ADR-004), reduced to what this skill calls. GenerationParams / GenerationConfig are dataclasses, so a field
# name that does not exist upstream fails with TypeError. Behaviour is set by acestep/fake_config.json; every call is
# appended to the "log" file named there.
FAKE_ACESTEP = {
    "__init__.py": '"""Fake ACE-Step package for sound-design-skill tests."""\n',
    "_fake.py": '''import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def config():
    with open(os.path.join(HERE, "fake_config.json"), encoding="utf-8") as fh:
        return json.load(fh)
def log(event, **data):
    path = config().get("log")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(data, event=event)) + "\\n")
''',
    "model_downloader.py": '''import os, socket
from pathlib import Path
from acestep._fake import log
MAIN_MODEL_COMPONENTS = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"]
DEFAULT_VAE_VARIANT = "official"
def get_checkpoints_dir(custom_dir=None):
    if custom_dir:
        return Path(custom_dir)
    env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(os.getcwd()) / "checkpoints"
def _contains_model_weights(model_path):
    names = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json",
             "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors.index.json",
             "diffusion_pytorch_model.bin", "diffusion_pytorch_model.bin.index.json")
    return model_path.is_dir() and any((model_path / n).exists() for n in names)
def check_main_model_exists(checkpoints_dir=None):
    d = Path(checkpoints_dir) if checkpoints_dir is not None else get_checkpoints_dir()
    return all(_contains_model_weights(d / c) for c in MAIN_MODEL_COMPONENTS)
def check_model_exists(model_name, checkpoints_dir=None):
    d = Path(checkpoints_dir) if checkpoints_dir is not None else get_checkpoints_dir()
    return bool(model_name) and _contains_model_weights(d / model_name)
def _can_access_google(timeout=3.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(("www.google.com", 443))
        return True
    except (socket.timeout, socket.error, OSError) as e:
        log("google_probe_failed", error=type(e).__name__)
        return False
    finally:
        sock.close()
def _download_from_huggingface_internal(repo_id, local_dir, token=None):
    log("download", source="huggingface", repo=repo_id)
    raise RuntimeError("fake: a real download would start here")
def _download_from_modelscope_internal(repo_id, local_dir):
    log("download", source="modelscope", repo=repo_id)
    raise RuntimeError("fake: a real download would start here")
def _smart_download(repo_id, local_dir, token=None, prefer_source=None):
    use_hf = _can_access_google() if prefer_source is None else prefer_source == "huggingface"
    first, second = ((_download_from_huggingface_internal, _download_from_modelscope_internal) if use_hf
                     else (_download_from_modelscope_internal, _download_from_huggingface_internal))
    try:
        first(repo_id, local_dir) if first is _download_from_modelscope_internal else first(repo_id, local_dir, token)
        return True, "ok"
    except Exception as e:
        try:
            second(repo_id, local_dir) if second is _download_from_modelscope_internal else second(repo_id, local_dir, token)
            return True, "ok"
        except Exception as e2:
            return False, f"both failed: {e}; {e2}"
def ensure_main_model(checkpoints_dir=None, prefer_source=None):
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    if check_main_model_exists(checkpoints_dir):
        return True, "present"
    return _smart_download("ACE-Step/Ace-Step1.5", Path(checkpoints_dir), prefer_source=prefer_source)
''',
    "handler.py": '''import os, time
from pathlib import Path
from acestep import model_downloader
from acestep._fake import config, log
class AceStepHandler:
    def __init__(self):
        self.device = "cpu"
        self.sample_rate = 48000
    def initialize_service(self, project_root, config_path, device="auto", use_flash_attention=False, compile_model=False,
                           offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None, prefer_source=None,
                           use_mlx_dit=True, vae_checkpoint=None):
        try:
            return self._initialize(project_root, config_path, device, vae_checkpoint, prefer_source)
        except Exception as exc:      # as upstream: every exception becomes (message, False)
            return f"Error initializing model: {exc}", False
    def _initialize(self, project_root, config_path, device, vae_checkpoint, prefer_source):
        cfg = config()
        env_ckpt = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
        ckpt = str(model_downloader.get_checkpoints_dir()) if env_ckpt else os.path.join(project_root, "checkpoints")
        log("initialize_service", project_root=project_root, config_path=config_path, device=device,
            vae_checkpoint=vae_checkpoint, prefer_source=prefer_source, checkpoint_dir=ckpt,
            hf_offline=os.environ.get("HF_HUB_OFFLINE"))
        if cfg.get("mode") == "force_download" or not model_downloader.check_main_model_exists(Path(ckpt)):
            ok, msg = model_downloader._smart_download("ACE-Step/Ace-Step1.5", Path(ckpt), prefer_source=prefer_source)
            if not ok:
                return f"ERROR: Failed to download main model: {msg}", False
        if cfg.get("mode") == "init_fail":
            return "Error initializing model: CUDA out of memory\\n\\nTraceback: ...", False
        if cfg.get("mode") == "sleep":
            time.sleep(60)
        self.device = "cpu" if device == "auto" else device
        return "ok", True
''',
    "inference.py": '''import array, hashlib, math, os, random, sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from acestep._fake import config, log
@dataclass
class GenerationParams:
    task_type: str = "text2music"
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "unknown"
    bpm: Optional[int] = None
    duration: float = -1.0
    inference_steps: int = 8
    seed: int = -1
    thinking: bool = True
    use_cot_metas: bool = True
    use_cot_caption: bool = True
    use_cot_lyrics: bool = False
    use_cot_language: bool = True
@dataclass
class GenerationConfig:
    batch_size: int = 2
    allow_lm_batch: bool = False
    use_random_seed: bool = True
    seeds: Optional[List[int]] = None
    lm_batch_chunk_size: int = 8
    constrained_decoding_debug: bool = False
    audio_format: str = "flac"
@dataclass
class GenerationResult:
    audios: List[Dict[str, Any]] = field(default_factory=list)
    status_message: str = ""
    extra_outputs: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
def _wav(path, seed, seconds, rate=48000):
    """Stereo 16-bit PCM, a function of (seed, seconds) only: a seeded tone plus seeded noise."""
    import wave
    rng = random.Random(seed)
    freq = 110.0 + rng.random() * 330.0
    n = int(round(seconds * rate))
    step = 2 * math.pi * freq / rate
    data = array.array("h", [0]) * (2 * n)
    for i in range(n):
        v = int(6000 * math.sin(step * i)) + rng.randint(-500, 500)
        data[2 * i] = v
        data[2 * i + 1] = -v
    if sys.byteorder == "big":
        data.byteswap()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
def generate_music(dit_handler, llm_handler, params, config_, save_dir=None, progress=None):
    cfg = config()
    log("generate_music", params=asdict(params), config=asdict(config_), llm_handler_is_none=llm_handler is None,
        save_dir=save_dir)
    if cfg.get("mode") == "fail":
        return GenerationResult(success=False, error="fake: DiT diffusion failed", status_message="failed")
    seed = config_.seeds[0] if (not config_.use_random_seed and config_.seeds) else random.randint(0, 2 ** 32 - 1)
    if cfg.get("mode") == "wrong_seed":
        seed = (seed + 1) % 2 ** 32
    seconds = params.duration - (2.0 if cfg.get("mode") == "short" else 0.0)
    os.makedirs(save_dir, exist_ok=True)
    key = hashlib.sha256(f"{params.caption}|{seed}|{params.duration}".encode("utf-8")).hexdigest()[:16]
    path = os.path.join(save_dir, f"{key}.wav")
    _wav(path, seed, seconds)
    return GenerationResult(audios=[{"path": path, "tensor": None, "key": key, "sample_rate": 48000,
                                     "params": dict(asdict(params), seed=seed)}], status_message="ok")
''',
}


@contextmanager
def fake_acestep_engine(root: str, version: str = "1.5.0", **cfg: Any) -> Iterator[Dict[str, Any]]:
    """`ace-step` metadata plus a working fake `acestep` package (FAKE_ACESTEP) in `root`, on sys.path and PYTHONPATH,
    so the worker subprocess runs it as it would run ACE-Step. Yields {"site", "config", "log", "calls"()}; change the
    behaviour with set_mode(mode) or keyword arguments (mode="ok" | "short" | "fail" | "wrong_seed" | "init_fail" |
    "force_download" | "sleep")."""
    site = Path(root) / "site"
    pkg = site / "acestep"
    pkg.mkdir(parents=True, exist_ok=True)
    for name, text in FAKE_ACESTEP.items():
        (pkg / name).write_text(text, encoding="utf-8")
    log_path = Path(root) / "fake_acestep_calls.jsonl"
    conf = dict({"mode": "ok", "log": str(log_path)}, **cfg)

    def set_mode(mode: str) -> None:
        conf["mode"] = mode
        (pkg / "fake_config.json").write_text(json.dumps(conf), encoding="utf-8")

    def calls() -> list:
        if not log_path.is_file():
            return []
        return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    set_mode(conf["mode"])
    with _on_path(site, version):
        yield {"site": str(site), "log": str(log_path), "calls": calls, "set_mode": set_mode}


@contextmanager
def fake_acestep_install(root: str, version: str = "1.5.0") -> Iterator[str]:
    """`ace-step` distribution metadata and an empty `acestep` package in `root`, on sys.path and PYTHONPATH (so the
    worker subprocess sees it too). The package is only found, never imported (it raises if it is)."""
    site = Path(root) / "site"
    (site / "acestep").mkdir(parents=True, exist_ok=True)
    (site / "acestep" / "__init__.py").write_text("raise ImportError('the skill must never import acestep')\n", encoding="utf-8")
    with _on_path(site, version):
        yield str(site)


@contextmanager
def _on_path(site: Path, version: str) -> Iterator[None]:
    meta = site / f"ace_step-{version}.dist-info"
    meta.mkdir(exist_ok=True)
    (meta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ace-step\nVersion: {version}\n", encoding="utf-8")
    sys.path.insert(0, str(site))
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site)] + ([old] if old else []))
    try:
        yield
    finally:
        sys.path.remove(str(site))
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old


def write_tone_wav(path: str, seconds: float = 1.0, rate: int = 48000, freq: float = 440.0) -> str:
    import math
    import struct
    import wave
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)))
    return path
