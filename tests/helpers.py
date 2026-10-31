"""Shared test helpers: paths, the CLI runner, environment patching, a fake ACE-Step install (distribution metadata
and an empty `acestep` package on a temporary import path, never imported) and a fake checkpoints directory. Nothing
here is part of the package; no engine, no weights, no GPU and no network are needed."""
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


def fake_checkpoints(root: str, files: Sequence[str] = REQUIRED_FILES) -> str:
    """Empty weights files laid out like ACE-Step's checkpoints directory (= the HF repo layout)."""
    for rel in files:
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


def fake_hf_snapshot(cache: str, repo_id: str, files: Sequence[str] = REQUIRED_FILES) -> str:
    d = Path(cache) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "0123abcd"
    return fake_checkpoints(str(d), files)


@contextmanager
def fake_acestep_install(root: str, version: str = "1.5.0") -> Iterator[str]:
    """`ace-step` distribution metadata and an empty `acestep` package in `root`, on sys.path and PYTHONPATH (so the
    worker subprocess sees it too). The package is only found, never imported (it raises if it is)."""
    site = Path(root) / "site"
    (site / "acestep").mkdir(parents=True, exist_ok=True)
    (site / "acestep" / "__init__.py").write_text("raise ImportError('the skill must never import acestep')\n", encoding="utf-8")
    meta = site / f"ace_step-{version}.dist-info"
    meta.mkdir(exist_ok=True)
    (meta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ace-step\nVersion: {version}\n", encoding="utf-8")
    sys.path.insert(0, str(site))
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site)] + ([old] if old else []))
    try:
        yield str(site)
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
