"""Availability lookups shared by the engines. None of them imports an engine, downloads anything or leaves the
machine:

- python_packages: distributions / modules in this interpreter, or in the engine's own virtual environment named by
  an environment variable (ADR-001: ACE-Step 1.5 pins its own torch build and needs Python 3.11-3.12, so a separate
  environment, usually its own checkout's `.venv`, is the recommended install).
- weights_dir: a checkpoints directory the user already filled (explicit env var), else a Hugging Face cache
  snapshot of the model repository; it must hold every required file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..procs import run_process_group
from . import pkgprobe


def python_packages(dists: Sequence[str], modules: Sequence[str], interpreter_env: Optional[str] = None,
                    timeout: float = 30.0) -> Dict[str, Any]:
    """{"python", "dists": {name: version|None}, "modules": {name: bool}, "interpreter", "error"}. With
    `interpreter_env` set in the environment, the lookup runs under that interpreter (`-I`, fixed argv)."""
    explicit = os.environ.get(interpreter_env) if interpreter_env else None
    if not explicit:
        out = pkgprobe.probe(list(dists), list(modules))
        out.update(interpreter="this Python", error=None)
        return out
    empty = {"python": None, "dists": {d: None for d in dists}, "modules": {m: False for m in modules}, "interpreter": explicit}
    if not (os.path.isfile(explicit) and os.access(explicit, os.X_OK)):
        return dict(empty, error=f"{interpreter_env}={explicit} is not an executable file")
    code, out, err = run_process_group([explicit, "-I", pkgprobe.__file__, ",".join(dists), ",".join(modules)], timeout)
    try:
        doc = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    except ValueError:
        doc = None
    if code != 0 or not isinstance(doc, dict):
        tail = err.strip().splitlines()[-1] if err.strip() else "no output"
        return dict(empty, error=f"{interpreter_env}: package lookup failed (exit {code}): {tail}")
    doc.update(interpreter=explicit, error=None)
    return doc


def hf_cache_dirs() -> List[Path]:
    out: List[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        out.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    out.append(Path.home() / ".cache" / "huggingface" / "hub")
    return out


def weights_dir(repo_id: str, required: Sequence[str], dir_env: Optional[str] = None) -> Dict[str, Any]:
    """Where the weights are: {"dir": Path|None, "source": str, "missing": [relative paths]}.

    An explicit directory (`dir_env`) is never replaced by the cache: if it lacks a file, the answer is that file.
    Otherwise the first Hugging Face cache snapshot of `repo_id` holding every required file wins."""
    explicit = os.environ.get(dir_env) if dir_env else None
    if explicit:
        root = Path(explicit)
        missing = [r for r in required if not (root / r).is_file()]
        return {"dir": None if missing else root, "source": f"${dir_env}", "missing": missing}
    folder = "models--" + repo_id.replace("/", "--")
    best: Optional[List[str]] = None
    for cache in hf_cache_dirs():
        for snap in sorted((cache / folder / "snapshots").glob("*")):
            missing = [r for r in required if not (snap / r).is_file()]
            if not missing:
                return {"dir": snap, "source": "Hugging Face cache", "missing": []}
            if best is None or len(missing) < len(best):
                best = missing
    return {"dir": None, "source": "Hugging Face cache", "missing": list(required) if best is None else best}
