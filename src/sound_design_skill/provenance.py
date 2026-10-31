"""Provenance block every generated file and every op result carries (ADR-001, OS ADR-012): what ran, on which prompt
and seed, with which licences, and the output-use, credit and watermark facts the published video inherits. Recorded
in the ProductionReceipt."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional

from . import SKILL_ID, __version__

ENGINE_FIELDS = ("id", "version", "model", "code_license", "weights_license", "output_use", "credit", "credit_required",
                 "watermark", "execution_mode", "deterministic", "lm_status", "licence_checked")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_provenance(*, op: str, engine: Dict[str, Any], prompt: str, seed: Optional[int], duration: Dict[str, Any],
                     output_path: Optional[str], parameters: Dict[str, Any],
                     ffmpeg_skill: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`engine` is an engine report (Engine.report()); only publishable fields are copied (no local paths).
    `duration` is {"requested": s, "measured": s | None}. The prompt is kept as text (it is the creative brief, not
    personal data) and hashed."""
    return {
        "skill": {"id": SKILL_ID, "version": __version__},
        "op": op,
        "engine": {k: engine.get(k) for k in ENGINE_FIELDS},
        "prompt": {"text": prompt, "sha256": sha256_text(prompt)},
        "seed": seed,
        "duration": dict(duration),
        "output": {"sha256": sha256_file(output_path)} if output_path else None,
        "parameters": dict(parameters),
        "ffmpeg_skill": None if ffmpeg_skill is None else {k: ffmpeg_skill.get(k) for k in ("version", "contract_version")},
        "network": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
