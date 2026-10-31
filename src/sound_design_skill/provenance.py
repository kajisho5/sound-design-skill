"""Provenance block every generated file and every op result carries (ADR-001, OS ADR-012): what ran, on which prompt
and seed, with which licences, and the output-use, credit and watermark facts the published video inherits. Recorded
in the ProductionReceipt."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional

from . import SKILL_ID, __version__

ENGINE_FIELDS = ("id", "version", "model", "code_license", "weights_license", "output_use", "credit", "credit_required",
                 "watermark", "execution_mode", "deterministic", "lm_status", "training_data", "output_warning", "confirmation_required",
                 "licence_checked")


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
                     ffmpeg_skill: Optional[Dict[str, Any]] = None,
                     confirmation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`engine` is an engine report (Engine.report()); only publishable fields are copied (no local paths).
    `duration` is {"requested": s, "measured": s | None}. The prompt is kept as text (it is the creative brief, not
    personal data) and hashed. `confirmation` records that the user acknowledged an engine that needs it, and when
    (ADR-007); None when the engine needs none."""
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
        "confirmation": None if confirmation is None else dict(confirmation),
        "network": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_library_provenance(*, source: Dict[str, Any], licence: Dict[str, Any], selection: Dict[str, Any],
                             fit: Dict[str, Any], duration: Dict[str, Any], output_path: str,
                             ffmpeg_skill: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Provenance of a library-mode output (ADR-008): which user-supplied file it came from (library-relative path and
    SHA-256, never an absolute path), the licence exactly as the user declared it (never verified or cleared here),
    the selection rule and query, and how it was fitted. No engine ran: `engine` is None and `ai_generated` false."""
    return {
        "skill": {"id": SKILL_ID, "version": __version__},
        "op": "library",
        "engine": None,
        "ai_generated": False,
        "source": dict(source),
        "licence": dict(licence),
        "selection": dict(selection),
        "fit": dict(fit),
        "duration": dict(duration),
        "output": {"sha256": sha256_file(output_path)},
        "ffmpeg_skill": None if ffmpeg_skill is None else {k: ffmpeg_skill.get(k) for k in ("version", "contract_version")},
        "network": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
