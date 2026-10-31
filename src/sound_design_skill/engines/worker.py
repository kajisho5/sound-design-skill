"""Engine worker: `python -m sound_design_skill.engines.worker <request.json> <result.json>`.

The service starts this as a child process in its own process group, so a timeout kills a long generation for real
and an engine crash (CUDA out of memory, a segfault in a native extension) cannot take the caller down
(transcription-skill ADR-003, voice-enhance-skill ADR-002, dubbing-skill ADR-002). The argv is fixed; only two file
paths in a private temporary directory vary. Before any engine import the Hugging Face libraries are switched offline
and every non-loopback socket connect and name lookup is refused (netguard; ADR-004), so neither Hugging Face nor
ModelScope can be reached from here, whatever library tries. Errors go to the result file as structured JSON, never as
free text on stdout.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import List, Optional

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


def main(argv: Optional[List[str]] = None) -> int:
    os.environ.update(OFFLINE_ENV)
    from . import netguard
    netguard.install()
    from ..errors import SoundDesignError
    from . import get_engine
    from .base import EngineRequest

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write("usage: worker <request.json> <result.json>\n")
        return 2
    req_path, out_path = args
    try:
        with open(req_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        engine = get_engine(str(doc["engine_id"]))
        result = engine.process(EngineRequest.from_dict(doc["request"]))
        payload = {"ok": True, "result": result.to_dict()}
        code = 0
    except SoundDesignError as exc:
        payload = {"ok": False, "error": exc.to_dict()}
        code = 1
    except Exception as exc:  # anything else is an engine failure, reported structurally
        payload = {"ok": False, "error": {"code": "ENGINE_FAILED", "message": f"{type(exc).__name__}: {exc}", "retryable": False,
                                          "details": {"traceback": traceback.format_exc().splitlines()[-6:]}}}
        code = 1
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True)
    os.replace(tmp, out_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
