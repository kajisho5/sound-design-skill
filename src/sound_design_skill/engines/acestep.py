"""ACE-Step 1.5 (ACE Studio and StepFun), the BGM engine (OS ADR-012, ADR-001 here): text-to-music, 10 s to 10 min,
instrumental or with lyrics. Code MIT (ace-step/ACE-Step-1.5 LICENSE), weights MIT (Hugging Face ACE-Step/Ace-Step1.5
model card, which also states the output may be used commercially). The text encoder shipped in the same repository,
Qwen3-Embedding-0.6B, is Apache-2.0.

Looked up, never imported by doctor: the `ace-step` distribution (1.5.x, installed from the git checkout; the PyPI
project of the same name is an unrelated 0.1.0 release) in this interpreter or, as recommended, in ACE-Step's own
environment (SOUND_DESIGN_ACESTEP_PYTHON, usually <checkout>/.venv/bin/python), and the checkpoints in a directory
(SOUND_DESIGN_ACESTEP_CHECKPOINTS) or the Hugging Face cache of ACE-Step/Ace-Step1.5.

Generation (ADR-004) runs only inside the worker subprocess, under ACE-Step's interpreter, through the Python API of
ace-step/ACE-Step-1.5 at commit ca1e85fe (docs/en/INFERENCE.md, acestep/inference.py, acestep/handler.py):

    dit = acestep.handler.AceStepHandler()
    dit.initialize_service(project_root=..., config_path="acestep-v15-turbo", device=..., vae_checkpoint="official")
    acestep.inference.generate_music(dit, None, GenerationParams(...), GenerationConfig(batch_size=1,
                                     use_random_seed=False, seeds=[seed], audio_format="wav"), save_dir=...)

`initialize_service` downloads whatever `acestep.model_downloader.check_main_model_exists` /
`check_model_exists` find missing, from Hugging Face or, when a socket to www.google.com fails, from ModelScope
(`_smart_download`). The Hugging Face offline variables do not stop the ModelScope path, so this engine (1) checks the
same components with the same rule before anything is imported, (2) points ACE-Step at that local directory with
ACESTEP_CHECKPOINTS_DIR, (3) replaces ACE-Step's download functions in the worker with ones that refuse, and (4) the
worker blocks every non-loopback socket (engines/netguard.py). The 5 Hz language model is never loaded: it is off
until the Qwen3 base-model licences are verified (LM_STATUS)."""
from __future__ import annotations

import importlib
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..errors import SoundDesignError
from .base import INCOMPATIBLE, OK, Engine, EngineRequest, EngineResult, EngineSpec, in_range, missing, parse_version
from .lookup import python_packages, weights_dir

ENV_PYTHON = "SOUND_DESIGN_ACESTEP_PYTHON"
ENV_CHECKPOINTS = "SOUND_DESIGN_ACESTEP_CHECKPOINTS"
DIST = "ace-step"
MODULE = "acestep"
HF_REPO = "ACE-Step/Ace-Step1.5"
DIT_MODEL = "acestep-v15-turbo"
REQUIRED_FILES = (f"{DIT_MODEL}/config.json", f"{DIT_MODEL}/model.safetensors", "vae/diffusion_pytorch_model.safetensors",
                  "Qwen3-Embedding-0.6B/model.safetensors")
LM_MODELS = ("acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B")
# acestep/model_downloader.py MAIN_MODEL_COMPONENTS (ca1e85fe): initialize_service downloads the main repository when
# any of these has no weights file, the 1.7B LM included, although this skill never loads it.
MAIN_MODEL_COMPONENTS = (DIT_MODEL, "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
# acestep/model_downloader.py _contains_model_weights: a component "exists" when one of these files is in its directory
WEIGHT_FILENAMES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json",
                    "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors.index.json",
                    "diffusion_pytorch_model.bin", "diffusion_pytorch_model.bin.index.json")
VAE_VARIANT = "official"                         # model_downloader.DEFAULT_VAE_VARIANT: the VAE inside the main repo
DEVICES = ("auto", "cuda", "mps", "xpu", "cpu")  # values initialize_service resolves (_resolve_initialize_device)
SAMPLE_RATE = 48000                              # AceStepHandler.sample_rate
MAX_CAPTION = 512                                # GenerationParams docstring: caption < 512 characters
MAX_LYRICS = 4096                                # GenerationParams docstring: lyrics < 4096 characters
INSTRUMENTAL_LYRICS = "[Instrumental]"           # GenerationParams docstring: "[Instrumental]" for instrumental songs
LM_STATUS = ("off: the 5 Hz LM (fine-tuned from Qwen3-0.6B / 1.7B / 4B) is never loaded until the Qwen3 base-model "
             "licences are verified (docs/decisions.md ADR-001 follow-up, ADR-004); generation is DiT-only")
API_SOURCE = "ace-step/ACE-Step-1.5@ca1e85fe9430179831e6bc6be790c332190a3866"
SUPPORTED = ((1, 5, 0), (1, 6, 0))               # >=1.5.0 <1.6.0: the release this ADR checked (pyproject 1.5.0)

SPEC = EngineSpec(
    id="acestep",
    ops=("bgm",),
    description="ACE-Step 1.5 (2B turbo DiT + optional 5 Hz LM planner): text-to-music, instrumental beds or songs, 10 s to 10 min",
    kind="python",
    code_license="MIT (ace-step/ACE-Step-1.5 LICENSE; bundled nano-vllm MIT)",
    weights_license="MIT (Hugging Face ACE-Step/Ace-Step1.5 and the acestep-v15-* / acestep-5Hz-lm-* model cards); "
                    "bundled text encoder Qwen3-Embedding-0.6B Apache-2.0",
    output_use="commercial use of generated music stated by the model card (\"You can strictly use the generated music "
               "for commercial purposes\"); the README asks users to check originality and disclose AI involvement",
    training_data="model card claim: licensed tracks, royalty-free / public-domain music and MIDI-rendered synthetic "
                  "audio; the dataset itself is not published (unverified)",
    credit=None,
    credit_required=False,
    watermark=None,
    duration_range=(10.0, 600.0),
    devices=("cuda", "mps", "rocm", "xpu", "cpu"),
    min_vram_gb=4.0,
    isolation="worker subprocess under ACE-Step's own environment (SOUND_DESIGN_ACESTEP_PYTHON): it pins torch builds "
              "per platform, gradio, transformers <4.58 and needs Python 3.11-3.12; this package never imports it",
    version_range=">=1.5.0 <1.6.0",
    install=(f"git clone https://github.com/ace-step/ACE-Step-1.5 ~/ACE-Step-1.5 && cd ~/ACE-Step-1.5 && uv sync "
             f"(Python 3.11-3.12), then fetch the weights once: uv run acestep-download (or huggingface-cli download "
             f"{HF_REPO} --local-dir ~/ACE-Step-1.5/checkpoints); set {ENV_PYTHON}=~/ACE-Step-1.5/.venv/bin/python and "
             f"{ENV_CHECKPOINTS}=~/ACE-Step-1.5/checkpoints"),
    located_by=(f"${ENV_PYTHON} (else this Python)", f"${ENV_CHECKPOINTS} (else the Hugging Face cache, {HF_REPO})"),
    roadmap_step=2,
    lm_status=LM_STATUS,
    licence_notes=(
        "ACE-Step downloads models on first run by itself (Hugging Face, else ModelScope); this skill never lets it: the "
        "worker refuses before the checkpoints are complete, replaces ACE-Step's download functions and blocks every "
        "non-loopback socket (ADR-004)",
        "the PyPI project `ace-step` (0.1.0, Apache-2.0) is not this release; install from the git checkout",
        "the README disclaimer is guidance, not a licence condition: verify originality, disclose AI involvement",
        "XL (4B DiT) models need >= 12 GB VRAM; not the default here",
    ),
    sources=("https://github.com/ace-step/ACE-Step-1.5", "https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE",
             "https://github.com/ace-step/ACE-Step-1.5/blob/main/pyproject.toml",
             "https://huggingface.co/ACE-Step/Ace-Step1.5", "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
             "https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md"),
)


class AceStepEngine(Engine):
    spec = SPEC

    def inspect(self) -> Dict[str, Any]:
        found = python_packages([DIST], [MODULE], ENV_PYTHON)
        where = found["interpreter"]
        if found["error"]:
            return missing(found["error"], "package", interpreter=where)
        vstr = found["dists"].get(DIST)
        if not vstr or not found["modules"].get(MODULE):
            return missing(f"{DIST} is not installed in {where}", "package", interpreter=where)
        version = parse_version(vstr)
        if version is None or not in_range(version, *SUPPORTED):
            return {"status": INCOMPATIBLE, "version": vstr, "detail": f"{DIST} {vstr} is outside {SPEC.version_range}",
                    "model": None, "interpreter": where}
        weights = weights_dir(HF_REPO, REQUIRED_FILES, ENV_CHECKPOINTS)
        if weights["dir"] is None:
            return dict(missing(f"weights missing in {weights['source']}: {', '.join(weights['missing'])}", "model",
                                interpreter=where), version=vstr)
        absent = missing_components(weights["dir"])
        if absent:
            return dict(missing(f"ACE-Step would download the main model: no weights for {', '.join(absent)} in "
                                f"{weights['source']} (initialize_service needs every one of {', '.join(MAIN_MODEL_COMPONENTS)})",
                                "model", interpreter=where), version=vstr)
        lms: List[str] = [m for m in LM_MODELS if has_weights(weights["dir"] / m)]
        return {"status": OK, "version": vstr, "detail": f"{DIST} {vstr}, {DIT_MODEL} ({weights['source']})", "model": DIT_MODEL,
                "lm_models": lms, "lm_note": "present on disk, not used: " + LM_STATUS, "interpreter": where}

    def process(self, req: EngineRequest) -> EngineResult:
        """Runs in the worker only (under SOUND_DESIGN_ACESTEP_PYTHON). Checks first, imports ACE-Step last."""
        self.validate(req)
        caption = req.prompt.strip()
        if len(caption) > MAX_CAPTION:
            raise SoundDesignError("INVALID_ARGUMENT", f"the prompt is {len(caption)} characters; ACE-Step takes < {MAX_CAPTION}",
                                   {"engine": self.id, "length": len(caption)})
        if req.lyrics is not None and (not req.lyrics.strip() or len(req.lyrics) > MAX_LYRICS):
            raise SoundDesignError("INVALID_ARGUMENT", f"lyrics must be non-empty and < {MAX_LYRICS} characters",
                                   {"engine": self.id})
        device = req.params.get("device", "auto")
        if device not in DEVICES:
            raise SoundDesignError("INVALID_ARGUMENT", f"device must be one of {', '.join(DEVICES)}", {"device": device})
        info = self.require()
        weights = weights_dir(HF_REPO, REQUIRED_FILES, ENV_CHECKPOINTS)
        ckpt: Path = weights["dir"]
        seed = req.seed if req.seed is not None else random.SystemRandom().randrange(2 ** 32)
        out_dir = os.path.join(os.path.dirname(os.path.abspath(req.output_path)), "acestep-out")

        # ACE-Step reads its checkpoints from ACESTEP_CHECKPOINTS_DIR first (initialize_service, get_checkpoints_dir).
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt)
        os.environ.pop("ACESTEP_VAE_CHECKPOINT", None)
        downloader = importlib.import_module("acestep.model_downloader")
        attempts = refuse_downloads(downloader)
        if not (downloader.check_main_model_exists(ckpt) and downloader.check_model_exists(DIT_MODEL, ckpt)):
            raise SoundDesignError("MODEL_NOT_FOUND", f"ACE-Step's own check finds the checkpoints in {weights['source']} "
                                   "incomplete; it would download them, so nothing was run", {"engine": self.id,
                                                                                               "install": self.spec.install})
        handler_mod = importlib.import_module("acestep.handler")
        inference = importlib.import_module("acestep.inference")
        dit = handler_mod.AceStepHandler()
        try:
            status, ok = dit.initialize_service(project_root=str(ckpt.parent), config_path=DIT_MODEL, device=device,
                                                vae_checkpoint=VAE_VARIANT)
        except Exception as exc:     # upstream returns (message, False); a raise is treated the same way
            status, ok = f"{type(exc).__name__}: {exc}", False
        if attempts:
            raise SoundDesignError("MODEL_NOT_FOUND", f"ACE-Step tried to download ({attempts[0]}); refused, nothing was run",
                                   {"engine": self.id, "install": self.spec.install, "attempts": attempts})
        if not ok:
            raise SoundDesignError("ENGINE_FAILED", "ACE-Step initialize_service failed: " + _tail(status), {"engine": self.id})
        gen = generation_kwargs(caption, req.lyrics, float(req.duration), seed)
        params = inference.GenerationParams(**gen)
        config = inference.GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        result = inference.generate_music(dit, None, params, config, save_dir=out_dir)
        if attempts:
            raise SoundDesignError("MODEL_NOT_FOUND", f"ACE-Step tried to download ({attempts[0]}); refused",
                                   {"engine": self.id, "attempts": attempts})
        if not getattr(result, "success", False):
            raise SoundDesignError("ENGINE_FAILED", "ACE-Step generate_music failed: " + _tail(str(getattr(result, "error", ""))),
                                   {"engine": self.id})
        audios = list(getattr(result, "audios", None) or [])
        path = str(audios[0].get("path") or "") if len(audios) == 1 else ""
        if not path or not os.path.isfile(path):
            raise SoundDesignError("INVALID_RESULT", f"ACE-Step returned {len(audios)} audio item(s) and no saved file",
                                   {"engine": self.id})
        used_seed = (audios[0].get("params") or {}).get("seed")
        if used_seed != seed:
            raise SoundDesignError("INVALID_RESULT", f"ACE-Step used seed {used_seed!r}, not the requested {seed}",
                                   {"engine": self.id, "seed": seed, "used_seed": used_seed})
        os.replace(path, req.output_path)
        return EngineResult(output_path=req.output_path, engine_version=str(info.get("version")), model=DIT_MODEL,
                            sample_rate=int(audios[0].get("sample_rate") or SAMPLE_RATE), seed=seed, duration=float(req.duration),
                            extra={"device": getattr(dit, "device", device), "requested_device": device,
                                   "lm": {"enabled": False, "status": LM_STATUS}, "api": API_SOURCE,
                                   "generation_params": gen,
                                   "generation_config": {"batch_size": 1, "use_random_seed": False, "seeds": [seed],
                                                         "audio_format": "wav"},
                                   "vae": VAE_VARIANT})


def has_weights(directory: Path) -> bool:
    """acestep/model_downloader.py _contains_model_weights, without importing it."""
    return directory.is_dir() and any((directory / f).exists() for f in WEIGHT_FILENAMES)


def missing_components(ckpt: Path) -> List[str]:
    return [c for c in MAIN_MODEL_COMPONENTS if not has_weights(ckpt / c)]


def generation_kwargs(caption: str, lyrics: Optional[str], duration: float, seed: int) -> Dict[str, Any]:
    """GenerationParams for a text2music bed, DiT only: no LM chain-of-thought (thinking and every use_cot_* off)."""
    return {"task_type": "text2music", "caption": caption, "lyrics": INSTRUMENTAL_LYRICS if lyrics is None else lyrics,
            "instrumental": lyrics is None, "duration": duration, "seed": seed, "thinking": False, "use_cot_metas": False,
            "use_cot_caption": False, "use_cot_language": False}


def refuse_downloads(downloader: Any) -> List[str]:
    """Replace ACE-Step's download entry points (acestep/model_downloader.py) with ones that refuse and record the
    attempt. ensure_main_model / ensure_dit_model / ensure_vae_model reach the network only through these."""
    attempts: List[str] = []

    def refuse(name: str) -> Callable[..., Any]:
        def _refused(*args: Any, **kwargs: Any) -> Any:
            attempts.append(f"{name}{tuple(str(a) for a in args)}")
            raise RuntimeError(f"sound-design-skill: {name} refused (no Hugging Face or ModelScope access at run time)")
        return _refused

    for name in ("_smart_download", "_download_from_huggingface_internal", "_download_from_modelscope_internal"):
        if hasattr(downloader, name):
            setattr(downloader, name, refuse(name))
    if hasattr(downloader, "_can_access_google"):
        setattr(downloader, "_can_access_google", lambda *a, **k: False)
    return attempts


def _tail(text: str, lines: int = 3) -> str:
    rows = [r for r in str(text).strip().splitlines() if r.strip()]
    return " | ".join(rows[-lines:]) if rows else "no message"
