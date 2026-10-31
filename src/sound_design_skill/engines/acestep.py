"""ACE-Step 1.5 (ACE Studio and StepFun), the BGM engine (OS ADR-012, ADR-001 here): text-to-music, 10 s to 10 min,
instrumental or with lyrics. Code MIT (ace-step/ACE-Step-1.5 LICENSE), weights MIT (Hugging Face ACE-Step/Ace-Step1.5
model card, which also states the output may be used commercially). The text encoder shipped in the same repository,
Qwen3-Embedding-0.6B, is Apache-2.0.

Looked up, never imported: the `ace-step` distribution (1.5.x, installed from the git checkout; the PyPI project of
the same name is an unrelated 0.1.0 release) in this interpreter or, as recommended, in ACE-Step's own environment
(SOUND_DESIGN_ACESTEP_PYTHON, usually <checkout>/.venv/bin/python), and the turbo DiT, VAE and text encoder in a
checkpoints directory (SOUND_DESIGN_ACESTEP_CHECKPOINTS) or the Hugging Face cache of ACE-Step/Ace-Step1.5. The
5 Hz language model is optional (ACE-Step runs DiT-only on <= 6 GB GPUs) and only reported."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import INCOMPATIBLE, OK, Engine, EngineSpec, in_range, missing, parse_version
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
    licence_notes=(
        "ACE-Step downloads models on first run by itself; this skill never lets it: the worker forces the Hugging Face "
        "libraries offline and refuses to run before doctor finds the weights",
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
        lms: List[str] = [m for m in LM_MODELS if (weights["dir"] / m / "model.safetensors").is_file()]
        return {"status": OK, "version": vstr, "detail": f"{DIST} {vstr}, {DIT_MODEL} ({weights['source']})", "model": DIT_MODEL,
                "lm_models": lms, "lm_note": None if lms else "no 5 Hz LM found: DiT-only mode (ACE-Step's choice for <= 6 GB GPUs)",
                "interpreter": where}
