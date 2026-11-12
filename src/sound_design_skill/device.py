"""Compute device and GPU memory, for doctor (roadmap step 1: "doctor reports device/VRAM"). Never imports torch or
an engine: NVIDIA GPUs are read with `nvidia-smi --query-gpu` (a fixed list argv, its own process group, a short
timeout), Apple Silicon is recognised from the platform and its unified memory read with `sysctl -n hw.memsize`,
and ROCm and Intel XPU are only noticed (`rocm-smi` / `xpu-smi` on PATH), not measured. What torch would actually pick inside ACE-Step's own
environment is not checked here; ACE-Step does its own GPU detection when it starts.

The tier is ACE-Step 1.5's own recommendation table (README, "Your GPU VRAM"), so the agent can choose a model
before anything is loaded. vram_fit compares the measured memory with a model's requirement (ADR-016)."""
from __future__ import annotations

import os
import platform
import shutil
from typing import Any, Dict, List, Optional

from .procs import TIMEOUT_EXIT, run_process_group

NVIDIA_QUERY = ["--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]
# GPUs report slightly less than their nominal size (driver and system reservations); ACE-Step's gpu_config allows
# 0.5 GB for this (VRAM_16GB_TOLERANCE_GB at ca1e85fe), and a model requirement here is compared with the same margin
VRAM_TOLERANCE_GB = 0.5
# vram_fit statuses: OK, INSUFFICIENT (measured, too little), NO_GPU (CPU only), UNKNOWN (a GPU whose memory was not
# measured), NOT_CHECKED (no requirement, or no device document). A model with vram_check "refuse" is refused on these:
FIT_STATUSES = ("OK", "INSUFFICIENT", "NO_GPU", "UNKNOWN", "NOT_CHECKED")
REFUSED_FITS = ("INSUFFICIENT", "NO_GPU")
# (upper bound in GB, recommendation) from the ACE-Step 1.5 README table; checked 2026-10-31 (ADR-001)
ACESTEP_TIERS = (
    (6.0, "<= 6 GB: 2B turbo DiT only, no LM (INT8 quantization + CPU offload)"),
    (8.0, "6-8 GB: 2B turbo + acestep-5Hz-lm-0.6B (pt backend)"),
    (16.0, "8-16 GB: 2B turbo/sft + acestep-5Hz-lm-0.6B (8-12 GB) or 1.7B (12-16 GB)"),
    (20.0, "16-20 GB: 2B sft or XL turbo (XL with CPU offload) + acestep-5Hz-lm-1.7B"),
    (24.0, "20-24 GB: XL turbo/sft + acestep-5Hz-lm-1.7B"),
    (float("inf"), ">= 24 GB: XL sft + acestep-5Hz-lm-4B"),
)


def parse_nvidia_smi(text: str) -> List[Dict[str, Any]]:
    """Rows of `nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`."""
    gpus: List[Dict[str, Any]] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({"index": int(parts[0]), "name": ",".join(parts[1:-3]), "vram_total_mb": int(float(parts[-3])),
                         "vram_free_mb": int(float(parts[-2])), "driver": parts[-1]})
        except ValueError:
            continue
    return gpus


def acestep_tier(vram_gb: Optional[float]) -> Optional[str]:
    if vram_gb is None:
        return None
    for upper, text in ACESTEP_TIERS:
        if vram_gb <= upper:
            return text
    return None  # pragma: no cover - the last tier is unbounded


def vram_fit(required_gb: Optional[float], doc: Optional[Dict[str, Any]],
             requested_device: Optional[str] = None) -> Dict[str, Any]:
    """`doc` (a detect() document) against a model's GPU memory requirement in GB (GiB, as ACE-Step counts it). Pure.
    The largest GPU found is the one compared; on Apple Silicon it is the unified memory, which the CPU shares.
    An Intel GPU (XPU) is only noticed through xpu-smi, so "no GPU found" is not proof of none: with
    `requested_device` "xpu" it counts as not measured (UNKNOWN), and the NO_GPU detail says to pass --device xpu."""
    out: Dict[str, Any] = {"status": "NOT_CHECKED", "required_gb": required_gb, "tolerance_gb": VRAM_TOLERANCE_GB,
                           "accelerator": None, "vram_total_mb": None, "detail": "not checked"}
    if required_gb is None or doc is None:
        return out
    mb = doc.get("vram_total_mb")
    out.update(accelerator=doc.get("accelerator"), vram_total_mb=mb)
    if doc.get("status") == "CPU_ONLY" and requested_device == "xpu":
        out.update(status="UNKNOWN", accelerator="xpu",
                   detail="--device xpu was asked for, and an Intel GPU's memory is not measured without xpu-smi")
    elif doc.get("status") == "CPU_ONLY":
        out.update(status="NO_GPU", detail=f"no GPU found ({doc.get('detected_by') or 'CPU only'}); an Intel GPU (XPU) "
                                           "is only noticed through xpu-smi: on one, pass --device xpu")
    elif mb is None:
        out.update(status="UNKNOWN", detail=f"GPU memory not measured ({doc.get('detail') or doc.get('accelerator')})")
    else:
        have = mb / 1024.0
        ok = have + VRAM_TOLERANCE_GB >= required_gb
        what = "unified memory, shared with the CPU" if doc.get("unified_memory") else "GPU memory"
        out.update(status="OK" if ok else "INSUFFICIENT",
                   detail=f"{mb} MiB ({have:.1f} GB) of {what}, {'meets' if ok else 'below'} the {required_gb:g} GB required")
    return out


def _nvidia(timeout: float) -> Optional[Dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    code, out, err = run_process_group([exe, *NVIDIA_QUERY], timeout)
    gpus = parse_nvidia_smi(out) if code == 0 else []
    if not gpus:
        reason = "timed out" if code == TIMEOUT_EXIT else (err.strip().splitlines()[-1] if err.strip() else f"exit {code}")
        return {"accelerator": "cuda", "status": "UNKNOWN", "gpus": [], "detected_by": "nvidia-smi",
                "detail": f"nvidia-smi found but gave no GPU rows ({reason})"}
    return {"accelerator": "cuda", "status": "OK", "gpus": gpus, "detected_by": "nvidia-smi"}


def _apple_unified_memory_mb(timeout: float) -> Optional[int]:
    exe = shutil.which("sysctl") or ("/usr/sbin/sysctl" if os.path.isfile("/usr/sbin/sysctl") else None)
    if not exe:
        return None
    code, out, _ = run_process_group([exe, "-n", "hw.memsize"], timeout)
    try:
        return int(out.strip()) // (1024 * 1024) if code == 0 else None
    except ValueError:
        return None


def detect(timeout: float = 10.0) -> Dict[str, Any]:
    """{"status", "accelerator", "gpus", "vram_total_mb", "vram_free_mb", "unified_memory", "detected_by", "detail",
    "acestep_recommendation"}. status: OK (a GPU was measured) | CPU_ONLY | UNKNOWN (a GPU tool answered oddly)."""
    system, machine = platform.system(), platform.machine().lower()
    doc: Dict[str, Any] = {"platform": system, "machine": machine, "gpus": [], "vram_total_mb": None, "vram_free_mb": None,
                           "unified_memory": False, "detail": None}
    nv = _nvidia(timeout)
    if nv:
        doc.update(nv)
        if nv["gpus"]:
            best = max(nv["gpus"], key=lambda g: g["vram_total_mb"])
            doc.update(vram_total_mb=best["vram_total_mb"], vram_free_mb=best["vram_free_mb"],
                       detail=f"{best['name']}, {best['vram_total_mb']} MiB")
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        mem = _apple_unified_memory_mb(timeout)
        doc.update(accelerator="mps", status="OK" if mem else "UNKNOWN", unified_memory=True, vram_total_mb=mem,
                   detected_by="platform + sysctl hw.memsize",
                   detail="Apple Silicon: MPS shares system memory with the CPU; the GPU budget is lower than the total")
    elif shutil.which("rocm-smi"):
        doc.update(accelerator="rocm", status="UNKNOWN", detected_by="rocm-smi on PATH",
                   detail="AMD ROCm tools found; VRAM is not read by this skill (ACE-Step documents ROCm support)")
    elif shutil.which("xpu-smi"):
        doc.update(accelerator="xpu", status="UNKNOWN", detected_by="xpu-smi on PATH",
                   detail="Intel GPU (XPU) tools found; VRAM is not read by this skill (ACE-Step documents XPU support)")
    else:
        doc.update(accelerator="cpu", status="CPU_ONLY", detected_by="no nvidia-smi, not Apple Silicon, no rocm-smi, no xpu-smi",
                   detail="no GPU found: ACE-Step documents CPU support, but generation speed on CPU is unverified")
    gb = None if doc["vram_total_mb"] is None else doc["vram_total_mb"] / 1024.0
    doc["acestep_recommendation"] = acestep_tier(gb) if doc["accelerator"] in ("cuda", "mps") else None
    doc["meets_acestep_minimum"] = None if gb is None else gb >= 4.0
    return doc
