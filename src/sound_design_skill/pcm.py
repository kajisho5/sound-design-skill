"""PCM WAV measurement (roadmap step 7, ADR-011): reading the samples of a WAV this skill wrote through ffmpeg-skill,
and the measurements the loop and stinger tools need that no ffmpeg-skill tool reports: the time of a sound's peak,
the click / level / spectrum match across a loop seam, and a loop point. Pure Python, stdlib only, no process.

Nothing here edits audio. The one thing written is digital silence (zero samples) of a given length, the padding a
stinger needs before or after its sound; every change to a user's or an engine's audio is an ffmpeg-skill tool."""
from __future__ import annotations

import cmath
import math
import struct
import sys
import wave
from array import array
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import SoundDesignError

# Seam check (ADR-011). A seam passes when it has no click (a sample-level discontinuity, judged within +/- 1 ms) and
# the level and the spectrum of the SEAM_WINDOW seconds on each side match: long enough that the music's own beat-level
# movement (a kick right after the seam) averages out, short enough to catch a quiet outro looping into a loud intro.
SEAM_WINDOW = 1.0            # seconds compared on each side of a seam
CLICK_MAX = 3.0              # largest |2nd difference| within +/- 1 ms of the seam over the 99th percentile within +/- 50 ms
LEVEL_MAX_DB = 1.5           # |RMS after - RMS before|
SPECTRUM_MAX_DB = 6.0        # largest octave-band difference, over bands within SPECTRUM_RANGE_DB of the loudest
SPECTRUM_RANGE_DB = 30.0
SILENT_DBFS = -60.0          # both sides quieter than this: nothing to hear, the seam passes
OCTAVE_CENTRES = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
PEAK_WINDOW = 0.010          # RMS window of the envelope a peak is read from
PEAK_HOP = 0.001             # its step: peak times are resolved to 1 ms
LOOP_TIE = 0.02              # loop points this close in similarity count as equal: the one nearest the request wins
_EPS = 1e-9


@dataclass
class Pcm:
    rate: int
    channels: int
    sampwidth: int
    mono: List[float]         # the channels averaged, in [-1, 1]

    @property
    def frames(self) -> int:
        return len(self.mono)

    @property
    def duration(self) -> float:
        return len(self.mono) / float(self.rate)


def read_wav(path: str) -> Pcm:
    """The samples of a PCM WAV (8-bit unsigned, 16 / 24 / 32-bit signed, 32-bit float; WAVE_FORMAT_EXTENSIBLE
    accepted), averaged to mono. Raises UNSUPPORTED_MEDIA for anything else."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        raise SoundDesignError("INPUT_NOT_FOUND", f"cannot read {path}: {e}", {"input": path})
    name = path.rsplit("/", 1)[-1]
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SoundDesignError("UNSUPPORTED_MEDIA", f"{name} is not a RIFF WAVE file", {"input": path})
    pos, fmt, body = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = chunk
        elif cid == b"data":
            body = chunk
        pos += 8 + size + (size & 1)
    if fmt is None or body is None or len(fmt) < 16:
        raise SoundDesignError("UNSUPPORTED_MEDIA", f"{name} has no fmt / data chunk", {"input": path})
    tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == 0xFFFE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    width = bits // 8
    if channels < 1 or rate < 1 or width < 1 or align != channels * width or (tag, bits) not in (
            (1, 8), (1, 16), (1, 24), (1, 32), (3, 32)):
        raise SoundDesignError("UNSUPPORTED_MEDIA", f"{name}: format {tag} / {bits} bit is not PCM this skill reads",
                               {"input": path})
    n = len(body) // align
    body = body[:n * align]
    if tag == 3:
        flat = array("f")
        flat.frombytes(body)
        if sys.byteorder == "big":
            flat.byteswap()
        scale = 1.0
    elif bits == 8:
        flat = array("b", bytes((b - 128) & 0xFF for b in body))
        scale = 128.0
    elif bits == 16:
        flat = array("h")
        flat.frombytes(body)
        if sys.byteorder == "big":
            flat.byteswap()
        scale = 32768.0
    elif bits == 32:
        flat = array("i")
        flat.frombytes(body)
        if sys.byteorder == "big":
            flat.byteswap()
        scale = 2147483648.0
    else:   # 24-bit
        flat = array("i", (int.from_bytes(body[i:i + 3], "little", signed=True) for i in range(0, len(body), 3)))
        scale = 8388608.0
    if channels == 1:
        mono = [v / scale for v in flat]
    else:
        k = scale * channels
        mono = [sum(flat[i:i + channels]) / k for i in range(0, len(flat), channels)]
    return Pcm(rate=rate, channels=channels, sampwidth=width, mono=mono)


def write_silence(path: str, rate: int, channels: int, frames: int) -> str:
    """`frames` zero samples as 16-bit PCM WAV: the only audio this skill writes itself (stinger padding)."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * max(0, int(frames)))
    return path


def db(value: float) -> float:
    return 20.0 * math.log10(max(value, _EPS))


def rms(x: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in x) / len(x)) if len(x) else 0.0


def envelope(x: Sequence[float], rate: int, window: float = PEAK_WINDOW, hop: float = PEAK_HOP) -> List[Tuple[float, float]]:
    """(centre time, RMS) of a `window`-second window every `hop` seconds."""
    w, h = max(1, int(round(window * rate))), max(1, int(round(hop * rate)))
    acc = [0.0]
    for v in x:
        acc.append(acc[-1] + v * v)
    out = []
    for start in range(0, max(1, len(x) - w + 1), h):
        end = min(len(x), start + w)
        out.append(((start + end) / 2.0 / rate, math.sqrt(max(0.0, acc[end] - acc[start]) / max(1, end - start))))
    return out


def peak(pcm: Pcm) -> Dict[str, Any]:
    """Where a sound peaks: the centre of its loudest 10 ms (RMS, 1 ms steps), and its sample peak."""
    env = envelope(pcm.mono, pcm.rate)
    t, level = max(env, key=lambda e: e[1]) if env else (0.0, 0.0)
    i = max(range(len(pcm.mono)), key=lambda k: abs(pcm.mono[k])) if pcm.mono else 0
    return {"time": round(t, 6), "rms_dbfs": round(db(level), 2), "sample_peak_time": round(i / pcm.rate, 6),
            "sample_peak_dbfs": round(db(abs(pcm.mono[i]) if pcm.mono else 0.0), 2),
            "method": f"centre of the loudest {PEAK_WINDOW * 1000:g} ms RMS window, {PEAK_HOP * 1000:g} ms steps"}


def _fft(a: List[complex]) -> List[complex]:
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        step = cmath.exp(-2j * math.pi / size)
        half = size // 2
        for start in range(0, n, size):
            w = 1 + 0j
            for k in range(start, start + half):
                t = w * a[k + half]
                a[k + half] = a[k] - t
                a[k] += t
                w *= step
        size *= 2
    return a


def band_levels(x: Sequence[float], rate: int) -> Dict[str, float]:
    """Octave-band levels (dB) of `x`: the signal averaged down to about 12 kHz, then the mean power of 512-point
    Hann-windowed FFT frames (Welch); bands up to the reduced Nyquist frequency."""
    d = max(1, rate // 12000)
    y = [sum(x[i:i + d]) / d for i in range(0, len(x) - d + 1, d)]
    r, n = rate / d, 512
    while n > len(y) and n > 16:
        n >>= 1
    hann = [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
    power = [0.0] * (n // 2 + 1)
    frames = 0
    for start in range(0, max(1, len(y) - n + 1), n):
        seg = y[start:start + n]
        seg = seg + [0.0] * (n - len(seg))
        spec = _fft([complex(v * h, 0.0) for v, h in zip(seg, hann)])
        for k in range(n // 2 + 1):
            power[k] += abs(spec[k]) ** 2
        frames += 1
    out = {}
    for c in OCTAVE_CENTRES:
        lo, hi = c / math.sqrt(2), c * math.sqrt(2)
        if hi >= r / 2:
            break
        a, b = max(1, int(lo * n / r + 0.5)), min(n // 2, int(hi * n / r + 0.5))
        b = max(b, a + 1)
        out[f"{c:g}"] = round(10 * math.log10(sum(power[a:b]) / max(1, frames) / n + 1e-18), 2)
    return out


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def seam_metrics(x: Sequence[float], rate: int, index: int, window: float = SEAM_WINDOW) -> Dict[str, Any]:
    """Click, level and spectrum change at sample `index` of `x` (the first sample after the seam)."""
    w = max(16, int(window * rate))
    before, after = x[max(0, index - w):index], x[index:index + w]
    near, far = max(1, int(0.001 * rate)), max(4, int(0.05 * rate))
    d2 = {k: abs(x[k] - 2 * x[k - 1] + x[k - 2]) for k in range(max(2, index - far), min(len(x), index + far))}
    at = max((v for k, v in d2.items() if abs(k - index) <= near), default=0.0)
    around = [v for k, v in d2.items() if abs(k - index) > 2 * near]
    click = at / max(_percentile(around, 0.99), 1e-4)
    lb, la = db(rms(before)), db(rms(after))
    bb, ba = band_levels(before, rate), band_levels(after, rate)
    top = max(list(bb.values()) + list(ba.values()) + [-180.0])
    bands = [k for k in bb if k in ba and max(bb[k], ba[k]) >= top - SPECTRUM_RANGE_DB]
    spec = max((abs(ba[k] - bb[k]) for k in bands), default=0.0)
    return {"click_ratio": round(click, 3), "level_before_dbfs": round(lb, 2), "level_after_dbfs": round(la, 2),
            "level_delta_db": round(abs(la - lb), 3), "spectrum_delta_db": round(spec, 3),
            "bands_compared": bands, "silent": lb < SILENT_DBFS and la < SILENT_DBFS}


def seam_check(x: Sequence[float], rate: int, seams: Sequence[int], window: float = SEAM_WINDOW) -> Dict[str, Any]:
    """Every seam (sample index) checked against the fixed limits. A seam needs `window` seconds on each side."""
    w = int(window * rate)
    rows = []
    for s in seams:
        if not w <= s <= len(x) - w:
            rows.append({"time": round(s / rate, 6), "sample": s, "passed": False,
                         "reasons": [f"needs {window:g} s of audio on each side"]})
            continue
        m = seam_metrics(x, rate, s, window)
        reasons = []
        if not m["silent"]:
            if m["click_ratio"] > CLICK_MAX:
                reasons.append(f"click: ratio {m['click_ratio']:.2f} > {CLICK_MAX:g}")
            if m["level_delta_db"] > LEVEL_MAX_DB:
                reasons.append(f"level changes {m['level_delta_db']:.2f} dB > {LEVEL_MAX_DB:g} dB")
            if m["spectrum_delta_db"] > SPECTRUM_MAX_DB:
                reasons.append(f"spectrum changes {m['spectrum_delta_db']:.2f} dB > {SPECTRUM_MAX_DB:g} dB")
        rows.append(dict(m, time=round(s / rate, 6), sample=s, passed=not reasons, reasons=reasons))
    return {"passed": bool(rows) and all(r["passed"] for r in rows), "seams": rows, "limits": seam_limits(window)}


def seam_limits(window: float = SEAM_WINDOW) -> Dict[str, Any]:
    return {"click_ratio": CLICK_MAX, "level_delta_db": LEVEL_MAX_DB, "spectrum_delta_db": SPECTRUM_MAX_DB,
            "spectrum_range_db": SPECTRUM_RANGE_DB, "window_s": window, "silent_dbfs": SILENT_DBFS}


def _ncc(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(p * q for p, q in zip(a, b))
    den = math.sqrt(sum(p * p for p in a) * sum(q * q for q in b))
    return num / den if den > 1e-12 else 0.0


def _closest_best(scores: Dict[int, float], target: int, slack: float = LOOP_TIE) -> int:
    """The candidate closest to `target` among those within `slack` of the best score (ties: the earlier one)."""
    top = max(scores.values())
    return min((k for k, v in scores.items() if v >= top - slack), key=lambda k: (abs(k - target), k))


def find_loop_point(x: Sequence[float], rate: int, target: int, search: int, crossfade: int, lowest: int,
                    highest: int) -> Tuple[int, float]:
    """The loop length L (samples) in [target - search, target + search] (within [lowest, highest]) whose next
    `crossfade` samples look most like the first `crossfade` samples (normalised cross-correlation): the crossfade
    from x[L:] into x[0:] then blends near-identical audio. Coarse search on block averages at about 2 kHz, then
    sample by sample around the best; near-equal candidates (within LOOP_TIE) go to the one nearest the request.
    Returns (L, similarity in [-1, 1])."""
    lo, hi = max(lowest, target - search), min(highest, target + search)
    if lo > hi:
        raise SoundDesignError("INVALID_ARGUMENT", "the source is too short for this loop length and crossfade",
                               {"target": target, "lowest": lowest, "highest": highest})
    if lo == hi or search <= 0:
        L = min(max(target, lo), hi)
        return L, round(_ncc(x[L:L + crossfade], x[:crossfade]), 4)
    d = max(1, rate // 2000)
    xd = [sum(x[i:i + d]) / d for i in range(0, len(x) - d + 1, d)]
    ref = xd[:max(8, crossfade // d)]
    best = _closest_best({k * d: _ncc(xd[k:k + len(ref)], ref) for k in range(lo // d, hi // d + 1)}, target)
    fine = min(crossfade, 4096)
    ref_f = x[:fine]
    L = _closest_best({k: _ncc(x[k:k + fine], ref_f) for k in range(max(lo, best - d), min(hi, best + d) + 1)}, best)
    return L, round(_ncc(x[L:L + crossfade], x[:crossfade]), 4)


def frames_for(seconds: float, rate: int) -> int:
    return int(round(seconds * rate))


def describe(pcm: Optional[Pcm]) -> Optional[Dict[str, Any]]:
    if pcm is None:
        return None
    return {"sample_rate": pcm.rate, "channels": pcm.channels, "frames": pcm.frames, "duration": round(pcm.duration, 6)}
