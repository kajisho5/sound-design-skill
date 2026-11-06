"""Unit tests for loopable beds, timed stingers and the seam check (roadmap step 7, ADR-011): the PCM reader, the peak
and seam measurements, the loop-point search, the stinger plan, and loop-bed / stinger / seam-check end to end through
a stub ffmpeg-skill that edits PCM WAVs with the wave module. No FFmpeg, no engine and no network are needed; the
integration tests run the same through the real ffmpeg-skill."""
from __future__ import annotations

import array
import json
import math
import os
import struct
import tempfile
import unittest
from unittest import mock

import test_library as tl
from helpers import ROOT, run_cli, write_bed_wav, write_burst_wav, write_tone_wav

from sound_design_skill import pcm as P
from sound_design_skill.contract import contract
from sound_design_skill.doctor import doctor
from sound_design_skill.errors import SoundDesignError
from sound_design_skill.shaping import (PEAK_TOLERANCE, LoopBedRequest, SeamCheckRequest, StingerRequest, loop_bed,
                                        plan_stinger, seam_check, stinger)

RATE = 16000


class StubSkill(tl.StubLibrarySkill):
    """The library stub plus ffmpeg-skill/audio fades (linear), on 16-bit PCM WAVs."""

    def fade(self, src, dst, fade_in=0.0, fade_out=0.0, timeout=None):
        self.calls.append(("fade", fade_in, fade_out))
        p, s = tl._read(src)
        ch, n = p.nchannels, len(s) // p.nchannels
        fi, fo = int(round(fade_in * p.framerate)), int(round(fade_out * p.framerate))
        out = array.array("h", s)
        for i in range(n):
            g = 1.0
            if fi and i < fi:
                g = min(g, i / fi)
            if fo and i >= n - fo:
                g = min(g, (n - 1 - i) / fo)
            for c in range(ch):
                out[i * ch + c] = int(s[i * ch + c] * g)
        tl._write(dst, p, out)
        return {"status": "completed"}


def sine(seconds, rate=RATE, freq=220.0, amp=0.3, phase=0.0):
    return [amp * math.sin(2 * math.pi * freq * i / rate + phase) for i in range(int(seconds * rate))]


def write_raw(path, tag, bits, channels, rate, payload, extensible=False):
    """A WAV written by hand: PCM / float, any width, optionally WAVE_FORMAT_EXTENSIBLE."""
    align = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * align, align, bits)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq"
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, bits)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    return path


class PcmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_reads_16_24_bit_float_and_extensible(self):
        tone = write_tone_wav(self.path("t16.wav"), seconds=0.5, rate=RATE)
        p16 = P.read_wav(tone)
        self.assertEqual((p16.rate, p16.channels, p16.frames), (RATE, 1, RATE // 2))
        self.assertAlmostEqual(max(p16.mono), 8000 / 32768, places=3)
        values = [0.5, -0.25, 0.0, 1.0 - 2 ** -23]
        b24 = b"".join(int(v * 8388608).to_bytes(3, "little", signed=True) * 2 for v in values)
        p24 = P.read_wav(write_raw(self.path("t24.wav"), 1, 24, 2, 48000, b24, extensible=True))
        self.assertEqual((p24.channels, p24.sampwidth), (2, 3))
        for got, want in zip(p24.mono, values):
            self.assertAlmostEqual(got, want, places=6)
        f32 = P.read_wav(write_raw(self.path("f32.wav"), 3, 32, 1, 48000, struct.pack("<4f", *values)))
        self.assertAlmostEqual(f32.mono[1], -0.25, places=6)
        for bad, data in (("text.wav", b"not a wav"), ("mp3.wav", b"ID3\x03\x00\x00\x00")):
            with open(self.path(bad), "wb") as fh:
                fh.write(data)
            with self.assertRaises(SoundDesignError) as e:
                P.read_wav(self.path(bad))
            self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")
        with self.assertRaises(SoundDesignError) as e:
            P.read_wav(write_raw(self.path("alaw.wav"), 6, 8, 1, 8000, b"\x00" * 8))
        self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")

    def test_silence_and_peak(self):
        s = P.read_wav(P.write_silence(self.path("s.wav"), 24000, 2, 1234))
        self.assertEqual((s.frames, s.channels, max(abs(v) for v in s.mono)), (1234, 2, 0.0))
        hit = P.read_wav(write_burst_wav(self.path("hit.wav"), "hit", 1.0, at=0.3, rate=RATE))
        pk = P.peak(hit)
        self.assertLessEqual(abs(pk["time"] - 0.3), 0.01)
        self.assertGreaterEqual(pk["sample_peak_time"], 0.3)
        whoosh = P.read_wav(write_burst_wav(self.path("w.wav"), "whoosh", 1.2, at=0.6, rate=RATE))
        self.assertLess(abs(P.peak(whoosh)["time"] - 0.6), 0.05)
        rise = P.read_wav(write_burst_wav(self.path("r.wav"), "rise", 1.0, rate=RATE))
        self.assertGreater(P.peak(rise)["time"], 0.9)


class SeamTests(unittest.TestCase):
    def test_a_continuous_signal_passes(self):
        x = sine(3.0)
        r = P.seam_check(x, RATE, [RATE])
        self.assertTrue(r["passed"], r)
        self.assertLess(r["seams"][0]["click_ratio"], P.CLICK_MAX)

    def test_a_discontinuity_is_a_click(self):
        x = sine(1.0) + sine(1.0, phase=math.pi / 2)
        r = P.seam_check(x, RATE, [RATE], window=0.5)
        self.assertFalse(r["passed"])
        self.assertTrue(any(m.startswith("click") for m in r["seams"][0]["reasons"]), r)

    def test_a_level_jump_and_a_timbre_change_fail(self):
        x = sine(1.0) + sine(1.0, amp=0.1)                          # same phase, a third of the level
        r = P.seam_check(x, RATE, [RATE], window=0.5)
        self.assertTrue(any(m.startswith("level") for m in r["seams"][0]["reasons"]), r)
        y = sine(1.0, freq=200.0) + sine(1.0, freq=3200.0)        # level equal, spectrum moved four octaves
        r = P.seam_check(y, RATE, [RATE], window=0.5)
        self.assertTrue(any(m.startswith("spectrum") for m in r["seams"][0]["reasons"]), r)

    def test_silence_passes_and_an_edge_seam_is_reported(self):
        r = P.seam_check([0.0] * (2 * RATE), RATE, [RATE], window=0.5)
        self.assertTrue(r["passed"])
        self.assertTrue(r["seams"][0]["silent"])
        r = P.seam_check(sine(2.0), RATE, [100], window=0.5)
        self.assertFalse(r["passed"])
        self.assertIn("each side", r["seams"][0]["reasons"][0])
        self.assertEqual(r["limits"], P.seam_limits(0.5))


class LoopPointTests(unittest.TestCase):
    def test_finds_the_repeat_nearest_the_request(self):
        period = int(0.5 * RATE)
        rng = __import__("random").Random(3)
        base = [rng.random() * 2 - 1 for _ in range(period)]                # seeded noise, repeated: only whole periods match
        x = base * 12
        L, sim = P.find_loop_point(x, RATE, target=int(4.1 * RATE), search=int(0.3 * RATE), crossfade=int(0.1 * RATE),
                                   lowest=10, highest=len(x) - int(0.1 * RATE))
        self.assertEqual(L, 8 * period)
        self.assertGreater(sim, 0.99)

    def test_no_search_is_exact_and_impossible_requests_are_refused(self):
        x = sine(3.0)
        L, _ = P.find_loop_point(x, RATE, target=RATE, search=0, crossfade=800, lowest=10, highest=len(x) - 800)
        self.assertEqual(L, RATE)
        with self.assertRaises(SoundDesignError) as e:
            P.find_loop_point(x, RATE, target=5 * RATE, search=0, crossfade=800, lowest=10, highest=len(x) - 800)
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")


class StingerPlanTests(unittest.TestCase):
    def test_pad_before_cut_after_and_head_cut(self):
        self.assertEqual(plan_stinger(1000, 100, 800, 300), {"head": 0, "pre": 200, "body": 600, "post": 0})
        self.assertEqual(plan_stinger(1000, 600, 800, 100), {"head": 500, "pre": 0, "body": 500, "post": 300})
        self.assertEqual(plan_stinger(200, 50, 800, 400), {"head": 0, "pre": 350, "body": 200, "post": 250})
        for args in ((1000, 100, 800, 300), (1000, 600, 800, 100), (200, 50, 800, 400), (5, 0, 9, 9)):
            plan = plan_stinger(*args)
            self.assertEqual(plan["pre"] + plan["body"] + plan["post"], args[2])
            self.assertEqual(plan["pre"] + (args[1] - plan["head"]), args[3])     # the peak lands on peak_at


class OpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.skill = StubSkill()

    def path(self, name):
        return os.path.join(self.dir, name)

    def test_loop_bed_is_exact_seamless_and_recorded(self):
        src = write_bed_wav(self.path("bed.wav"), seconds=10.0, rate=RATE, channels=1)
        doc = loop_bed(LoopBedRequest(src, self.path("loop.wav"), duration=4.0, search=0.3), skill=self.skill)
        res = P.read_wav(doc["output"]["path"])
        self.assertEqual(res.frames, doc["loop"]["frames"])
        self.assertLessEqual(abs(doc["loop"]["length"] - 4.0), 0.3)
        self.assertTrue(doc["seam"]["passed"])
        self.assertGreater(doc["loop"]["similarity"], 0.9)
        self.assertEqual([c[0] for c in self.skill.calls if c[0] in ("cut", "join")], ["cut", "cut", "cut", "join", "cut"])
        src_pcm = P.read_wav(src)
        L, x = doc["loop"]["frames"], int(round(doc["loop"]["crossfade"] * RATE))
        self.assertAlmostEqual(res.mono[-1], src_pcm.mono[L - 1], places=3)        # the wrap is the source's continuity
        self.assertAlmostEqual(res.mono[0], src_pcm.mono[L], places=3)
        self.assertAlmostEqual(res.mono[x + 10], src_pcm.mono[x + 10], places=3)
        p = doc["provenance"]
        self.assertEqual((p["op"], p["engine"], p["source"]["file"]), ("loop-bed", None, "bed.wav"))
        self.assertNotIn(self.dir, json.dumps(p))
        self.assertEqual(sorted(os.listdir(self.dir)), ["bed.wav", "loop.wav"])
        tl._write(self.path("x10.wav"), tl._read(doc["output"]["path"])[0], tl._read(doc["output"]["path"])[1] * 10)
        check = seam_check(SeamCheckRequest(self.path("x10.wav"), period=doc["loop"]["length"]), skill=self.skill)
        self.assertTrue(check["passed"], check["seams"])
        self.assertEqual(len(check["seams"]), 9)

    def test_loop_bed_refusals(self):
        src = write_bed_wav(self.path("bed.wav"), seconds=4.0, rate=RATE, channels=1)
        for kw, code in ((dict(duration=10.0), "INVALID_ARGUMENT"), (dict(duration=2.0, crossfade=1.0), "INVALID_ARGUMENT"),
                         (dict(duration=0.5), "INVALID_ARGUMENT"), (dict(duration=2.0, search=-1), "INVALID_ARGUMENT")):
            with self.assertRaises(SoundDesignError) as e:
                loop_bed(LoopBedRequest(src, self.path("o.wav"), **kw), skill=self.skill)
            self.assertEqual(e.exception.code, code, kw)
        with self.assertRaises(SoundDesignError) as e:
            loop_bed(LoopBedRequest(src, src, duration=2.0), skill=self.skill)
        self.assertEqual(e.exception.code, "OUTPUT_IS_INPUT")
        with self.assertRaises(SoundDesignError) as e:
            loop_bed(LoopBedRequest(self.path("nope.wav"), self.path("o.wav"), duration=2.0), skill=self.skill)
        self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")
        self.assertFalse(os.path.exists(self.path("o.wav")))

    def test_stinger_peak_lands_within_10_ms_for_each_type(self):
        cases = (("whoosh", 1.5, 0.6, 1.0, 0.5), ("hit", 1.2, 0.05, 1.0, 0.5), ("rise", 2.0, 0.0, 1.0, 0.8),
                 ("hit", 0.3, 0.02, 2.0, 1.0), ("whoosh", 1.5, 0.6, 0.5, 0.1))
        for kind, secs, at, duration, peak_at in cases:
            with self.subTest(kind=kind, duration=duration, peak_at=peak_at):
                src = write_burst_wav(self.path(f"{kind}-{secs}.wav"), kind, secs, at=at, rate=RATE)
                out = self.path(f"st-{kind}-{duration}-{peak_at}.wav")
                doc = stinger(StingerRequest(src, out, duration=duration, peak_at=peak_at, type=kind), skill=self.skill)
                res = P.read_wav(out)
                self.assertEqual(res.frames, int(round(duration * RATE)))
                self.assertLessEqual(abs(P.peak(res)["time"] - peak_at), PEAK_TOLERANCE)
                self.assertEqual(doc["timing"]["peak"]["time"], P.peak(res)["time"])
                self.assertEqual((doc["timing"]["type"], doc["provenance"]["op"], doc["engine"]), (kind, "stinger", None))

    def test_stinger_refusals(self):
        src = write_burst_wav(self.path("hit.wav"), "hit", 1.0, at=0.1, rate=RATE)
        for kw in (dict(duration=1.0, peak_at=1.5), dict(duration=1.0, peak_at=0.0), dict(duration=1.0, peak_at=0.99),
                   dict(duration=0, peak_at=0), dict(duration=40, peak_at=1), dict(duration=1.0, peak_at=0.5, type="boom")):
            with self.assertRaises(SoundDesignError) as e:
                stinger(StingerRequest(src, self.path("o.wav"), **kw), skill=self.skill)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)
        self.assertEqual(self.skill.calls, [])

    def test_seam_check_finds_a_naive_loop(self):
        src = write_bed_wav(self.path("bed.wav"), seconds=6.0, rate=RATE, channels=1)
        p, s = tl._read(src)
        cut = s[:int(2.005 * RATE)]                                          # ends on a kick's peak, starts at zero
        tl._write(self.path("naive.wav"), p, cut * 4)
        doc = seam_check(SeamCheckRequest(self.path("naive.wav"), period=2.005, window=0.5), skill=self.skill)
        self.assertFalse(doc["passed"])
        self.assertEqual(len(doc["seams"]), 3)
        self.assertTrue(doc["warnings"])
        with self.assertRaises(SoundDesignError):
            seam_check(SeamCheckRequest(self.path("naive.wav")), skill=self.skill)

    def test_cli_through_a_stub_ffmpeg_skill(self):
        src = write_bed_wav(self.path("bed.wav"), seconds=6.0, rate=RATE, channels=1)
        with mock.patch("sound_design_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("loop-bed", src, "-o", self.path("loop.wav"), "--duration", "2", "--search", "0.2", "--json")
            self.assertEqual(rc, 0, doc)
            rc, bad = run_cli("seam-check", src, "--seams", "1.0,2.5", "--window", "0.5", "--json")
            self.assertEqual(rc, 0, bad)
            p, s = tl._read(src)
            tl._write(self.path("jump.wav"), p, s[:int(1.005 * RATE)] + s[int(1.25 * RATE):])
            rc, failed = run_cli("seam-check", self.path("jump.wav"), "--seams", "1.005", "--window", "0.5", "--json")
            self.assertEqual((rc, failed["passed"]), (1, False))
            rc, err = run_cli("stinger", src, "-o", self.path("s.wav"), "--duration", "1", "--peak-at", "2", "--json")
            self.assertEqual((rc, err["error"]["code"]), (2, "INVALID_ARGUMENT"))


class ContractDoctorTests(unittest.TestCase):
    def test_tools_and_ops_are_published(self):
        tools = {t["tool_id"]: t for t in contract()["tools"]}
        for name in ("loop-bed", "stinger", "seam-check"):
            t = tools[f"sound-design/{name}"]
            self.assertEqual((t["roadmap_step"], t["engine"], t["implemented"]), (7, None, True))
        self.assertEqual(tools["sound-design/stinger"]["peak_tolerance_s"], 0.010)
        self.assertEqual(tools["sound-design/seam-check"]["limits"], P.seam_limits())
        self.assertIn("ADR-011", contract()["policy"]["shaping"])
        with tempfile.TemporaryDirectory() as tmp:
            ops = doctor(os.path.join(tmp, "none"))["checks"]["operations"]
        for op in ("loop", "stinger"):
            self.assertEqual((ops[op]["roadmap_step"], ops[op]["engine"], ops[op]["confirmation_required"]), (7, None, False))

    def test_shaping_imports_no_engine_and_runs_no_ffmpeg(self):
        for name in ("shaping.py", "pcm.py"):
            text = (ROOT / "src" / "sound_design_skill" / name).read_text(encoding="utf-8")
            for banned in ("engines", "service", "run_engine", "subprocess", "socket", '"ffmpeg"'):
                self.assertNotIn(banned, text, f"{name}: {banned}")


if __name__ == "__main__":
    unittest.main()
