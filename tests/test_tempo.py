"""Unit tests for tempo-targeted GENERATE_BGM (roadmap step 8, ADR-012): the bar-aligned length, the checks, the beat
grid document, and generate-bgm --bpm end to end through the real worker, the fake ACE-Step engine (the verified
GenerationParams.bpm / timesignature) and a stub ffmpeg-skill. No engine, no FFmpeg and no network are needed."""
from __future__ import annotations

import array
import contextlib
import io
import json
import os
import tempfile
import unittest
import wave

import test_unit as tu
from helpers import env, fake_acestep_engine, fake_checkpoints, no_engines_env, run_cli

from sound_design_skill.contract import contract
from sound_design_skill.errors import SoundDesignError
from sound_design_skill.ops import BgmRequest, generate_bgm
from sound_design_skill.tempo import GRID_SCHEMA, beat_grid, check_tempo, grid_path, plan_tempo


class PlanTests(unittest.TestCase):
    def test_whole_bars_nearest_the_request(self):
        p = plan_tempo(15.0, 120, 4, 10.0, 600.0)                       # "120 BPM, 15 s": 7.5 bars -> 8 bars
        self.assertEqual((p["bars"], p["beats"], p["duration"], p["adjusted"]), (8, 32, 16.0, True))
        self.assertEqual((p["beat_s"], p["bar_s"], p["time_signature"]), (0.5, 2.0, "4/4"))
        self.assertEqual(plan_tempo(16.0, 120, 4, 10.0, 600.0)["adjusted"], False)
        self.assertEqual(plan_tempo(14.9, 120, 4, 10.0, 600.0)["duration"], 14.0)
        self.assertEqual(plan_tempo(30.0, 90, 3, 10.0, 600.0)["duration"], 30.0)            # 3/4 at 90: 2 s bars
        six = plan_tempo(12.0, 120, 6, 10.0, 600.0)
        self.assertEqual((six["time_signature"], six["bar_s"], six["bars"]), ("6/8", 3.0, 4))

    def test_kept_inside_the_engine_range(self):
        self.assertEqual(plan_tempo(5.0, 120, 4, 10.0, 600.0)["duration"], 10.0)            # raised to 5 bars
        self.assertEqual(plan_tempo(610.0, 30, 4, 10.0, 600.0)["duration"], 600.0)          # 75 bars of 8 s
        for bpm in (30, 77, 123, 299):
            for ts in (2, 3, 4, 6):
                p = plan_tempo(37.0, bpm, ts, 10.0, 600.0)
                self.assertTrue(10.0 <= p["duration"] <= 600.0)
                self.assertAlmostEqual(p["duration"] / p["bar_s"], p["bars"], places=6)     # ends on a bar line

    def test_checks(self):
        self.assertEqual(check_tempo(None, None), (None, None))
        self.assertEqual(check_tempo(120, None), (120, 4))
        for bpm, ts in ((29, None), (301, None), (120.5, None), (True, None), ("120", None), (120, 5), (120, 0), (None, 4)):
            with self.assertRaises(SoundDesignError, msg=(bpm, ts)) as e:
                check_tempo(bpm, ts)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    def test_grid_document(self):
        g = beat_grid(plan_tempo(15.0, 120, 4, 10.0, 600.0), {"file": "bed.wav", "sha256": "x"})
        self.assertEqual((g["schema"], g["bpm"], g["bars"], g["measured"], g["first_beat"]), (GRID_SCHEMA, 120, 8, False, 0.0))
        self.assertEqual(len(g["beats"]), 32)
        self.assertEqual(g["beats"][5], {"index": 5, "time": 2.5, "bar": 2, "beat": 2})
        self.assertEqual(g["downbeats"], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        self.assertEqual(g["bar_lines"][-1], g["duration"])
        self.assertEqual(grid_path(os.path.join("a", "bed.wav")), os.path.join("a", "bed.beats.json"))


class TempoBgmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.out_dir = os.path.join(self.dir, "out")
        os.mkdir(self.out_dir)
        ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
        for cm in (env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        engine_cm = fake_acestep_engine(self.dir)
        self.fake = engine_cm.__enter__()
        self.addCleanup(engine_cm.__exit__, None, None, None)
        self.skill = tu.StubFfmpegSkill()

    def out(self, name="bed.wav"):
        return os.path.join(self.out_dir, name)

    def test_bpm_reaches_the_engine_and_the_grid_lands_beside_the_wav(self):
        doc = generate_bgm(BgmRequest(output_path=self.out(), duration=15.0, prompt="upbeat pop", seed=3, bpm=120),
                           skill=self.skill)
        gen = [c for c in self.fake["calls"]() if c["event"] == "generate_music"][0]["params"]
        self.assertEqual((gen["bpm"], gen["timesignature"], gen["duration"]), (120, "4", 16.0))
        self.assertEqual((doc["request"]["duration"], doc["request"]["requested_duration"], doc["output"]["duration"]),
                         (16.0, 15.0, 16.0))
        grid_file = self.out("bed.beats.json")
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["bed.beats.json", "bed.wav"])
        with open(grid_file, encoding="utf-8") as fh:
            grid = json.load(fh)
        self.assertEqual({k: v for k, v in doc["tempo_map"].items() if k != "path"}, grid)
        self.assertEqual((grid["audio"]["file"], grid["audio"]["sha256"]), ("bed.wav", doc["output"]["sha256"]))
        self.assertEqual((len(grid["beats"]), grid["bars"], grid["duration"]), (32, 8, 16.0))
        self.assertEqual(doc["handoff"]["beat_grid"], grid_file)
        self.assertEqual(doc["provenance"]["parameters"]["tempo"]["bpm"], 120)
        self.assertTrue(any("ends on a bar line" in w for w in doc["warnings"]))
        self.assertTrue(any("not measured" in w for w in doc["warnings"]))
        # the fake engine keeps the tempo exactly: its clicks fall on the written grid (a check of the grid's maths,
        # not of ACE-Step)
        with wave.open(self.out()) as w:                                                # the left channel (the fake's
            r, frames = w.getframerate(), array.array("h", w.readframes(w.getnframes()))  # right one is its negative)
        left, half = frames[0::2], int(0.1 * r)
        for b in grid["beats"][1:]:
            i = int(round(b["time"] * r))
            loudest = max(range(i - half, i + half), key=lambda k: abs(left[k]))
            self.assertLessEqual(abs(loudest / r - b["time"]), 0.020, b)

    def test_three_four_and_refusals(self):
        doc = generate_bgm(BgmRequest(output_path=self.out("w.wav"), duration=20.0, prompt="waltz", seed=1, bpm=90,
                                      time_signature=3), skill=self.skill)
        self.assertEqual((doc["tempo_map"]["time_signature"], doc["tempo_map"]["bars"], doc["output"]["duration"]),
                         ("3/4", 10, 20.0))
        self.assertEqual(doc["warnings"][-1:], [w for w in doc["warnings"] if "not measured" in w])
        with open(self.out("other.beats.json"), "w", encoding="utf-8") as fh:            # the grid exists already
            fh.write("{}")
        with self.assertRaises(SoundDesignError) as e:
            generate_bgm(BgmRequest(output_path=self.out("other.wav"), duration=20.0, prompt="x", bpm=90, time_signature=3),
                         skill=self.skill)
        self.assertEqual(e.exception.code, "OUTPUT_EXISTS")
        self.assertFalse(os.path.exists(self.out("other.wav")))
        for kw in (dict(bpm=10), dict(bpm=120, time_signature=7), dict(time_signature=4)):
            with self.assertRaises(SoundDesignError) as e:
                generate_bgm(BgmRequest(output_path=self.out("bad.wav"), duration=20.0, prompt="x", **kw), skill=self.skill)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)
        self.assertFalse(os.path.exists(self.out("bad.wav")))

    def test_without_bpm_nothing_changes(self):
        doc = generate_bgm(BgmRequest(output_path=self.out(), duration=12.0, prompt="calm", seed=2), skill=self.skill)
        gen = [c for c in self.fake["calls"]() if c["event"] == "generate_music"][0]["params"]
        self.assertEqual((gen["bpm"], gen["timesignature"]), (None, ""))
        self.assertEqual((doc["tempo_map"], doc["handoff"]["beat_grid"], doc["warnings"]), (None, None, []))
        self.assertNotIn("tempo", doc["provenance"]["parameters"])
        self.assertEqual(os.listdir(self.out_dir), ["bed.wav"])

    def test_cli_and_contract(self):
        with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()):   # argparse refuses 5
            run_cli("generate-bgm", "--prompt", "x", "--duration", "15", "--bpm", "300", "--time-signature", "5",
                    "-o", self.out(), "--json")
        self.assertEqual(e.exception.code, 2)
        rc, doc = run_cli("generate-bgm", "--prompt", "x", "--duration", "15", "--bpm", "301", "-o", self.out(), "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "INVALID_ARGUMENT"))
        argv = " ".join(next(t for t in contract()["tools"] if t["tool_id"] == "sound-design/generate-bgm")["argv"])
        self.assertIn("--bpm", argv)
        self.assertIn("--time-signature 2|3|4|6", argv)


if __name__ == "__main__":
    unittest.main()
