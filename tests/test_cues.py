"""Unit tests for SFX_CUES (roadmap step 9, ADR-013): reading the caller's events, the spacing rule (dropped, never
moved), effect anchors, and sfx-cues end to end through the stub ffmpeg-skill of test_shaping with a library of
effects and a named WAV. No FFmpeg, no engine and no network are needed."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import test_library as tl
import test_shaping as ts
from helpers import run_cli, write_burst_wav

from sound_design_skill import pcm as P
from sound_design_skill.contract import contract
from sound_design_skill.cues import CUES_SCHEMA, CuesRequest, anchors, default_align, read_events, sfx_cues, space
from sound_design_skill.errors import SoundDesignError

RATE = 16000
CUTS = [1.0, 2.5, 3.0, 4.75, 6.0, 7.2, 8.0, 9.5, 11.0, 12.25]          # the fixture's 10 cut times


class EventTests(unittest.TestCase):
    def test_cuts_emphasis_and_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"events": [{"time": 2.0, "kind": "emphasis", "effect": "ding", "level": -18, "label": "テロップ"}]},
                          fh, ensure_ascii=False)
            evs = read_events(CuesRequest(cuts=[1.0], emphasis=[3.0], events_path=path))
        self.assertEqual([(e["time"], e["kind"]) for e in evs], [(1.0, "cut"), (3.0, "emphasis"), (2.0, "emphasis")])
        self.assertEqual((evs[2]["effect"], evs[2]["level"], evs[2]["label"]), ("ding", -18.0, "テロップ"))
        for bad in (CuesRequest(), CuesRequest(cuts=[-1.0]), CuesRequest(cuts=["1"])):
            with self.assertRaises(SoundDesignError):
                read_events(bad)

    def test_spacing_drops_and_never_moves(self):
        evs = read_events(CuesRequest(cuts=[1.0, 1.1, 1.3, 2.0, 2.0], emphasis=[1.2]))
        kept, dropped = space(evs, 0.25)
        self.assertEqual([e["time"] for e in kept], [1.0, 1.3, 2.0])
        self.assertEqual(sorted(e["time"] for e in dropped), [1.1, 1.2, 2.0])
        self.assertTrue(all("minimum spacing" in d["reason"] for d in dropped))
        given = {e["time"] for e in evs}
        self.assertTrue(all(e["time"] in given for e in kept))                       # no time is invented
        self.assertEqual(space(evs, 0.0)[1], [])

    def test_anchors_and_alignment(self):
        self.assertEqual((default_align("whoosh"), default_align("Rise"), default_align("pop"), default_align("ding")),
                         ("peak", "peak", "onset", "onset"))
        with tempfile.TemporaryDirectory() as tmp:
            hit = P.read_wav(write_burst_wav(os.path.join(tmp, "h.wav"), "hit", 0.5, at=0.1, rate=RATE))
        a = anchors(hit)
        self.assertLessEqual(abs(a["onset"] - 0.1), 0.002)
        self.assertLessEqual(abs(a["peak"] - 0.105), 0.01)


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.lib = os.path.join(self.dir, "sfx")
        os.mkdir(self.lib)
        write_burst_wav(os.path.join(self.lib, "whoosh.wav"), "whoosh", 0.8, at=0.4, rate=RATE)
        write_burst_wav(os.path.join(self.lib, "pop.wav"), "hit", 0.3, at=0.02, rate=RATE)
        write_burst_wav(os.path.join(self.lib, "unknown.wav"), "hit", 0.3, at=0.02, rate=RATE)
        tl.manifest(self.lib, {"whoosh.wav": {"licence": "CC0-1.0", "kind": "sfx", "tags": ["whoosh"]},
                               "pop.wav": {"licence": "CC-BY-4.0", "credit": "SFX: 作者", "kind": "sfx", "tags": ["pop"]},
                               "unknown.wav": {"kind": "sfx", "tags": ["pop", "ding"]}})
        self.skill = ts.StubSkill()

    def test_ten_cut_times_give_ten_cues_at_those_times_respecting_the_spacing(self):
        out = os.path.join(self.dir, "cues.json")
        doc = sfx_cues(CuesRequest(cuts=CUTS, emphasis=[2.6, 8.1], library_dir=self.lib, min_spacing=0.4, output_path=out),
                       skill=self.skill)
        cut_cues = [c for c in doc["cues"] if c["kind"] == "cut"]
        self.assertEqual([c["time"] for c in cut_cues], CUTS)
        for c in doc["cues"]:
            self.assertAlmostEqual(c["start"] + c["anchor_in_file"], c["time"], places=6)   # the anchor sits on the event
        self.assertEqual(sorted(d["time"] for d in doc["dropped"]), [2.6, 8.1])        # 0.1 s after a cut
        times = [c["time"] for c in doc["cues"]]
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.4)
        self.assertEqual(doc["summary"], {"events": 12, "cues": 10, "dropped": 2, "smallest_gap": 0.5})
        whoosh = doc["sources"]["whoosh"]
        self.assertEqual((whoosh["from"], whoosh["file"], whoosh["licence"]["licence"]), ("library", "whoosh.wav", "CC0-1.0"))
        self.assertEqual(cut_cues[0]["align"], "peak")
        self.assertAlmostEqual(cut_cues[0]["gain_db"], -12.0 - whoosh["anchors"]["sample_peak_dbfs"], places=2)
        with open(out, encoding="utf-8") as fh:
            sheet = json.load(fh)
        self.assertEqual((sheet["schema"], sheet["cues"]), (CUES_SCHEMA, doc["cues"]))
        self.assertNotIn(self.dir, json.dumps(sheet))
        self.assertEqual(doc["engine"], None)

    def test_emphasis_effects_named_files_levels_and_refusals(self):
        ding = write_burst_wav(os.path.join(self.dir, "ding.wav"), "hit", 0.6, at=0.05, rate=RATE)
        doc = sfx_cues(CuesRequest(emphasis=[0.02, 3.0], cuts=[5.0], library_dir=self.lib, files={"ding": ding},
                                   effects={"emphasis": "ding"}, level=-20.0), skill=self.skill)
        self.assertEqual([(c["time"], c["effect"], c["from"]) for c in doc["cues"]],
                         [(0.02, "ding", "file"), (3.0, "ding", "file"), (5.0, "whoosh", "library")])
        first = doc["cues"][0]                                    # the onset at 0.05 s would start before 0 s
        self.assertEqual(first["start"], 0.0)
        self.assertAlmostEqual(first["file_offset"], first["anchor_in_file"] - 0.02, places=6)
        self.assertTrue(any("before 0 s" in w for w in doc["warnings"]))
        self.assertEqual(doc["cues"][1]["level_dbfs"], -20.0)
        with self.assertRaises(SoundDesignError) as e:                       # "ding" only on a file without a licence
            sfx_cues(CuesRequest(emphasis=[1.0], effects={"emphasis": "ding"}, library_dir=self.lib), skill=self.skill)
        self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")
        with self.assertRaises(SoundDesignError) as e:
            sfx_cues(CuesRequest(cuts=[1.0]), skill=self.skill)
        self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")
        with self.assertRaises(SoundDesignError) as e:
            sfx_cues(CuesRequest(cuts=[1.0], effects={"intro": "x"}, library_dir=self.lib), skill=self.skill)
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    def test_cli_and_contract(self):
        out = os.path.join(self.dir, "c.json")
        with mock.patch("sound_design_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("sfx-cues", "--cuts", ",".join(map(str, CUTS)), "--library", self.lib, "--min-spacing", "0.4",
                              "--effect", "cut=pop", "-o", out, "--json")
            self.assertEqual(rc, 0, doc)
            self.assertEqual([c["time"] for c in doc["cues"]], CUTS)
            self.assertTrue(all(c["effect"] == "pop" and c["align"] == "onset" for c in doc["cues"]))
            rc, again = run_cli("sfx-cues", "--cuts", "1", "--library", self.lib, "-o", out, "--json")
            self.assertEqual((rc, again["error"]["code"]), (2, "OUTPUT_EXISTS"))
        tool = next(t for t in contract()["tools"] if t["tool_id"] == "sound-design/sfx-cues")
        self.assertEqual((tool["roadmap_step"], tool["engine"], tool["writes_media"]), (9, None, False))


if __name__ == "__main__":
    unittest.main()
