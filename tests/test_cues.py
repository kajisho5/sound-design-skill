"""Unit tests for SFX_CUES (roadmap step 9, ADR-013): reading the caller's events, the spacing rule (dropped, never
moved), effect anchors, and sfx-cues end to end through the stub ffmpeg-skill of test_shaping with a library of
effects and a named WAV; and ADR-015: the same-file reuse spacing (rotation in rank order, the fallback drop, its
interaction with the minimum spacing, off by default) and the character filter on tags and mood. No FFmpeg, no engine
and no network are needed."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import test_library as tl
import test_shaping as ts
from helpers import ROOT, run_cli, write_burst_wav

from sound_design_skill import pcm as P
from sound_design_skill.contract import contract
from sound_design_skill.cues import (CHARACTER_VOCABULARY, CUES_SCHEMA, CuesRequest, anchors, default_align, read_events,
                                     sfx_cues, space)
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



def fixed_candidates(table, fixed=False):
    """A candidates() for space(): every event of an effect can use the files in table[effect], in that rank order."""
    return lambda ev: ([(name, name) for name in table[ev["effect"]]], fixed)


def events_at(times, effect="whoosh", kind="cut"):
    return [{"source_index": i, "time": t, "kind": kind, "effect": effect} for i, t in enumerate(times)]


class ReuseSpacingTests(unittest.TestCase):
    """space() with a minimum reuse spacing (ADR-015): pure, no files."""

    def test_rotation_takes_the_first_free_file_in_rank_order(self):
        kept, dropped = space(events_at([0.0, 10.0, 20.0, 30.0, 40.0, 100.0]), 0.25, 30.0,
                              fixed_candidates({"whoosh": ["a.wav", "b.wav", "c.wav"]}))
        self.assertEqual(dropped, [])
        self.assertEqual([e["pick"] for e in kept], [0, 1, 2, 0, 1, 0])       # a b c a b a: never random
        self.assertEqual(kept[1]["passed_over"], [{"file": "a.wav", "played_at": 0.0}])
        self.assertEqual(kept[2]["passed_over"], [{"file": "a.wav", "played_at": 0.0}, {"file": "b.wav", "played_at": 10.0}])
        self.assertEqual(kept[3]["passed_over"], [])                          # 30 s after a: exactly the spacing is allowed
        self.assertEqual([e["time"] for e in kept], [0.0, 10.0, 20.0, 30.0, 40.0, 100.0])   # nothing moved

    def test_no_free_file_drops_the_event_with_its_reason(self):
        kept, dropped = space(events_at([0.0, 10.0, 29.9, 30.0]), 0.25, 30.0, fixed_candidates({"whoosh": ["a.wav"]}))
        self.assertEqual([e["time"] for e in kept], [0.0, 30.0])
        self.assertEqual([(d["time"], d["by"]) for d in dropped], [(10.0, "min_reuse_spacing"), (29.9, "min_reuse_spacing")])
        self.assertIn("a.wav at 0.000s (10.000s earlier), the only eligible file for whoosh", dropped[0]["reason"])
        self.assertIn("minimum reuse spacing 30s", dropped[0]["reason"])
        kept, dropped = space(events_at([0.0, 5.0, 8.0]), 0.25, 30.0, fixed_candidates({"whoosh": ["a.wav", "b.wav"]}))
        self.assertEqual(([e["pick"] for e in kept], [d["time"] for d in dropped]), ([0, 1], [8.0]))
        self.assertIn("every eligible file for whoosh played less than 30s earlier: a.wav at 0.000s", dropped[0]["reason"])
        _, dropped = space(events_at([0.0, 5.0]), 0.25, 30.0, fixed_candidates({"whoosh": ["named.wav"]}, fixed=True))
        self.assertIn("a named --file is never replaced", dropped[0]["reason"])

    def test_off_by_default_and_with_no_candidates(self):
        evs = events_at([0.0, 1.0, 2.0])
        for kept, dropped in (space(evs, 0.25), space(evs, 0.25, None, fixed_candidates({"whoosh": ["a.wav"]})),
                              space(evs, 0.25, 30.0)):
            self.assertEqual(([e["time"] for e in kept], dropped), ([0.0, 1.0, 2.0], []))
            self.assertTrue(all("pick" not in e for e in kept))

    def test_interaction_with_the_minimum_spacing(self):
        table = {"whoosh": ["a.wav", "b.wav"], "pop": ["p.wav"]}
        evs = events_at([0.0, 0.1, 10.0]) + [dict(e, source_index=e["source_index"] + 3)
                                             for e in events_at([20.0, 25.0], effect="pop", kind="emphasis")]
        evs += [dict(events_at([20.1])[0], source_index=5)]
        kept, dropped = space(evs, 0.25, 30.0, fixed_candidates(table))
        # 0.1 s: closer than min_spacing, dropped by that rule first, and never counted as playing b.wav (10 s takes it)
        self.assertEqual([(d["time"], d["by"]) for d in dropped], [(0.1, "min_spacing"), (20.1, "min_spacing"),
                                                                    (25.0, "min_reuse_spacing")])
        self.assertEqual([(e["time"], e["effect"], e["pick"]) for e in kept], [(0.0, "whoosh", 0), (10.0, "whoosh", 1),
                                                                               (20.0, "pop", 0)])
        # a reuse drop is not a cue either: the min_spacing check after it measures from the last kept cue
        kept, dropped = space(events_at([0.0, 5.0]) + [dict(events_at([5.2], effect="pop")[0], source_index=2)], 0.25,
                              30.0, fixed_candidates({"whoosh": ["a.wav"], "pop": ["p.wav"]}))
        self.assertEqual(([e["time"] for e in kept], [(d["time"], d["by"]) for d in dropped]),
                         ([0.0, 5.2], [(5.0, "min_reuse_spacing")]))
        without, _ = space(events_at([0.0, 5.0]) + [dict(events_at([5.2], effect="pop")[0], source_index=2)], 0.25)
        self.assertEqual([e["time"] for e in without], [0.0, 5.0])            # off: 5.2 is 0.2 s after the kept 5.0
        times = [e["time"] for e in kept]
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.25)

    def test_read_events_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([{"time": 1.0, "kind": "emphasis", "character": "Bright"},
                           {"time": 2.0, "kind": "emphasis", "character": ["short", "bright", "SHORT"]},
                           {"time": 3.0, "kind": "emphasis", "character": []}], fh)
            evs = read_events(CuesRequest(events_path=path))
            self.assertEqual([e.get("character") for e in evs], [["bright"], ["bright", "short"], []])
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([{"time": 1.0, "character": 3}], fh)
            with self.assertRaises(SoundDesignError) as e:
                read_events(CuesRequest(events_path=path))
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")


class ReuseAndCharacterSheetTests(unittest.TestCase):
    """sfx-cues end to end with several files per effect (ADR-015)."""

    LIBRARY = {
        "whoosh-a.wav": ({"licence": "CC0-1.0", "kind": "sfx", "tags": ["whoosh"]}, "whoosh", 0.8, 0.4, 1),
        "whoosh-b.wav": ({"licence": "CC0-1.0", "kind": "sfx", "tags": ["whoosh", "falling"]}, "whoosh", 0.8, 0.3, 2),
        "whoosh-c.wav": ({"licence": "CC-BY-4.0", "credit": "SFX: C", "kind": "sfx", "tags": ["whoosh"]}, "whoosh", 0.8, 0.5, 3),
        "pop-a.wav": ({"licence": "CC0-1.0", "kind": "sfx", "tags": ["pop"]}, "hit", 0.3, 0.02, 4),
        "pop-bright.wav": ({"licence": "CC0-1.0", "kind": "sfx", "tags": ["pop", "Bright"]}, "hit", 0.3, 0.03, 5),
        "pop-soft.wav": ({"licence": "CC0-1.0", "kind": "sfx", "tags": ["pop"], "mood": ["soft"]}, "hit", 0.3, 0.04, 6),
        "pop-unlicensed.wav": ({"kind": "sfx", "tags": ["pop", "bright", "short"]}, "hit", 0.3, 0.02, 7),
        "ding-lib.wav": ({"licence": "CC0-1.0", "kind": "sfx", "tags": ["ding"]}, "hit", 0.5, 0.01, 8),
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.lib = self.library(os.path.join(self.dir, "sfx"), sorted(self.LIBRARY))
        self.skill = ts.StubSkill()

    def library(self, path, order):
        os.mkdir(path)
        for name in order:
            _, kind, secs, at, seed = self.LIBRARY[name]
            write_burst_wav(os.path.join(path, name), kind, secs, at=at, rate=RATE, seed=seed)
        tl.manifest(path, {name: dict(self.LIBRARY[name][0]) for name in order})
        return path

    def cues(self, **kw):
        kw.setdefault("library_dir", self.lib)
        return sfx_cues(CuesRequest(**kw), skill=self.skill)

    def test_default_off_keeps_the_first_ranked_file(self):
        doc = self.cues(cuts=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual([c["file"] for c in doc["cues"]], ["whoosh-a.wav"] * 4)        # the same file, 1 s apart
        self.assertEqual((doc["min_reuse_spacing"], doc["substitutions"], doc["dropped"]), (None, [], []))
        self.assertEqual(list(doc["sources"]), ["whoosh"])
        self.assertTrue(all(c["source"] == "whoosh" and "substituted" not in c for c in doc["cues"]))
        self.assertEqual(doc["summary"], {"events": 4, "cues": 4, "dropped": 0, "smallest_gap": 1.0})
        explicit = self.cues(cuts=[1.0, 2.0, 3.0, 4.0], min_reuse_spacing=None)
        self.assertEqual(json.dumps(explicit, sort_keys=True), json.dumps(doc, sort_keys=True))

    def test_rotation_through_the_library_ranking(self):
        doc = self.cues(cuts=[0.5, 10.0, 20.0, 30.5, 40.0, 75.0], min_reuse_spacing=30)
        self.assertEqual([c["file"] for c in doc["cues"]],
                         ["whoosh-a.wav", "whoosh-b.wav", "whoosh-c.wav", "whoosh-a.wav", "whoosh-b.wav", "whoosh-a.wav"])
        self.assertEqual([c["time"] for c in doc["cues"]], [0.5, 10.0, 20.0, 30.5, 40.0, 75.0])   # none moved or dropped
        self.assertEqual([(s["cue"], s["time"], s["file"], s["instead_of"], s["rank"]) for s in doc["substitutions"]],
                         [(1, 10.0, "whoosh-b.wav", "whoosh-a.wav", 2), (2, 20.0, "whoosh-c.wav", "whoosh-a.wav", 3),
                          (4, 40.0, "whoosh-b.wav", "whoosh-a.wav", 2)])
        self.assertIn("whoosh-a.wav at 0.500s (9.500s earlier)", doc["substitutions"][0]["reason"])
        self.assertEqual(doc["cues"][1]["substituted"], {"for": "whoosh-a.wav", "rank": 2,
                                                         "reason": doc["substitutions"][0]["reason"]})
        self.assertEqual(sorted(doc["sources"]), ["whoosh", "whoosh@2", "whoosh@3"])
        self.assertEqual([(doc["sources"][k]["file"], doc["sources"][k]["rank"]) for k in ("whoosh", "whoosh@2", "whoosh@3")],
                         [("whoosh-a.wav", 1), ("whoosh-b.wav", 2), ("whoosh-c.wav", 3)])
        for c in doc["cues"]:                                     # each cue measured on its own file
            src = doc["sources"][c["source"]]
            self.assertEqual((c["file"], c["anchor_in_file"]), (src["file"], src["anchors"]["peak"]))
            self.assertAlmostEqual(c["start"] + c["anchor_in_file"], c["time"], places=6)
            self.assertAlmostEqual(c["gain_db"], -12.0 - src["anchors"]["sample_peak_dbfs"], places=2)
        self.assertNotEqual(doc["sources"]["whoosh"]["anchors"]["peak"], doc["sources"]["whoosh@2"]["anchors"]["peak"])
        self.assertEqual(doc["sources"]["whoosh@3"]["licence"]["licence"], "CC-BY-4.0")    # a substitute's licence travels
        self.assertEqual(doc["min_reuse_spacing"], 30.0)

    def test_fallback_drop_when_no_other_file_is_eligible(self):
        doc = self.cues(emphasis=[1.0, 5.0, 40.0], effects={"emphasis": "ding"}, min_reuse_spacing=30.0)
        self.assertEqual(([c["time"] for c in doc["cues"]], [(d["time"], d["by"]) for d in doc["dropped"]]),
                         ([1.0, 40.0], [(5.0, "min_reuse_spacing")]))
        self.assertIn("ding-lib.wav at 1.000s (4.000s earlier), the only eligible file for ding", doc["dropped"][0]["reason"])
        self.assertEqual(doc["summary"]["dropped"], 1)
        named = write_burst_wav(os.path.join(self.dir, "ding.wav"), "hit", 0.4, at=0.05, rate=RATE, seed=9)
        doc = self.cues(emphasis=[1.0, 5.0, 40.0], effects={"emphasis": "ding"}, files={"ding": named}, min_reuse_spacing=30.0)
        self.assertEqual([(c["time"], c["file"], c["from"]) for c in doc["cues"]], [(1.0, "ding.wav", "file"),
                                                                                   (40.0, "ding.wav", "file")])
        self.assertEqual([d["time"] for d in doc["dropped"]], [5.0])          # never replaced by the library's ding
        self.assertIn("a named --file is never replaced", doc["dropped"][0]["reason"])
        self.assertEqual(doc["substitutions"], [])

    def test_the_same_inputs_give_the_same_sheet(self):
        req = dict(cuts=[0.5, 10.0, 20.0, 21.0, 30.5], emphasis=[3.0, 4.0, 33.0], min_reuse_spacing=30.0,
                   characters={"pop": "bright"}, min_spacing=0.25)
        first = json.dumps(self.cues(**req), sort_keys=True)
        for _ in range(2):
            self.assertEqual(json.dumps(self.cues(**req), sort_keys=True), first)
        # the same library written in the opposite order, and the same times given in the opposite order
        other = self.library(os.path.join(self.dir, "sfx-reversed"), sorted(self.LIBRARY, reverse=True))
        again = self.cues(**dict(req, library_dir=other, cuts=req["cuts"][::-1], emphasis=req["emphasis"][::-1]))
        pick = lambda d: [(c["time"], c["file"], c["source"]) for c in d["cues"]]    # noqa: E731
        self.assertEqual(pick(again), pick(json.loads(first)))
        self.assertEqual([(d["time"], d["by"]) for d in again["dropped"]],
                         [(d["time"], d["by"]) for d in json.loads(first)["dropped"]])

    def test_character_filter_matches_tags_and_mood_only(self):
        before = self._manifest_text()
        doc = self.cues(emphasis=[1.0, 2.0], characters={"pop": ["BRIGHT"]})
        self.assertEqual([(c["file"], c["source"], c["character"]) for c in doc["cues"]],
                         [("pop-bright.wav", "pop[bright]", ["bright"])] * 2)        # never the unlicensed one
        self.assertEqual(doc["sources"]["pop[bright]"]["character"], ["bright"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([{"time": 1.0, "kind": "emphasis", "character": "soft"},       # mood
                           {"time": 2.0, "kind": "emphasis", "character": []},           # clears the effect's word
                           {"time": 3.0, "kind": "emphasis"},
                           {"time": 4.0, "kind": "cut", "character": ["falling"]}], fh)
            doc = self.cues(events_path=path, characters={"pop": "bright"})
        self.assertEqual([(c["time"], c["file"]) for c in doc["cues"]],
                         [(1.0, "pop-soft.wav"), (2.0, "pop-a.wav"), (3.0, "pop-bright.wav"), (4.0, "whoosh-b.wav")])
        self.assertNotIn("character", doc["cues"][1])
        for bad in ({"pop": ["bright", "short"]}, {"pop": "loud"}):           # every word is required; none invented
            with self.subTest(bad=bad), self.assertRaises(SoundDesignError) as e:
                self.cues(emphasis=[1.0], characters=bad)
            self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")
            self.assertIn("in its tags or mood", e.exception.message)
        self.assertEqual(self._manifest_text(), before)                       # no file gained a tag
        named = write_burst_wav(os.path.join(self.dir, "pop.wav"), "hit", 0.3, at=0.02, rate=RATE)
        for kw in ({"characters": {"pop": "bright"}}, {"events_path": self._events([{"time": 1.0, "kind": "emphasis",
                                                                                     "character": "bright"}])}):
            with self.subTest(kw=list(kw)), self.assertRaises(SoundDesignError) as e:
                self.cues(emphasis=[1.0], files={"pop": named}, **kw)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        doc = self.cues(cuts=[1.0], characters={"ding": "bright"})
        self.assertTrue(any("no event uses the effect ding" in w for w in doc["warnings"]), doc["warnings"])

    def test_character_and_reuse_together(self):
        doc = self.cues(cuts=[1.0, 5.0, 40.0], characters={"whoosh": "falling"}, min_reuse_spacing=30.0)
        self.assertEqual([c["file"] for c in doc["cues"]], ["whoosh-b.wav", "whoosh-b.wav"])   # the only falling whoosh
        self.assertEqual([(d["time"], d["by"]) for d in doc["dropped"]], [(5.0, "min_reuse_spacing")])
        # a pop and a bright pop are separate rankings, but one file played through either is the same sound
        doc = self.cues(events_path=self._events([{"time": 1.0, "kind": "emphasis"},
                                                  {"time": 2.0, "kind": "emphasis", "character": "bright"}]),
                        min_reuse_spacing=30.0)
        self.assertEqual([c["file"] for c in doc["cues"]], ["pop-a.wav", "pop-bright.wav"])
        doc = self.cues(events_path=self._events([{"time": 1.0, "kind": "emphasis", "character": "bright"},
                                                  {"time": 2.0, "kind": "emphasis"}]), min_reuse_spacing=30.0)
        self.assertEqual([c["file"] for c in doc["cues"]], ["pop-bright.wav", "pop-a.wav"])
        doc = self.cues(events_path=self._events([{"time": 1.0, "kind": "emphasis"},
                                                  {"time": 2.0, "kind": "emphasis", "character": "bright"},
                                                  {"time": 3.0, "kind": "emphasis"}]), min_reuse_spacing=30.0)
        self.assertEqual([(c["file"], c["source"]) for c in doc["cues"]],
                         [("pop-a.wav", "pop"), ("pop-bright.wav", "pop[bright]"), ("pop-soft.wav", "pop@3")])

    def test_probes_are_shared_between_pop_selectors(self):
        doc = self.cues(events_path=self._events([{"time": 1.0, "kind": "emphasis"},
                                                  {"time": 2.0, "kind": "emphasis", "character": "bright"},
                                                  {"time": 3.0, "kind": "emphasis", "character": "soft"}]),
                        min_reuse_spacing=30.0)
        self.assertEqual([c["file"] for c in doc["cues"]], ["pop-a.wav", "pop-bright.wav", "pop-soft.wav"])
        probes = [name for op, name in ((c[0], c[1]) for c in self.skill.calls) if op == "probe"]
        self.assertEqual(sorted(probes), ["pop-a.wav", "pop-bright.wav", "pop-soft.wav"])   # once each, three selectors

    def test_reuse_spacing_values_and_the_cli(self):
        for bad in (-1, "30", True, float("nan"), 86400.5):
            with self.subTest(bad=bad), self.assertRaises(SoundDesignError) as e:
                self.cues(cuts=[1.0], min_reuse_spacing=bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        with mock.patch("sound_design_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("sfx-cues", "--cuts", "0.5,10,20,30.5", "--emphasis", "3,4", "--library", self.lib,
                              "--min-reuse-spacing", "30", "--character", "pop=bright,short", "--json")
            self.assertEqual((rc, doc["error"]["code"]), (2, "INPUT_NOT_FOUND"))
            rc, doc = run_cli("sfx-cues", "--cuts", "0.5,10,20,30.5", "--emphasis", "3,4", "--library", self.lib,
                              "--min-reuse-spacing", "30", "--character", "pop=Bright", "--json")
            self.assertEqual(rc, 0, doc)
            rc, empty = run_cli("sfx-cues", "--emphasis", "3", "--library", self.lib, "--character", "pop=,", "--json")
            self.assertEqual((rc, empty["error"]["code"]), (2, "INVALID_ARGUMENT"))     # no word: refused, not ignored
            rc, both = run_cli("sfx-cues", "--emphasis", "3", "--library", self.lib, "--character", "pop=short",
                               "--character", "pop=bright", "--json")
            self.assertEqual((rc, both["error"]["code"]), (2, "INPUT_NOT_FOUND"))        # words add up: short AND bright
            self.assertEqual(both["error"]["details"]["character"], ["bright", "short"])
        self.assertEqual([c["file"] for c in doc["cues"]],
                         ["whoosh-a.wav", "pop-bright.wav", "whoosh-b.wav", "whoosh-c.wav", "whoosh-a.wav"])
        self.assertEqual([(d["time"], d["by"]) for d in doc["dropped"]], [(4.0, "min_reuse_spacing")])
        self.assertEqual(len(doc["substitutions"]), 2)
        tool = next(t for t in contract()["tools"] if t["tool_id"] == "sound-design/sfx-cues")
        self.assertIn("[--min-reuse-spacing <seconds>]", tool["argv"])
        self.assertIn("[--character <effect>=<word,...>]", tool["argv"])
        self.assertEqual((tool["reuse"]["default"], tool["reuse"]["request_key"]), (None, "min_reuse_spacing"))
        self.assertEqual((tool["character"]["vocabulary"], tool["character"]["vocabulary_closed"]),
                         (list(CHARACTER_VOCABULARY), False))
        skill_md = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        for word in CHARACTER_VOCABULARY:                                   # the vocabulary is documented where agents read
            self.assertIn(f"`{word}`", skill_md)

    def _manifest_text(self):
        with open(os.path.join(self.lib, tl.MANIFEST_NAME), encoding="utf-8") as fh:
            return fh.read()

    def _events(self, rows):
        fd, path = tempfile.mkstemp(suffix=".json", dir=self.dir)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        return path


if __name__ == "__main__":
    unittest.main()
