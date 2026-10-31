"""Unit tests for library mode (roadmap step 4, ADR-008): licence reporting (a missing licence is "unknown", never
cleared, refused with LICENCE_REQUIRED), the fitting maths (trim, crossfaded loop, refusals), deterministic selection,
indexing (Japanese names, NFD / NFC keys, hidden files, changed files), import, pick end to end through a stub
ffmpeg-skill that edits PCM WAVs with the wave module, the provenance, and that no engine (EzAudio or ACE-Step) is ever
reached. No FFmpeg, no engine and no network are needed; the integration tests run the same through the real
ffmpeg-skill."""
from __future__ import annotations

import array
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unicodedata
import unittest
import wave
from pathlib import Path
from unittest import mock

from helpers import ROOT, run_cli, write_tone_wav

from sound_design_skill import library, ops, service
from sound_design_skill.contract import contract
from sound_design_skill.engines import acestep, ezaudio
from sound_design_skill.errors import SoundDesignError
from sound_design_skill.ffmpeg_skill import FfmpegSkill, SkillInfo
from sound_design_skill.library import (MANIFEST_NAME, LibraryImportRequest, LibraryIndexRequest, LibraryPickRequest,
                                        library_import, library_index, library_pick, licence_info, licence_warnings,
                                        plan_fit, rank, require_licence, screen)


def _read(path):
    with wave.open(path) as w:
        return w.getparams(), array.array("h", w.readframes(w.getnframes()))


def _write(path, params, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(samples.tobytes())


class StubLibrarySkill:
    """ffmpeg-skill stand-in: probe, cut, join (linear crossfade) and loudness on 16-bit PCM WAVs, in pure Python."""

    def __init__(self):
        self.calls = []

    def info(self):
        return SkillInfo(Path("stub"), "2.23.1", "1.0", {t: {} for t in ("probe", "cut", "join", "loudness")}, [])

    def probe(self, path):
        self.calls.append(("probe", os.path.basename(path)))
        try:
            with wave.open(path) as w:
                rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        except (wave.Error, EOFError, OSError) as e:
            raise SoundDesignError("UNSUPPORTED_MEDIA", f"stub probe: {e}")
        return {"duration": n / rate, "video": None, "stream_counts": {"video": 0, "audio": 1},
                "audio": {"codec": "pcm_s16le", "sample_rate": rate, "channels": ch}}

    def cut(self, src, dst, duration, start=0.0, timeout=None):
        self.calls.append(("cut", round(duration, 6)))
        p, s = _read(src)
        a, b = int(round(start * p.framerate)), int(round((start + duration) * p.framerate))
        _write(dst, p, s[a * p.nchannels:b * p.nchannels])
        return {"status": "completed"}

    def join(self, inputs, dst, crossfade, timeout=None):
        self.calls.append(("join", len(inputs), crossfade))
        p, out = _read(inputs[0])
        for path in inputs[1:]:
            _, nxt = _read(path)
            n = int(round(crossfade * p.framerate)) * p.nchannels
            if n:
                head, tail = out[:-n], out[-n:]
                mixed = array.array("h", (int(t * (1 - i / n) + h * (i / n)) for i, (t, h) in enumerate(zip(tail, nxt[:n]))))
                out = head + mixed + nxt[n:]
            else:
                out = out + nxt
        _write(dst, p, out)
        return {"status": "completed"}

    def loudness(self, path, timeout=None):
        self.calls.append(("loudness", os.path.basename(path)))
        _, s = _read(path)
        rms = math.sqrt(sum(x * x for x in s) / max(1, len(s))) or 1e-9
        return {"integrated_lufs": round(20 * math.log10(rms / 32768.0), 2), "true_peak_dbtp": None, "lra_lu": 0.0,
                "tool": "ffmpeg-skill/loudness --measure-only"}


def manifest(lib, files):
    with open(os.path.join(lib, MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump({"schema": library.MANIFEST_SCHEMA, "files": files}, fh, ensure_ascii=False)


class LicenceTests(unittest.TestCase):
    def test_missing_or_placeholder_licences_are_unknown_and_never_usable(self):
        none = licence_info(None)
        self.assertEqual((none["licence"], none["status"], none["usable"], none["verified"], none["declared_in"]),
                         ("unknown", "unknown", False, False, None))
        for value in (None, "", "   ", "unknown", "Unknown", "N/A", "none", "?", 123, ["CC0-1.0"]):
            info = licence_info({"licence": value} if value is not None else {"kind": "bgm"})
            self.assertEqual((info["licence"], info["status"], info["usable"]), ("unknown", "unknown", False), value)
            self.assertFalse(info["verified"])
            self.assertEqual(info["declared_in"], MANIFEST_NAME)
            self.assertEqual(info["declared_as"], value if isinstance(value, str) else None)   # kept as written, never replaced
            self.assertNotIn("cleared", json.dumps(info).replace("does not verify or clear", ""))

    def test_a_declared_licence_is_recorded_as_declared_and_not_verified(self):
        info = licence_info({"licence": " CC-BY-4.0 ", "credit": "Music: 山田太郎", "source": "https://example.invalid/x",
                             "author": "山田太郎"})
        self.assertEqual((info["licence"], info["status"], info["usable"], info["verified"]), ("CC-BY-4.0", "declared", True, False))
        self.assertEqual((info["credit"], info["author"]), ("Music: 山田太郎", "山田太郎"))
        self.assertIn("does not verify or clear", info["note"])

    def test_warnings_state_facts_of_the_declared_text(self):
        self.assertEqual(licence_warnings("a.wav", licence_info(None)), ["a.wav: licence unknown; not used"])
        nc = licence_warnings("a.wav", licence_info({"licence": "CC-BY-NC-4.0", "credit": "x"}))
        self.assertTrue(any("non-commercial" in w for w in nc), nc)
        nd = {"licence": "CC-BY-ND-4.0", "credit": "x"}
        self.assertEqual(licence_warnings("a.wav", licence_info(nd)), [])
        self.assertTrue(any("no-derivatives" in w for w in licence_warnings("a.wav", licence_info(nd), fitted=True)))
        self.assertTrue(any("no credit" in w for w in licence_warnings("a.wav", licence_info({"licence": "CC BY 4.0"}))))
        self.assertEqual(licence_warnings("a.wav", licence_info({"licence": "Purchased from Sync Inc., standard licence"})), [])
        self.assertEqual(licence_warnings("a.wav", licence_info({"licence": "CC0-1.0"})), [])

    def test_require_licence_refuses_in_english_and_japanese(self):
        for value in (None, "", "unknown"):
            with self.assertRaises(SoundDesignError) as e:
                require_licence(value, "朝.wav")
            err = e.exception
            self.assertEqual((err.code, err.exit_code, err.retryable), ("LICENCE_REQUIRED", 2, False))
            self.assertIn("--licence", err.details["message_en"])
            self.assertIn("ライセンス", err.details["message_ja"])
            self.assertEqual(err.details["licence"], "unknown")
        self.assertEqual(require_licence("CC0-1.0", "a.wav")["status"], "declared")


class FitTests(unittest.TestCase):
    def test_no_target_or_within_tolerance_is_used_whole(self):
        self.assertEqual(plan_fit(7.3, None, False)["mode"], "none")
        p = plan_fit(30.2, 30.0, False)                    # tolerance max(0.1, 1 %) = 0.3
        self.assertEqual((p["mode"], p["output_duration"], p["tolerance"]), ("none", 30.2, 0.3))
        self.assertEqual(plan_fit(10.05, 10.0, False)["mode"], "none")
        self.assertEqual(plan_fit(10.2, 10.0, False)["mode"], "trim")

    def test_trim(self):
        p = plan_fit(30.0, 12.0, True)
        self.assertEqual((p["mode"], p["trimmed"], p["output_duration"]), ("trim", 18.0, 12.0))

    def test_loop_repeats_and_crossfade(self):
        p = plan_fit(4.0, 10.0, True, crossfade=0.5)
        self.assertEqual((p["mode"], p["repeats"], p["crossfade"], p["joined_duration"], p["trimmed"]), ("loop", 3, 0.5, 11.0, 1.0))
        self.assertEqual(plan_fit(1.0, 3.0, True, crossfade=0.5)["crossfade"], 0.25)      # at most a quarter of the file
        exact = plan_fit(5.0, 10.0, True, crossfade=0.0)
        self.assertEqual((exact["repeats"], exact["joined_duration"], exact["trimmed"]), (2, 10.0, 0.0))

    def test_loop_uses_the_fewest_repeats_that_reach_the_target(self):
        for source, target, x in itertools.product((0.4, 1.0, 2.5, 4.0, 7.3), (1.0, 3.0, 10.0, 25.0), (0.0, 0.2, 0.5, 2.0)):
            p = plan_fit(source, target, True, crossfade=x)
            if p["mode"] != "loop":
                continue
            n, xf = p["repeats"], p["crossfade"]
            self.assertLessEqual(xf, source / 4 + 1e-9)
            self.assertGreaterEqual(p["joined_duration"] + 1e-6, target)
            self.assertAlmostEqual(p["joined_duration"], n * source - (n - 1) * xf, places=5)
            if n > 2:
                self.assertLess((n - 1) * source - (n - 2) * xf, target)

    def test_refusals(self):
        self.assertIsNone(plan_fit(4.0, 10.0, False)["mode"])
        self.assertIn("not marked loopable", plan_fit(4.0, 10.0, False)["reason"])
        many = plan_fit(0.5, 600.0, True)
        self.assertIsNone(many["mode"])
        self.assertIn("repeats", many["reason"])
        self.assertIsNone(plan_fit(0.0, 10.0, True)["mode"])


def _row(key, duration, matched=(), loopable=False, licence="CC0-1.0", sha="0"):
    entry = {"licence": licence, "kind": "bgm", "loopable": loopable}
    return {"key": key, "path": key, "entry": entry, "licence": licence_info(entry), "matched": list(matched),
            "duration": duration, "sha256": sha}


class SelectionTests(unittest.TestCase):
    def test_screen_reports_every_exclusion(self):
        files = {k: k for k in ("a.wav", "b.wav", "c.wav", "d.wav", "e.wav")}
        entries = {"a.wav": {"licence": "CC0-1.0", "kind": "bgm", "tags": ["Calm", "piano"]},
                   "b.wav": {"licence": "unknown", "kind": "bgm", "tags": ["calm"]},
                   "c.wav": {"licence": "CC0-1.0", "kind": "sfx", "tags": ["calm"]},
                   "d.wav": {"licence": "CC0-1.0", "kind": "bgm", "tags": ["rock"], "mood": ["ｃａｌｍ"]}}
        kept, excluded = screen(files, entries, "bgm", ["calm"], [])
        self.assertEqual([r["key"] for r in kept], ["a.wav"])
        reasons = {x["file"]: x["reasons"] for x in excluded}
        self.assertEqual(reasons["b.wav"], ["licence unknown"])
        self.assertEqual(reasons["c.wav"], ["kind 'sfx', not 'bgm'"])
        self.assertEqual(reasons["d.wav"], ["no requested tag or mood"])
        self.assertEqual(reasons["e.wav"], ["no manifest entry (licence unknown)"])
        kept2, _ = screen(files, entries, "bgm", [], ["Calm"])              # moods match width- and case-insensitively
        self.assertEqual([r["key"] for r in kept2], ["d.wav"])

    def test_rank_order_and_tie_breaks(self):
        rows = [_row("z.wav", 30.0, ["calm"]), _row("y.wav", 12.0, ["calm"]), _row("x.wav", 50.0, ["calm", "piano"]),
                _row("w.wav", 5.0, ["calm"], loopable=True), _row("v.wav", 20.0, ["calm"]), _row("u.wav", 20.0, ["calm"]),
                _row("t.wav", 3.0, ["calm"])]
        ranked, excluded = rank(rows, 12.0)
        self.assertEqual([r["key"] for r in ranked], ["x.wav", "y.wav", "u.wav", "v.wav", "z.wav", "w.wav"])
        self.assertEqual([r["fit"]["mode"] for r in ranked], ["trim", "none", "trim", "trim", "trim", "loop"])
        self.assertEqual([x["file"] for x in excluded], ["t.wav"])

    def test_the_choice_does_not_depend_on_the_order(self):
        rows = [_row(f"{n}.wav", d, m, loopable=True, sha=str(i)) for i, (n, d, m) in
                enumerate([("朝", 8.0, ["calm"]), ("夜", 8.0, ["calm"]), ("b", 15.0, ["calm"]), ("a", 4.0, [])])]
        first = [r["key"] for r in rank(rows, 10.0)[0]]
        for perm in itertools.permutations(rows):
            self.assertEqual([r["key"] for r in rank([dict(r) for r in perm], 10.0)[0]], first)
        self.assertEqual(first[0], "b.wav")


class LibraryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.lib = os.path.join(self.dir, "ライブラリ")
        os.mkdir(self.lib)
        self.out_dir = os.path.join(self.dir, "out")
        os.mkdir(self.out_dir)
        self.skill = StubLibrarySkill()
        # library mode never reaches an engine: every entry point to one fails loudly if it is called
        for target, name in ((ops, "run_engine"), (ops, "select_engine"), (ops, "require_confirmation"), (service, "run_engine"),
                             (ezaudio.EzAudioEngine, "inspect"), (ezaudio.EzAudioEngine, "process"),
                             (acestep.AceStepEngine, "inspect"), (acestep.AceStepEngine, "process")):
            p = mock.patch.object(target, name, side_effect=AssertionError(f"library mode reached {name}"))
            p.start()
            self.addCleanup(p.stop)

    def tone(self, rel, seconds, rate=8000, channels=1, freq=440.0):
        path = os.path.join(self.lib, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_tone_wav(path, seconds=seconds, rate=rate, freq=freq)
        if channels == 2:
            p, s = _read(path)
            stereo = array.array("h", (v for x in s for v in (x, x)))
            with wave.open(path, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(stereo.tobytes())
        return path

    def listing(self):
        return sorted(str(p.relative_to(self.dir)) for p in Path(self.dir).rglob("*"))


class IndexTests(LibraryCase):
    def test_index_lists_every_audio_file_with_its_licence_status(self):
        self.tone("bgm/朝のピアノ.wav", 2.0)
        nfd = unicodedata.normalize("NFD", "ドア.wav")             # a macOS-style name; the manifest key is NFC
        self.tone("sfx/" + nfd, 0.5)
        self.tone("bgm/不明.wav", 1.0)
        self.tone(".hidden/x.wav", 1.0)
        self.tone(".x.wav", 1.0)
        Path(self.lib, "notes.txt").write_text("not audio", encoding="utf-8")
        Path(self.lib, "broken.wav").write_bytes(b"RIFF....")
        manifest(self.lib, {"bgm/朝のピアノ.wav": {"licence": "CC0-1.0", "kind": "bgm", "tags": ["Piano"], "mood": ["calm"],
                                                  "loopable": True},
                            "sfx/ドア.wav": {"licence": "Purchased: Acme SFX pack, standard licence", "kind": "sfx",
                                            "sha256": "0" * 64},
                            "bgm/不明.wav": {"licence": "unknown", "kind": "bgm"},
                            "gone.wav": {"licence": "CC0-1.0", "kind": "sfx"}})
        before = self.listing()
        doc = library_index(LibraryIndexRequest(self.lib, loudness=True), skill=self.skill)
        self.assertEqual(self.listing(), before)                                   # read-only
        self.assertEqual(doc["schema"], library.INDEX_SCHEMA)
        self.assertEqual([f["file"] for f in doc["files"]], ["bgm/不明.wav", "bgm/朝のピアノ.wav", "broken.wav", "sfx/ドア.wav"])
        by = {f["file"]: f for f in doc["files"]}
        self.assertEqual(by["bgm/朝のピアノ.wav"]["licence"]["licence"], "CC0-1.0")
        self.assertEqual((by["bgm/朝のピアノ.wav"]["tags"], by["bgm/朝のピアノ.wav"]["loopable"]), (["piano"], True))
        self.assertAlmostEqual(by["bgm/朝のピアノ.wav"]["measured"]["duration"], 2.0, places=3)
        self.assertIsNotNone(by["bgm/朝のピアノ.wav"]["loudness"]["integrated_lufs"])
        for key in ("bgm/不明.wav", "broken.wav"):
            self.assertEqual((by[key]["licence"]["licence"], by[key]["licence"]["status"]), ("unknown", "unknown"))
        self.assertFalse(by["broken.wav"]["in_manifest"])
        self.assertFalse(by["broken.wav"]["readable"])
        self.assertTrue(by["sfx/ドア.wav"]["changed_since_import"])
        self.assertEqual(doc["unknown_licence"], ["bgm/不明.wav", "broken.wav"])
        self.assertEqual(doc["missing_files"], ["gone.wav"])
        self.assertEqual(doc["summary"], {"files": 4, "declared": 2, "unknown_licence": 2, "unreadable": 1, "missing_files": 1})
        self.assertIsNone(doc["engine"])
        json.dumps(doc, ensure_ascii=True, allow_nan=False)

    def test_manifest_with_a_byte_order_mark_and_invalid_manifests(self):
        self.tone("a.wav", 1.0)
        Path(self.lib, MANIFEST_NAME).write_bytes(b"\xef\xbb\xbf" + json.dumps({"files": {"a.wav": {"licence": "CC0-1.0"}}}).encode())
        doc = library_index(LibraryIndexRequest(self.lib), skill=self.skill)
        self.assertEqual(doc["files"][0]["licence"]["status"], "declared")
        for bad in ("{", json.dumps({"files": []}), json.dumps({"files": {"a.wav": "CC0"}}),
                    json.dumps({"files": {"a.wav": {}, "./a.wav/": {}}})):
            Path(self.lib, MANIFEST_NAME).write_text(bad, encoding="utf-8")
            with self.assertRaises(SoundDesignError) as e:
                library_index(LibraryIndexRequest(self.lib), skill=self.skill)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", bad)

    def test_a_missing_library_is_input_not_found(self):
        with self.assertRaises(SoundDesignError) as e:
            library_index(LibraryIndexRequest(os.path.join(self.dir, "none")), skill=self.skill)
        self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")


class ImportTests(LibraryCase):
    def src(self, name="持ち込み.wav", seconds=2.0):
        return write_tone_wav(os.path.join(self.dir, name), seconds=seconds, rate=8000)

    def imp(self, **kw):
        kw.setdefault("input_path", self.src())
        kw.setdefault("library_dir", self.lib)
        kw.setdefault("licence", "CC-BY-4.0")
        kw.setdefault("kind", "bgm")
        return library_import(LibraryImportRequest(**kw), skill=self.skill)

    def test_a_missing_licence_is_refused_before_anything_runs(self):
        src = self.src()
        before = self.listing()
        with mock.patch.object(FfmpegSkill, "locate", side_effect=AssertionError("ffmpeg-skill reached")):
            for value in (None, "", " ", "unknown", "N/A"):
                with self.assertRaises(SoundDesignError) as e:
                    library_import(LibraryImportRequest(src, self.lib, value, "bgm"))
                self.assertEqual(e.exception.code, "LICENCE_REQUIRED", value)
        self.assertEqual(self.listing(), before)
        self.assertEqual(self.skill.calls, [])

    def test_import_copies_measures_and_records_the_declared_licence(self):
        src = self.src()
        sha = library.sha256_file(src)
        doc = self.imp(input_path=src, tags=["Piano", "piano", " calm "], mood=["明るい"], loopable=True, credit="Music: 山田",
                       source="https://example.invalid/track", author="山田")
        self.assertEqual(doc["schema"], library.IMPORT_SCHEMA)
        self.assertEqual((doc["file"]["key"], doc["file"]["copied"], doc["file"]["sha256"]), ("持ち込み.wav", True, sha))
        self.assertEqual(library.sha256_file(os.path.join(self.lib, "持ち込み.wav")), sha)
        self.assertEqual(library.sha256_file(src), sha)                         # the input is untouched
        self.assertEqual((doc["licence"]["licence"], doc["licence"]["status"], doc["licence"]["verified"]),
                         ("CC-BY-4.0", "declared", False))
        self.assertEqual(doc["warnings"], [])
        self.assertIsNone(doc["engine"])
        with open(os.path.join(self.lib, MANIFEST_NAME), encoding="utf-8") as fh:
            m = json.load(fh)
        e = m["files"]["持ち込み.wav"]
        self.assertEqual((e["licence"], e["kind"], e["tags"], e["mood"], e["loopable"], e["credit"]),
                         ("CC-BY-4.0", "bgm", ["piano", "calm"], ["明るい"], True, "Music: 山田"))
        self.assertAlmostEqual(e["measured"]["duration"], 2.0, places=3)
        self.assertEqual(e["imported_from"], "持ち込み.wav")                    # a name, never the local path
        self.assertNotIn(self.dir, json.dumps(m, ensure_ascii=False))
        self.assertEqual([n for n in os.listdir(self.lib) if n.startswith(".")], [])
        with self.assertRaises(SoundDesignError) as again:
            self.imp(input_path=src)
        self.assertEqual(again.exception.code, "OUTPUT_EXISTS")
        self.assertEqual(self.imp(input_path=src, licence="CC0-1.0", overwrite=True)["entry"]["licence"], "CC0-1.0")

    def test_register_in_place_and_name_inside_the_library(self):
        path = self.tone("既存/曲.wav", 1.0)
        doc = self.imp(input_path=path, licence="Purchased: AudioStock licence #123")
        self.assertEqual((doc["file"]["key"], doc["file"]["copied"]), ("既存/曲.wav", False))
        named = self.imp(name="sfx/扉.wav", kind="sfx")
        self.assertEqual(named["file"]["key"], "sfx/扉.wav")
        self.assertTrue(os.path.isfile(os.path.join(self.lib, "sfx", "扉.wav")))
        for bad in ("../escape.wav", "/abs.wav", "a/../../b.wav", "C:/x.wav", "x.txt"):
            with self.assertRaises(SoundDesignError) as e:
                self.imp(name=bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", bad)

    def test_input_checks(self):
        with self.assertRaises(SoundDesignError) as e:
            self.imp(input_path=os.path.join(self.dir, "none.wav"))
        self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")
        with self.assertRaises(SoundDesignError) as e:
            self.imp(kind="music")
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        bad = os.path.join(self.dir, "fake.wav")
        Path(bad).write_bytes(b"not audio")
        with self.assertRaises(SoundDesignError) as e:
            self.imp(input_path=bad)
        self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")
        self.assertFalse(os.path.exists(os.path.join(self.lib, "fake.wav")))


class PickTests(LibraryCase):
    def setUp(self):
        super().setUp()
        self.tone("bgm/朝のピアノ.wav", 4.0, channels=2)
        self.tone("bgm/長い曲.wav", 20.0, freq=330.0)
        self.tone("bgm/不明.wav", 30.0)
        self.tone("sfx/ドア.wav", 0.8)
        manifest(self.lib, {"bgm/朝のピアノ.wav": {"licence": "CC0-1.0", "kind": "bgm", "tags": ["piano"], "mood": ["calm"],
                                                  "loopable": True},
                            "bgm/長い曲.wav": {"licence": "CC-BY-4.0", "kind": "bgm", "tags": ["corporate"], "credit": "Music: A"},
                            "bgm/不明.wav": {"kind": "bgm", "tags": ["piano", "calm"]},
                            "sfx/ドア.wav": {"licence": "CC0-1.0", "kind": "sfx", "tags": ["door"]}})

    def pick(self, name="bed.wav", **kw):
        kw.setdefault("kind", "bgm")
        return library_pick(LibraryPickRequest(self.lib, os.path.join(self.out_dir, name), **kw), skill=self.skill)

    def assert_clean(self):
        for d in (self.lib, self.out_dir):
            self.assertEqual([n for n in os.listdir(d) if n.startswith(".")], [])

    def test_loop_to_a_longer_duration_with_provenance(self):
        doc = self.pick(tags=["piano"], mood=["calm"], duration=10.0)
        self.assertEqual(doc["schema"], library.PICK_SCHEMA)
        self.assertEqual(doc["selection"]["chosen"], "bgm/朝のピアノ.wav")
        self.assertNotIn("bgm/不明.wav", [r["file"] for r in doc["selection"]["ranked"]])
        unknown = next(x for x in doc["selection"]["excluded"] if x["file"] == "bgm/不明.wav")
        self.assertEqual((unknown["licence"], unknown["reasons"]), ("unknown", ["licence unknown"]))
        self.assertEqual((doc["fit"]["mode"], doc["fit"]["repeats"], doc["fit"]["crossfade"]), ("loop", 3, 0.5))
        self.assertAlmostEqual(doc["output"]["duration"], 10.0, delta=0.01)
        self.assertEqual((doc["output"]["sample_rate"], doc["output"]["channels"]), (8000, 2))
        self.assertIsNotNone(doc["output"]["loudness"]["integrated_lufs"])
        self.assertIn(("join", 3, 0.5), self.skill.calls)
        p = doc["provenance"]
        self.assertEqual((p["op"], p["engine"], p["ai_generated"], p["network"]), ("library", None, False, False))
        self.assertEqual((p["source"]["file"], p["licence"]["licence"], p["licence"]["verified"]),
                         ("bgm/朝のピアノ.wav", "CC0-1.0", False))
        self.assertEqual(p["source"]["sha256"], library.sha256_file(os.path.join(self.lib, "bgm", "朝のピアノ.wav")))
        self.assertEqual(p["output"]["sha256"], library.sha256_file(doc["output"]["path"]))
        self.assertEqual(p["selection"]["query"]["duration"], 10.0)
        self.assertNotIn(self.dir, json.dumps(p, ensure_ascii=False))                 # no local paths in the provenance
        self.assertEqual(doc["handoff"]["to"], "audio-production-skill MIX")
        self.assert_clean()

    def test_trim_and_the_ranking_prefers_the_matching_file(self):
        doc = self.pick(duration=12.0)
        self.assertEqual(doc["selection"]["chosen"], "bgm/長い曲.wav")              # trim beats loop
        self.assertEqual(doc["fit"]["mode"], "trim")
        self.assertAlmostEqual(doc["output"]["duration"], 12.0, delta=0.01)
        self.assertTrue(any("attribution" not in w and "no fade" in w for w in doc["warnings"]))
        self.assertEqual(doc["licence"]["credit"], "Music: A")
        doc2 = self.pick("b.wav", duration=12.0, tags=["piano"])
        self.assertEqual(doc2["selection"]["chosen"], "bgm/朝のピアノ.wav")         # a matched tag outranks the fit

    def test_same_library_and_query_give_the_same_file(self):
        a = self.pick("a.wav", tags=["piano", "corporate"], duration=9.0)
        b = self.pick("b.wav", tags=["piano", "corporate"], duration=9.0)
        self.assertEqual((a["selection"], a["fit"]), (b["selection"], b["fit"]))
        self.assertEqual(a["output"]["sha256"], b["output"]["sha256"])

    def test_whole_file_is_copied_unchanged(self):
        doc = self.pick("door.wav", kind="sfx", tags=["door"])
        self.assertEqual((doc["fit"]["mode"], doc["selection"]["chosen"]), ("none", "sfx/ドア.wav"))
        self.assertEqual(doc["output"]["sha256"], library.sha256_file(os.path.join(self.lib, "sfx", "ドア.wav")))
        self.assertFalse([c for c in self.skill.calls if c[0] in ("cut", "join")])

    def test_a_file_without_a_declared_licence_is_never_used(self):
        with self.assertRaises(SoundDesignError) as e:
            self.pick(file="bgm/不明.wav")
        self.assertEqual(e.exception.code, "LICENCE_REQUIRED")
        self.tone("bgm/manifestにない.wav", 3.0)
        with self.assertRaises(SoundDesignError) as e:
            self.pick(file="bgm/manifestにない.wav")
        self.assertEqual(e.exception.code, "LICENCE_REQUIRED")
        self.assertEqual(os.listdir(self.out_dir), [])
        self.assertEqual(self.skill.calls, [])

    def test_explicit_file_and_refusals(self):
        doc = self.pick(file="bgm/長い曲.wav", duration=5.0, tags=["corporate"])
        self.assertEqual((doc["selection"]["chosen"], doc["selection"]["matched"]), ("bgm/長い曲.wav", ["corporate"]))
        with self.assertRaises(SoundDesignError) as e:
            self.pick("x.wav", file="sfx/ドア.wav")                                 # kind bgm requested
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(SoundDesignError) as e:
            self.pick("x.wav", file="bgm/長い曲.wav", duration=40.0)               # shorter and not loopable
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(SoundDesignError) as e:
            self.pick("x.wav", file="../escape.wav")
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(SoundDesignError) as e:
            self.pick("x.wav", tags=["jazz"])
        self.assertEqual(e.exception.code, "INPUT_NOT_FOUND")
        self.assertIn("bgm/不明.wav", [x["file"] for x in e.exception.details["excluded"]])
        with self.assertRaises(SoundDesignError) as e:
            self.pick("x.wav", kind=None)
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        for bad in ({"duration": 0}, {"duration": -1.0}, {"crossfade": -0.1}, {"kind": "music"}, {"timeout": 0}):
            with self.assertRaises(SoundDesignError) as e:
                self.pick("x.wav", **bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", bad)
        self.assertEqual(os.listdir(self.out_dir), ["bed.wav"])
        self.assert_clean()

    def test_output_checks(self):
        self.pick(duration=12.0)
        with self.assertRaises(SoundDesignError) as e:
            self.pick(duration=12.0)
        self.assertEqual(e.exception.code, "OUTPUT_EXISTS")
        self.assertEqual(self.pick(duration=12.0, overwrite=True)["fit"]["mode"], "trim")
        with self.assertRaises(SoundDesignError) as e:
            library_pick(LibraryPickRequest(self.lib, os.path.join(self.lib, "bgm", "長い曲.wav"), kind="bgm", overwrite=True),
                         skill=self.skill)
        self.assertEqual(e.exception.code, "OUTPUT_IS_INPUT")

    def test_cli(self):
        with mock.patch.object(FfmpegSkill, "locate", return_value=self.skill):
            out = os.path.join(self.out_dir, "cli.wav")
            rc, doc = run_cli("library-pick", self.lib, "-o", out, "--kind", "bgm", "--tags", "piano", "--duration", "6", "--json")
            self.assertEqual(rc, 0, doc)
            self.assertEqual(doc["selection"]["chosen"], "bgm/朝のピアノ.wav")
            rc, idx = run_cli("library-index", self.lib, "--json")
            self.assertEqual((rc, idx["summary"]["unknown_licence"]), (0, 1))
            src = write_tone_wav(os.path.join(self.dir, "新規.wav"), seconds=1.0, rate=8000)
            rc, err = run_cli("library-import", src, "--library", self.lib, "--kind", "sfx", "--json")
            self.assertEqual((rc, err["error"]["code"]), (2, "LICENCE_REQUIRED"))
            rc, ok = run_cli("library-import", src, "--library", self.lib, "--kind", "sfx", "--license", "CC0-1.0",
                             "--tags", "click", "--json")
            self.assertEqual((rc, ok["licence"]["licence"]), (0, "CC0-1.0"))


class EngineFreeTests(unittest.TestCase):
    def test_library_mode_never_references_an_engine_or_the_network(self):
        text = (ROOT / "src" / "sound_design_skill" / "library.py").read_text(encoding="utf-8")
        for banned in ("engines", "run_engine", "select_engine", "require_confirmation", "service", "worker", "acestep",
                       "import ezaudio", "from .engines", "urllib", "socket", "http"):
            self.assertNotIn(banned, text.replace("never ACE-Step or EzAudio", ""), banned)

    def test_contract_and_doctor_publish_library_mode(self):
        c = contract()
        self.assertEqual({p["id"]: p["lifecycle"] for p in c["provides"]}["audio.library_import"], "EXPERIMENTAL")
        tools = {t["tool_id"]: t for t in c["tools"]}
        for name in ("library-import", "library-index", "library-pick"):
            t = tools[f"sound-design/{name}"]
            self.assertEqual((t["implemented"], t["engine"], t["engines"], t["roadmap_step"]), (True, None, [], 4))
            self.assertEqual(t["schema"], f"sound-design/{name}@1")
        self.assertIn("--licence <SPDX|text>", tools["sound-design/library-import"]["argv"])
        self.assertIn("LICENCE_REQUIRED", c["policy"]["library_licence"])
        self.assertIn("never ACE-Step or EzAudio", c["policy"]["library_engine"])
        self.assertEqual(c["errors"]["LICENCE_REQUIRED"], {"exit_code": 2, "retryable": False})
        self.assertEqual(c["handoff"]["library"].split()[0], "audio-production-skill")

    def test_licence_refusal_on_a_legacy_code_page_console(self):
        """A Japanese file name through the CLI on cp1252 (Windows): ASCII JSON, no crash in human mode (ADR-003)."""
        with tempfile.TemporaryDirectory() as tmp:
            src = write_tone_wav(os.path.join(tmp, "朝の音.wav"), seconds=0.2, rate=8000)
            child = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0",
                         PYTHONPATH=os.pathsep.join([str(ROOT / "src")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
            argv = [sys.executable, "-m", "sound_design_skill.cli", "library-import", src, "--library", tmp, "--kind", "bgm"]
            proc = subprocess.run(argv + ["--json"], capture_output=True, env=child, timeout=60)
            self.assertEqual(proc.returncode, 2, proc.stderr.decode("utf-8", "replace"))
            self.assertTrue(all(b < 128 for b in proc.stdout))
            doc = json.loads(proc.stdout.decode("ascii"))
            self.assertEqual(doc["error"]["code"], "LICENCE_REQUIRED")
            self.assertIn("朝の音.wav", doc["error"]["details"]["file"])
            human = subprocess.run(argv, capture_output=True, env=child, timeout=60)
            self.assertEqual(human.returncode, 2)
            self.assertIn(b"LICENCE_REQUIRED", human.stderr)


if __name__ == "__main__":
    unittest.main()
