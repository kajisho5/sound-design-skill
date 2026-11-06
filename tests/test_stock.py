"""Unit tests for stock music and SFX with licence records (roadmap step 10, ADR-014): the attribution line (declared,
the credit, or composed in TASL order for a CC BY licence from declared parts only), the Content ID statement and its
warning, licence texts and URLs recorded at import, the receipt of library-pick, the refusal of a file without a
licence (unchanged), and the stock sources doctor reports (MISSING without an API key; nothing is ever fetched). No
FFmpeg, no engine and no network are needed."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import test_library as tl
from helpers import env, run_cli, write_tone_wav

from sound_design_skill import stock
from sound_design_skill.contract import contract
from sound_design_skill.doctor import doctor
from sound_design_skill.errors import SoundDesignError
from sound_design_skill.library import (MANIFEST_NAME, LibraryImportRequest, LibraryIndexRequest, LibraryPickRequest,
                                        library_import, library_index, library_pick, licence_info, licence_warnings)
from sound_design_skill.provenance import sha256_file

CC_BY_LINE = '"Morning Light" by Aiko Tanaka (https://example.invalid/track/42), licensed under CC-BY-4.0 ' \
             '(https://creativecommons.org/licenses/by/4.0/)'


class AttributionTests(unittest.TestCase):
    def test_composed_in_tasl_order_from_declared_parts_only(self):
        info = licence_info({"licence": "CC-BY-4.0", "title": "Morning Light", "author": "Aiko Tanaka",
                             "source": "https://example.invalid/track/42",
                             "licence_url": "https://creativecommons.org/licenses/by/4.0/"})
        self.assertEqual((info["attribution"], info["attribution_source"]), (CC_BY_LINE, "composed"))
        self.assertTrue(any("composed" in w for w in licence_warnings("a.wav", info)))
        only_author = licence_info({"licence": "CC BY 3.0", "author": "山田"})
        self.assertEqual(only_author["attribution"], "山田, licensed under CC BY 3.0")

    def test_declared_line_and_credit_win_and_nothing_is_invented(self):
        declared = licence_info({"licence": "CC-BY-4.0", "title": "T", "author": "A", "credit": "Music: A",
                                 "attribution": "Music: T by A (CC BY 4.0)"})
        self.assertEqual((declared["attribution"], declared["attribution_source"]), ("Music: T by A (CC BY 4.0)", "declared"))
        credit = licence_info({"licence": "CC-BY-4.0", "title": "T", "credit": "Music: A"})
        self.assertEqual((credit["attribution"], credit["attribution_source"]), ("Music: A", "credit"))
        for entry in ({"licence": "CC-BY-4.0"}, {"licence": "CC0-1.0", "title": "T", "author": "A"},
                      {"licence": "unknown", "title": "T", "author": "A"}, None):
            info = licence_info(entry)
            self.assertEqual((info["attribution"], info["attribution_source"]), (None, None), entry)
        self.assertTrue(any("no credit" in w for w in licence_warnings("a.wav", licence_info({"licence": "CC-BY-4.0"}))))

    def test_content_id_statement_and_warning(self):
        reg = licence_info({"licence": "Royalty-free (provider licence)", "content_id": "registered", "provider": "ShopX"})
        self.assertEqual(reg["content_id"], "registered")
        warn = licence_warnings("bgm.wav", reg)
        self.assertTrue(any("Content ID" in w and "ShopX" in w for w in warn), warn)
        for value, want in (("not-registered", "not-registered"), (None, "unknown"), ("maybe", "unknown")):
            info = licence_info({"licence": "CC0-1.0", **({"content_id": value} if value else {})})
            self.assertEqual(info["content_id"], want)
            self.assertFalse(any("Content ID" in w for w in licence_warnings("x.wav", info)))
        self.assertNotIn("cleared", json.dumps(reg))


class StockImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.lib = os.path.join(self.dir, "lib")
        os.mkdir(self.lib)
        self.dl = os.path.join(self.dir, "downloads")
        os.mkdir(self.dl)
        self.skill = tl.StubLibrarySkill()
        self.track = write_tone_wav(os.path.join(self.dl, "morning-light.wav"), seconds=6.0, rate=8000)
        self.text = os.path.join(self.dl, "LICENSE.txt")
        with open(self.text, "w", encoding="utf-8", newline="\r\n") as fh:       # CRLF on every OS, as on Windows
            fh.write("Creative Commons Attribution 4.0 International\n...\n")

    def imp(self, src, **kw):
        base = dict(input_path=src, library_dir=self.lib, licence="CC-BY-4.0", kind="bgm", tags=["morning"])
        base.update(kw)
        return library_import(LibraryImportRequest(**base), skill=self.skill)

    def test_a_cc_by_track_yields_its_attribution_line_in_the_receipt(self):
        doc = self.imp(self.track, title="Morning Light", author="Aiko Tanaka", source="https://example.invalid/track/42",
                       provider="Example Music", licence_url="https://creativecommons.org/licenses/by/4.0/",
                       licence_file=self.text, content_id="not-registered")
        entry = doc["entry"]
        sha = sha256_file(self.text)                                      # the file as downloaded, byte for byte
        self.assertEqual((entry["licence_text_sha256"], entry["licence_file"]), (sha, f"licences/{sha}.txt"))
        self.assertEqual(sha256_file(os.path.join(self.lib, "licences", f"{sha}.txt")), sha)
        self.assertEqual((entry["title"], entry["provider"], entry["content_id"]), ("Morning Light", "Example Music",
                                                                                    "not-registered"))
        self.assertEqual(doc["licence"]["attribution"], CC_BY_LINE)
        pick = library_pick(LibraryPickRequest(self.lib, os.path.join(self.dir, "bed.wav"), kind="bgm", duration=4.0),
                            skill=self.skill)
        self.assertEqual(pick["licence"]["attribution"], CC_BY_LINE)                  # the receipt
        self.assertEqual(pick["provenance"]["licence"]["attribution"], CC_BY_LINE)
        self.assertEqual((pick["handoff"]["credit_line"], pick["handoff"]["content_id"]), (CC_BY_LINE, "not-registered"))
        self.assertFalse(any("Content ID" in w for w in pick["warnings"]))
        idx = library_index(LibraryIndexRequest(self.lib), skill=self.skill)
        self.assertEqual([f["file"] for f in idx["files"]], ["morning-light.wav"])     # licences/*.txt are not library files

    def test_a_content_id_registered_track_is_warned_about(self):
        other = write_tone_wav(os.path.join(self.dl, "hit-song.wav"), seconds=6.0, rate=8000)
        doc = self.imp(other, licence="Royalty-free, provider standard licence", provider="ShopX", content_id="registered")
        self.assertTrue(any("Content ID" in w for w in doc["warnings"]))
        pick = library_pick(LibraryPickRequest(self.lib, os.path.join(self.dir, "bed.wav"), kind="bgm"), skill=self.skill)
        self.assertEqual(pick["licence"]["content_id"], "registered")
        self.assertTrue(any("Content ID" in w and "ShopX" in w for w in pick["warnings"]), pick["warnings"])
        idx = library_index(LibraryIndexRequest(self.lib), skill=self.skill)
        self.assertTrue(any("Content ID" in w for w in idx["warnings"]))

    def test_refusals_and_a_missing_licence_as_before(self):
        with self.assertRaises(SoundDesignError) as e:                      # a URL is not a licence
            self.imp(self.track, licence=None, licence_url="https://example.invalid/licence")
        self.assertEqual(e.exception.code, "LICENCE_REQUIRED")
        binary = os.path.join(self.dl, "licence.bin")
        with open(binary, "wb") as fh:
            fh.write(b"\xff\xfe\x00\x80")
        for kw, code in ((dict(licence_url="example.invalid/licence"), "INVALID_ARGUMENT"),
                         (dict(content_id="yes"), "INVALID_ARGUMENT"),
                         (dict(licence_file=os.path.join(self.dl, "nope.txt")), "INPUT_NOT_FOUND"),
                         (dict(licence_file=binary), "INVALID_ARGUMENT")):
            with self.assertRaises(SoundDesignError) as e:
                self.imp(self.track, **kw)
            self.assertEqual(e.exception.code, code, kw)
        self.assertEqual(self.skill.calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.lib, MANIFEST_NAME)))

    def test_cli_flags(self):
        from unittest import mock
        with mock.patch("sound_design_skill.ops.FfmpegSkill.locate", return_value=self.skill):
            rc, doc = run_cli("library-import", self.track, "--library", self.lib, "--license", "CC-BY-4.0", "--kind", "bgm",
                              "--title", "Morning Light", "--author", "Aiko Tanaka", "--license-url",
                              "https://creativecommons.org/licenses/by/4.0/", "--licence-file", self.text,
                              "--content-id", "registered", "--provider", "Example", "--json")
        self.assertEqual(rc, 0, doc)
        self.assertEqual(doc["licence"]["attribution"], '"Morning Light" by Aiko Tanaka, licensed under CC-BY-4.0 '
                                                        '(https://creativecommons.org/licenses/by/4.0/)')
        self.assertTrue(any("Content ID" in w for w in doc["warnings"]))


class StockSourceTests(unittest.TestCase):
    def test_doctor_reports_missing_keys_and_never_fetches(self):
        keys = {s["api_key_env"]: None for s in stock.SOURCES if s["api_key_env"]}
        with tempfile.TemporaryDirectory() as tmp, env(**keys):
            d = doctor(os.path.join(tmp, "none"))
        st = d["checks"]["stock_sources"]
        self.assertEqual({k: v["status"] for k, v in st.items()},
                         {"freesound": "MISSING", "jamendo": "MISSING", "youtube-audio-library": "MANUAL"})
        self.assertTrue(all("not implemented" in v["fetch"] for v in st.values()))
        self.assertFalse(any("stock" in p or "freesound" in p.casefold() for p in d["problems"] + d["warnings"]))
        with env(FREESOUND_API_KEY="dummy"):
            self.assertEqual(stock.report()["freesound"]["status"], "KEY_SET")

    def test_contract(self):
        c = contract()
        self.assertEqual(c["stock"]["sources"], ["freesound", "jamendo", "youtube-audio-library"])
        self.assertIn("ADR-014", c["policy"]["stock_licences"])
        argv = next(t for t in c["tools"] if t["tool_id"] == "sound-design/library-import")["argv"]
        for flag in ("--licence-url", "--licence-file", "--attribution", "--content-id", "--title", "--provider"):
            self.assertIn(flag, " ".join(argv))
        self.assertIn("licence.attribution", c["provenance"]["library_fields"])


if __name__ == "__main__":
    unittest.main()
