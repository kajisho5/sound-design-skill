"""Integration with a real ffmpeg-skill checkout and a real FFmpeg: locate it, check every tool and flag this skill
uses against its live contract, run its doctor, probe a WAV through it, and run GENERATE_BGM and GENERATE_SFX end to
end through the CLI (the real worker with the fake ACE-Step engine / fake EzAudio checkout of tests/helpers.py, the
result validated by the real ffmpeg-skill/probe), and library mode end to end (ADR-008: WAVs written by the test under
Japanese names, imported, indexed, chosen, trimmed and looped through the real ffmpeg-skill cut / join / probe /
loudness). Skipped without a checkout (SOUND_DESIGN_FFMPEG_SKILL_DIR or ../ffmpeg-skill); CI's integration
job always has one."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from helpers import (FFMPEG_SKILL_DIR, env, fake_acestep_engine, fake_checkpoints, fake_ezaudio_engine, no_engines_env, run_cli,
                     write_tone_wav)

from sound_design_skill.doctor import doctor
from sound_design_skill.errors import SoundDesignError
from sound_design_skill.ffmpeg_skill import TOOLS_USED, FfmpegSkill

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class FfmpegSkillIntegrationTests(unittest.TestCase):
    def test_live_contract_declares_every_tool_and_flag(self):
        info = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR)).info()
        self.assertTrue(info.supported, info.problems)
        self.assertEqual(sorted(t for t in info.tools if t in TOOLS_USED), sorted(TOOLS_USED))
        self.assertNotEqual(info.version, "unknown")

    def test_probe_through_ffmpeg_skill(self):
        skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))
        with tempfile.TemporaryDirectory() as tmp:
            wav = write_tone_wav(os.path.join(tmp, "tone.wav"), seconds=1.0, rate=48000)
            facts = skill.probe(wav)
            self.assertAlmostEqual(float(facts["duration"]), 1.0, delta=0.05)
            not_media = os.path.join(tmp, "not-media.wav")
            with open(not_media, "w", encoding="utf-8") as fh:
                fh.write("not audio")
            with self.assertRaises(SoundDesignError) as e:
                skill.probe(not_media)
            self.assertEqual(e.exception.code, "UNSUPPORTED_MEDIA")

    def test_doctor_with_real_ffmpeg_skill(self):
        with tempfile.TemporaryDirectory() as tmp, env(**no_engines_env(tmp)):
            d = doctor(str(FFMPEG_SKILL_DIR))
            rc, cli_doc = run_cli("doctor", "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
        self.assertEqual(d["checks"]["ffmpeg_skill"]["status"], "OK", d["problems"])
        self.assertEqual((d["checks"]["ffmpeg"]["status"], d["checks"]["ffprobe"]["status"]), ("OK", "OK"))
        self.assertEqual(sorted(d["checks"]["engines"]), ["acestep", "ezaudio"])
        for eid in ("acestep", "ezaudio"):
            self.assertIn(d["checks"]["engines"][eid]["status"], ("MISSING", "INCOMPATIBLE"))
        self.assertIn(d["checks"]["device"]["status"], ("OK", "CPU_ONLY", "UNKNOWN"))
        self.assertEqual(d["status"], "degraded")
        self.assertEqual((rc, cli_doc["status"]), (0, d["status"]))


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class GenerateBgmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
        for cm in (env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        engine_cm = fake_acestep_engine(self.dir)
        self.fake = engine_cm.__enter__()
        self.addCleanup(engine_cm.__exit__, None, None, None)

    def cli(self, name, *extra):
        out = os.path.join(self.dir, name)
        return out, run_cli("generate-bgm", "--prompt", "calm piano bed", "--tags", "ambient,70 bpm", "--duration", "10.5",
                            "-o", out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json", *extra)

    def test_bed_validated_by_the_real_probe_and_repeatable_by_seed(self):
        out, (rc, doc) = self.cli("a.wav", "--seed", "2026")
        self.assertEqual(rc, 0, doc)
        facts = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR)).probe(out)
        self.assertAlmostEqual(float(facts["duration"]), 10.5, delta=0.1)
        self.assertEqual((facts["audio"]["sample_rate"], facts["audio"]["channels"]), (48000, 2))
        self.assertEqual(doc["provenance"]["ffmpeg_skill"]["version"], FfmpegSkill.locate(str(FFMPEG_SKILL_DIR)).info().version)
        self.assertEqual((doc["provenance"]["seed"], doc["provenance"]["engine"]["id"]), (2026, "acestep"))
        self.assertEqual(doc["provenance"]["prompt"]["text"], "calm piano bed, ambient, 70 bpm")
        _, (rc2, doc2) = self.cli("b.wav", "--seed", "2026")
        _, (rc3, doc3) = self.cli("c.wav", "--seed", "2027")
        self.assertEqual((rc2, rc3), (0, 0))
        self.assertEqual(doc2["output"]["sha256"], doc["output"]["sha256"])
        self.assertNotEqual(doc3["output"]["sha256"], doc["output"]["sha256"])
        self.assertEqual(sorted(os.listdir(self.dir)), sorted(["a.wav", "b.wav", "c.wav", "ckpt", "site",
                                                               "fake_acestep_calls.jsonl"]))

    def test_a_bed_of_the_wrong_length_is_refused_by_the_real_probe(self):
        self.fake["set_mode"]("short")
        out, (rc, doc) = self.cli("short.wav", "--seed", "1")
        self.assertEqual((rc, doc["error"]["code"]), (4, "INVALID_RESULT"))
        self.assertIn("duration", doc["error"]["message"])
        self.assertFalse(os.path.exists(out))


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class GenerateSfxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.out_dir = os.path.join(self.dir, "out")
        os.mkdir(self.out_dir)
        cm = env(**no_engines_env(self.dir))
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        engine_cm = fake_ezaudio_engine(self.dir)
        self.fake = engine_cm.__enter__()
        self.addCleanup(engine_cm.__exit__, None, None, None)

    def cli(self, name, *extra, acknowledge=True):
        out = os.path.join(self.out_dir, name)
        ack = ("--acknowledge-ezaudio-training-data",) if acknowledge else ()   # ADR-007
        return out, run_cli("generate-sfx", "--prompt", "a wooden door closes", "--tags", "indoor", "--duration", "3",
                            "-o", out, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json", *ack, *extra)

    def test_without_the_flag_nothing_runs(self):
        out, (rc, doc) = self.cli("refused.wav", "--seed", "1", acknowledge=False)
        self.assertEqual((rc, doc["error"]["code"]), (8, "CONFIRMATION_REQUIRED"))
        self.assertIn("--acknowledge-ezaudio-training-data", doc["error"]["message"])
        self.assertEqual(self.fake["calls"](), [])
        self.assertFalse(os.path.exists(out))
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_effect_validated_by_the_real_probe_and_repeatable_by_seed(self):
        out, (rc, doc) = self.cli("a.wav", "--seed", "2026")
        self.assertEqual(rc, 0, doc)
        skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))
        facts = skill.probe(out)
        self.assertAlmostEqual(float(facts["duration"]), 3.0, delta=0.05)
        self.assertEqual((facts["audio"]["sample_rate"], facts["audio"]["channels"]), (24000, 1))
        self.assertEqual(doc["provenance"]["ffmpeg_skill"]["version"], skill.info().version)
        self.assertEqual((doc["provenance"]["seed"], doc["provenance"]["engine"]["id"]), (2026, "ezaudio"))
        self.assertEqual(doc["provenance"]["prompt"]["text"], "a wooden door closes, indoor")
        self.assertTrue(doc["warnings"])
        self.assertTrue(doc["provenance"]["confirmation"]["acknowledged"])
        self.assertTrue(doc["provenance"]["confirmation"]["acknowledged_at"].endswith("Z"))
        _, (rc2, doc2) = self.cli("b.wav", "--seed", "2026")
        _, (rc3, doc3) = self.cli("c.wav", "--seed", "2027")
        self.assertEqual((rc2, rc3), (0, 0))
        self.assertEqual(doc2["output"]["sha256"], doc["output"]["sha256"])
        self.assertNotEqual(doc3["output"]["sha256"], doc["output"]["sha256"])
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["a.wav", "b.wav", "c.wav"])

    def test_an_effect_of_the_wrong_length_is_refused_by_the_real_probe(self):
        self.fake["set_mode"]("short")
        out, (rc, doc) = self.cli("short.wav", "--seed", "1")
        self.assertEqual((rc, doc["error"]["code"]), (4, "INVALID_RESULT"))
        self.assertIn("duration", doc["error"]["message"])
        self.assertFalse(os.path.exists(out))
        self.assertEqual(os.listdir(self.out_dir), [])


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class LibraryIntegrationTests(unittest.TestCase):
    """Library mode through the CLI and the real ffmpeg-skill (ADR-008). No engine is involved: the engine
    environment points nowhere, and the results say engine null."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        cm = env(**no_engines_env(self.dir))
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        self.lib = os.path.join(self.dir, "音素材ライブラリ")
        os.mkdir(self.lib)
        self.out = os.path.join(self.dir, "出力")
        os.mkdir(self.out)
        self.inbox = os.path.join(self.dir, "受け取り")
        os.mkdir(self.inbox)
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))

    def cli(self, *argv):
        return run_cli(*argv, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")

    def test_import_index_pick_trim_loop_and_licence_refusal(self):
        piano = write_tone_wav(os.path.join(self.inbox, "朝のピアノ.wav"), seconds=4.0, rate=48000, freq=262.0)
        long_bed = write_tone_wav(os.path.join(self.inbox, "長いBGM.wav"), seconds=20.0, rate=48000, freq=330.0)
        door = write_tone_wav(os.path.join(self.inbox, "ドア.wav"), seconds=0.8, rate=24000, freq=880.0)
        flac = os.path.join(self.inbox, "鐘.flac")
        self.skill.cut(write_tone_wav(os.path.join(self.dir, "bell.wav"), seconds=2.0, rate=44100, freq=660.0), flac, duration=2.0)

        rc, doc = self.cli("library-import", piano, "--library", self.lib, "--kind", "bgm")
        self.assertEqual((rc, doc["error"]["code"]), (2, "LICENCE_REQUIRED"))
        rc, doc = self.cli("library-import", piano, "--library", self.lib, "--kind", "bgm", "--licence", "unknown")
        self.assertEqual((rc, doc["error"]["code"]), (2, "LICENCE_REQUIRED"))
        self.assertEqual(os.listdir(self.lib), [])

        rc, doc = self.cli("library-import", piano, "--library", self.lib, "--kind", "bgm", "--licence", "CC0-1.0",
                           "--tags", "piano", "--mood", "calm", "--loopable", "--name", "bgm/朝のピアノ.wav")
        self.assertEqual(rc, 0, doc)
        self.assertAlmostEqual(doc["file"]["measured"]["duration"], 4.0, delta=0.02)
        self.assertIsNotNone(doc["file"]["loudness"]["integrated_lufs"])
        self.assertEqual((doc["licence"]["status"], doc["licence"]["verified"], doc["engine"]), ("declared", False, None))
        for path, kind, lic, extra in ((long_bed, "bgm", "CC-BY-4.0", ["--tags", "corporate", "--credit", "Music: 作曲者"]),
                                       (door, "sfx", "Purchased: SFX pack standard licence", ["--tags", "door"]),
                                       (flac, "sfx", "CC0-1.0", ["--tags", "bell"])):
            rc, doc = self.cli("library-import", path, "--library", self.lib, "--kind", kind, "--licence", lic, *extra)
            self.assertEqual(rc, 0, doc)
        shutil.copyfile(piano, os.path.join(self.lib, "出所不明.wav"))           # dropped in by hand: no licence

        rc, idx = self.cli("library-index", self.lib, "--loudness")
        self.assertEqual(rc, 0, idx)
        self.assertEqual(idx["summary"], {"files": 5, "declared": 4, "unknown_licence": 1, "unreadable": 0, "missing_files": 0})
        by = {f["file"]: f for f in idx["files"]}
        self.assertEqual((by["出所不明.wav"]["licence"]["licence"], by["出所不明.wav"]["licence"]["status"]), ("unknown", "unknown"))
        self.assertAlmostEqual(by["鐘.flac"]["measured"]["duration"], 2.0, delta=0.05)
        self.assertTrue(all(f["loudness"]["integrated_lufs"] is not None for f in idx["files"]))

        rc, trim = self.cli("library-pick", self.lib, "-o", os.path.join(self.out, "trim.wav"), "--kind", "bgm",
                            "--duration", "7.5")
        self.assertEqual(rc, 0, trim)
        self.assertEqual((trim["selection"]["chosen"], trim["fit"]["mode"]), ("長いBGM.wav", "trim"))
        facts = self.skill.probe(trim["output"]["path"])
        self.assertAlmostEqual(float(facts["duration"]), 7.5, delta=0.02)
        self.assertEqual(trim["licence"]["credit"], "Music: 作曲者")

        rc, loop = self.cli("library-pick", self.lib, "-o", os.path.join(self.out, "loop.wav"), "--kind", "bgm",
                            "--tags", "piano", "--duration", "10")
        self.assertEqual(rc, 0, loop)
        self.assertEqual((loop["selection"]["chosen"], loop["fit"]["mode"], loop["fit"]["repeats"]), ("bgm/朝のピアノ.wav", "loop", 3))
        facts = self.skill.probe(loop["output"]["path"])
        self.assertAlmostEqual(float(facts["duration"]), 10.0, delta=0.1)
        self.assertEqual((facts["audio"]["sample_rate"], facts["audio"]["channels"]), (48000, 1))
        self.assertIsNotNone(loop["output"]["loudness"]["integrated_lufs"])
        p = loop["provenance"]
        self.assertEqual((p["engine"], p["ai_generated"], p["source"]["file"]), (None, False, "bgm/朝のピアノ.wav"))
        self.assertEqual(p["ffmpeg_skill"]["version"], self.skill.info().version)
        self.assertNotIn(self.dir, json.dumps(p, ensure_ascii=False))

        rc, again = self.cli("library-pick", self.lib, "-o", os.path.join(self.out, "loop2.wav"), "--kind", "bgm",
                             "--tags", "piano", "--duration", "10")
        self.assertEqual(rc, 0, again)
        self.assertEqual((again["selection"], again["output"]["sha256"]), (loop["selection"], loop["output"]["sha256"]))

        rc, bell = self.cli("library-pick", self.lib, "-o", os.path.join(self.out, "bell.wav"), "--kind", "sfx", "--tags", "bell")
        self.assertEqual(rc, 0, bell)
        self.assertEqual((bell["fit"]["mode"], bell["output"]["codec"]), ("none", "pcm_s16le"))      # FLAC converted to WAV
        rc, door_doc = self.cli("library-pick", self.lib, "-o", os.path.join(self.out, "door.wav"), "--kind", "sfx",
                                "--tags", "door")
        self.assertEqual(rc, 0, door_doc)
        self.assertEqual(door_doc["output"]["sha256"], door_doc["source"]["sha256"])                 # PCM WAV used as is

        rc, refused = self.cli("library-pick", self.lib, "-o", os.path.join(self.out, "x.wav"), "--file", "出所不明.wav")
        self.assertEqual((rc, refused["error"]["code"]), (2, "LICENCE_REQUIRED"))
        self.assertEqual(sorted(os.listdir(self.out)), ["bell.wav", "door.wav", "loop.wav", "loop2.wav", "trim.wav"])
        self.assertEqual([n for n in os.listdir(self.lib) if n.startswith(".")], [])


if __name__ == "__main__":
    unittest.main()
