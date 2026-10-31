"""Integration with a real ffmpeg-skill checkout and a real FFmpeg: locate it, check every tool and flag this skill
uses against its live contract, run its doctor, probe a WAV through it, and run GENERATE_BGM and GENERATE_SFX end to
end through the CLI (the real worker with the fake ACE-Step engine / fake EzAudio checkout of tests/helpers.py, the
result validated by the real ffmpeg-skill/probe). Skipped without a checkout (SOUND_DESIGN_FFMPEG_SKILL_DIR or ../ffmpeg-skill); CI's integration
job always has one."""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
