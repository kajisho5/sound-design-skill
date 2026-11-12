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
import subprocess
import sys
import tempfile
import unittest

from helpers import (FFMPEG_SKILL_DIR, QC_SKILL_DIR, XL_CHECKPOINT_FILES, env, fake_acestep_engine, fake_checkpoints,
                     fake_ezaudio_engine, gpu, no_engines_env, run_cli, run_qc_loop_seam, write_bed_wav, write_burst_wav,
                     write_drifting_bed_wav, write_tone_wav)

from sound_design_skill.doctor import doctor
from sound_design_skill.errors import SoundDesignError
from sound_design_skill.ffmpeg_skill import TOOLS_USED, FfmpegSkill
from sound_design_skill import pcm

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
        self.ckpt = ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
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

    def test_a_tempo_bed_ends_on_a_bar_line_with_its_grid_beside_it(self):
        out = os.path.join(self.dir, "pv.wav")
        rc, doc = run_cli("generate-bgm", "--prompt", "upbeat pop", "--duration", "15", "--bpm", "120", "-o", out,
                          "--seed", "9", "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")      # "120 BPM, 15 s"
        self.assertEqual(rc, 0, doc)
        self.assertAlmostEqual(float(FfmpegSkill.locate(str(FFMPEG_SKILL_DIR)).probe(out)["duration"]), 16.0, delta=0.01)
        with open(os.path.join(self.dir, "pv.beats.json"), encoding="utf-8") as fh:
            grid = json.load(fh)
        self.assertEqual((grid["bpm"], grid["bars"], len(grid["beats"]), grid["bar_lines"][-1]), (120, 8, 32, 16.0))
        self.assertEqual((grid["measured"], grid["audio"]["sha256"]), (False, doc["output"]["sha256"]))

    def test_scenes_beats_on_a_tempo_bed_agrees_with_its_written_grid_within_20_ms(self):
        """Roadmap step 8's Done-when, on the fake engine's output (ADR-018): ffmpeg-skill `scenes --beats` (audio-only
        input from 3.2.2) finds every written beat within 20 ms; older releases refuse a WAV, which is asserted."""
        skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))
        audio_only = tuple(int(p) for p in skill.info().version.split(".")[:3]) >= (3, 2, 2)
        for bpm, duration, meter in (("120", "15", "4"), ("90", "20", "3"), ("137", "12", "4")):
            with self.subTest(bpm=bpm):
                out = os.path.join(self.dir, f"pv{bpm}.wav")
                rc, doc = run_cli("generate-bgm", "--prompt", "upbeat pop", "--duration", duration, "--bpm", bpm,
                                  "--time-signature", meter, "-o", out, "--seed", "9",
                                  "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
                self.assertEqual(rc, 0, doc)
                with open(os.path.join(self.dir, f"pv{bpm}.beats.json"), encoding="utf-8") as fh:
                    written = [b["time"] for b in json.load(fh)["beats"]]
                proc = subprocess.run([sys.executable, os.path.join(str(FFMPEG_SKILL_DIR), "scripts", "scenes.py"), out,
                                       "--beats", "--json"], capture_output=True, text=True, encoding="utf-8")
                if not audio_only:
                    self.assertNotEqual(proc.returncode, 0, proc.stdout)                 # "input has no video stream"
                    continue
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                scenes = json.loads(proc.stdout)
                measured = scenes["beats"]
                self.assertTrue(scenes["audio_only"] and scenes["beat_grid"]["usable"], scenes["beat_grid"])
                self.assertAlmostEqual(scenes["beat_grid"]["tempo_bpm"], float(bpm), delta=0.5)
                self.assertEqual(len(measured), len(written), (measured, written))
                worst = max(min(abs(t - m) for m in measured) for t in written)
                self.assertLessEqual(worst, 0.020, (measured, written))

    def test_the_xl_model_through_the_same_api_and_its_refusals(self):
        """ADR-016: --model acestep-v15-xl-turbo reaches ACE-Step as config_path; too small a GPU and missing files are
        refused before the worker starts."""
        model = ["--model", "acestep-v15-xl-turbo"]
        with gpu(mb=24576):
            out, (rc, doc) = self.cli("xl-missing.wav", "--seed", "3", *model)
        self.assertEqual((rc, doc["error"]["code"]), (3, "MODEL_NOT_FOUND"))
        self.assertIn("acestep-download --model acestep-v15-xl-turbo", doc["error"]["details"]["install"])
        fake_checkpoints(self.ckpt, XL_CHECKPOINT_FILES)
        with gpu(mb=12288):
            out, (rc, doc) = self.cli("xl-small.wav", "--seed", "3", *model)
        self.assertEqual((rc, doc["error"]["code"], doc["error"]["details"]["default_model"]),
                         (3, "INSUFFICIENT_VRAM", "acestep-v15-turbo"))
        self.assertEqual(self.fake["calls"](), [])
        with gpu(mb=24576):
            out, (rc, doc) = self.cli("xl.wav", "--seed", "3", *model)
        self.assertEqual(rc, 0, doc)
        self.assertAlmostEqual(float(FfmpegSkill.locate(str(FFMPEG_SKILL_DIR)).probe(out)["duration"]), 10.5, delta=0.1)
        self.assertEqual([c["config_path"] for c in self.fake["calls"]() if c["event"] == "initialize_service"],
                         ["acestep-v15-xl-turbo"])
        self.assertEqual((doc["engine"]["model"], doc["provenance"]["parameters"]["model"]["selected_by"]),
                         ("acestep-v15-xl-turbo", "request"))
        self.assertIn("acestep-v15-xl-turbo model card", doc["provenance"]["engine"]["weights_license"])
        self.assertEqual(sorted(n for n in os.listdir(self.dir) if n.endswith(".wav")), ["xl.wav"])

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


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class ShapingIntegrationTests(unittest.TestCase):
    """Loopable beds and timed stingers through the CLI and the real ffmpeg-skill (ADR-011)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))

    def cli(self, *argv):
        return run_cli(*argv, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")

    def test_a_loop_repeated_ten_times_passes_the_seam_check_and_a_naive_one_does_not(self):
        bed = write_bed_wav(os.path.join(self.dir, "ベッド.wav"), seconds=12.0)
        loop = os.path.join(self.dir, "loop.wav")
        rc, doc = self.cli("loop-bed", bed, "-o", loop, "--duration", "8", "--search", "0.5")
        self.assertEqual(rc, 0, doc)
        self.assertEqual((doc["loop"]["length"], doc["loop"]["frames"]), (8.0, 384000))      # two bars of the fixture
        self.assertGreater(doc["loop"]["similarity"], 0.99)
        self.assertTrue(doc["seam"]["passed"])
        self.assertAlmostEqual(float(self.skill.probe(loop)["duration"]), 8.0, delta=0.001)
        x10 = os.path.join(self.dir, "x10.wav")
        self.skill.join([loop] * 10, x10, 0.0)                                               # repeated end to end
        self.assertEqual(pcm.read_wav(x10).frames, 10 * 384000)
        rc, check = self.cli("seam-check", x10, "--period", str(doc["loop"]["length"]))
        self.assertEqual((rc, check["passed"], len(check["seams"])), (0, True, 9), check["seams"])
        for s in check["seams"]:
            self.assertLess(s["click_ratio"], 2.0)
            self.assertLess(s["level_delta_db"], 0.5)
        naive = os.path.join(self.dir, "naive.wav")
        self.skill.cut(bed, naive, duration=8.005)                                           # ends on a kick's peak
        n10 = os.path.join(self.dir, "n10.wav")
        self.skill.join([naive] * 10, n10, 0.0)
        rc, bad = self.cli("seam-check", n10, "--period", "8.005")
        self.assertEqual((rc, bad["passed"]), (1, False))
        self.assertTrue(all(any(m.startswith("click") for m in s["reasons"]) for s in bad["seams"]), bad["seams"])

    def test_stinger_peaks_land_within_10_ms(self):
        for kind, secs, at, duration, peak_at in (("whoosh", 1.5, 0.6, 1.0, 0.5), ("hit", 1.2, 0.05, 2.0, 1.25),
                                                  ("rise", 2.0, 0.0, 1.5, 1.2)):
            with self.subTest(kind=kind):
                src = write_burst_wav(os.path.join(self.dir, f"{kind}.wav"), kind, secs, at=at)
                out = os.path.join(self.dir, f"stinger-{kind}.wav")
                rc, doc = self.cli("stinger", src, "-o", out, "--duration", str(duration), "--peak-at", str(peak_at),
                                   "--type", kind)
                self.assertEqual(rc, 0, doc)
                measured = pcm.read_wav(out)
                self.assertEqual(measured.frames, int(round(duration * 48000)))
                self.assertLessEqual(abs(pcm.peak(measured)["time"] - peak_at), 0.010)
                self.assertAlmostEqual(float(self.skill.probe(out)["duration"]), duration, delta=0.001)
                self.assertEqual(doc["provenance"]["engine"], None)


@unittest.skipUnless(FFMPEG_SKILL_DIR and QC_SKILL_DIR and HAVE_FFMPEG,
                     "needs ffmpeg-skill and qc-skill (>= 1.3.0) checkouts and ffmpeg/ffprobe on PATH")
class QcLoopSeamIntegrationTests(unittest.TestCase):
    """Roadmap step 7's Done-when (ADR-017): a loop-bed result repeated 10 times passes qc-skill's own loop seam check
    (its ADR-025 `AudioRule.loop_seam`, run through its CLI) at the wrap and all nine seams; a naive cut fails it."""
    # qc-skill sets no default limit; these are this skill's seam-check limits (ADR-011), so the two checks agree
    LIMITS = {"window_sec": 1.0, "max_click_ratio": 3.0, "max_level_diff_db": 1.5, "max_band_diff_db": 6.0}
    CHECKS = ("audio.loop_seam_click_within_limit", "audio.loop_seam_level_within_limit",
              "audio.loop_seam_spectrum_within_limit")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.skill = FfmpegSkill.locate(str(FFMPEG_SKILL_DIR))

    def qc(self, wav, period):
        report = run_qc_loop_seam(QC_SKILL_DIR, wav, [round(period * k, 6) for k in range(1, 10)], self.dir, **self.LIMITS)
        checks = {c["check_id"]: c["status"] for c in report["checks"]}
        seams = next(m for m in report["measurements"] if m["id"] == "audio.loop_seams")["value"]["seams"]
        return report["overall_status"], tuple(checks.get(c) for c in self.CHECKS), seams

    def test_a_loop_bed_repeated_ten_times_passes_qc_skills_loop_seam_check(self):
        for name, write, duration in (("periodic", write_bed_wav, "8"), ("drifting", write_drifting_bed_wav, "8")):
            with self.subTest(bed=name):
                bed = write(os.path.join(self.dir, f"{name}.wav"))
                loop = os.path.join(self.dir, f"{name}-loop.wav")
                rc, doc = run_cli("loop-bed", bed, "-o", loop, "--duration", duration, "--search", "1",
                                  "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")
                self.assertEqual(rc, 0, doc)
                length = doc["loop"]["length"]
                x10 = os.path.join(self.dir, f"{name}-x10.wav")
                self.skill.join([loop] * 10, x10, 0.0)                                       # end to end, no crossfade
                self.assertEqual(pcm.read_wav(x10).frames, 10 * doc["loop"]["frames"])
                status, checks, seams = self.qc(x10, length)
                self.assertEqual((status, checks), ("PASS", ("PASS",) * 3), seams)
                self.assertEqual(len(seams), 10)                                             # the wrap and nine seams
                for s in seams:
                    self.assertLess(s["click"]["ratio"], 2.0, s)
                    self.assertLess(s["level"]["diff_db"], 1.5, s)

    def test_a_naive_cut_repeated_ten_times_fails_it_with_a_click(self):
        bed = write_bed_wav(os.path.join(self.dir, "bed.wav"), seconds=12.0)
        naive = os.path.join(self.dir, "naive.wav")
        self.skill.cut(bed, naive, duration=8.005)                                           # ends on a kick's peak
        n10 = os.path.join(self.dir, "n10.wav")
        self.skill.join([naive] * 10, n10, 0.0)
        status, checks, seams = self.qc(n10, 8.005)
        self.assertEqual((status, checks[0]), ("FAIL", "FAIL"), seams)
        self.assertTrue(all(s["click"]["ratio"] > 3.0 for s in seams), seams)


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class CueSheetIntegrationTests(unittest.TestCase):
    """SFX_CUES through the CLI and the real ffmpeg-skill (ADR-013): effects imported into a library, a cue sheet for
    the fixture's 10 cut times."""

    def test_ten_cut_times_give_ten_cues_at_those_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = os.path.join(tmp, "効果音")
            os.mkdir(lib)
            cli = lambda *a: run_cli(*a, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")   # noqa: E731
            for name, kind, at, lic in (("whoosh", "whoosh", 0.4, "CC0-1.0"), ("pop", "hit", 0.02, "CC0-1.0")):
                src = write_burst_wav(os.path.join(tmp, f"{name}.wav"), kind, 0.8, at=at)
                rc, doc = cli("library-import", src, "--library", lib, "--licence", lic, "--kind", "sfx", "--tags", name)
                self.assertEqual(rc, 0, doc)
            cuts = [1.0, 2.5, 3.0, 4.75, 6.0, 7.2, 8.0, 9.5, 11.0, 12.25]
            out = os.path.join(tmp, "cues.json")
            rc, doc = cli("sfx-cues", "--cuts", ",".join(map(str, cuts)), "--emphasis", "3.1,9.0", "--library", lib,
                          "--min-spacing", "0.4", "-o", out)
            self.assertEqual(rc, 0, doc)
            self.assertEqual([c["time"] for c in doc["cues"] if c["kind"] == "cut"], cuts)
            self.assertEqual([d["time"] for d in doc["dropped"]], [3.1])
            times = [c["time"] for c in doc["cues"]]
            self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.4)
            for c in doc["cues"]:
                self.assertAlmostEqual(c["start"] + c["anchor_in_file"], c["time"], places=6)
            self.assertLessEqual(abs(doc["sources"]["whoosh"]["anchors"]["peak"] - 0.4), 0.05)
            self.assertLessEqual(abs(doc["sources"]["pop"]["anchors"]["onset"] - 0.02), 0.002)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["cues"], doc["cues"])

    def test_reuse_spacing_rotates_and_character_filters(self):
        """ADR-015 through the CLI: two whooshes and two pops imported with their tags and mood; a whoosh is not
        replayed within 30 s (the second-ranked one takes its place), the pops are the bright one, and a bright pop
        with no free file is dropped."""
        with tempfile.TemporaryDirectory() as tmp:
            lib = os.path.join(tmp, "効果音")
            os.mkdir(lib)
            cli = lambda *a: run_cli(*a, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")   # noqa: E731
            for name, kind, at, seed, extra in (("whoosh-a", "whoosh", 0.4, 1, ["--tags", "whoosh"]),
                                                ("whoosh-b", "whoosh", 0.3, 2, ["--tags", "whoosh", "--mood", "falling"]),
                                                ("pop-a", "hit", 0.02, 3, ["--tags", "pop"]),
                                                ("pop-bright", "hit", 0.03, 4, ["--tags", "pop,bright"])):
                src = write_burst_wav(os.path.join(tmp, f"{name}.wav"), kind, 0.8, at=at, seed=seed)
                rc, doc = cli("library-import", src, "--library", lib, "--licence", "CC0-1.0", "--kind", "sfx", *extra)
                self.assertEqual(rc, 0, doc)
            args = ("sfx-cues", "--cuts", "1,10,40", "--emphasis", "5,6", "--library", lib, "--min-reuse-spacing", "30",
                    "--character", "pop=bright")
            rc, doc = cli(*args)
            self.assertEqual(rc, 0, doc)
            self.assertEqual([(c["time"], c["file"]) for c in doc["cues"]],
                             [(1.0, "whoosh-a.wav"), (5.0, "pop-bright.wav"), (10.0, "whoosh-b.wav"), (40.0, "whoosh-a.wav")])
            self.assertEqual([(s["time"], s["file"], s["instead_of"]) for s in doc["substitutions"]],
                             [(10.0, "whoosh-b.wav", "whoosh-a.wav")])
            self.assertEqual([(d["time"], d["by"]) for d in doc["dropped"]], [(6.0, "min_reuse_spacing")])
            for c in doc["cues"]:
                src = doc["sources"][c["source"]]
                self.assertEqual(c["anchor_in_file"], src["anchors"][c["align"]])
                self.assertAlmostEqual(c["start"] + c["anchor_in_file"], c["time"], places=6)
            self.assertLessEqual(abs(doc["sources"]["whoosh@2"]["anchors"]["peak"] - 0.3), 0.05)
            rc, again = cli(*args)
            self.assertEqual((rc, again["cues"], again["dropped"], again["substitutions"]),
                             (0, doc["cues"], doc["dropped"], doc["substitutions"]))
            rc, falling = cli("sfx-cues", "--cuts", "1", "--library", lib, "--character", "whoosh=falling")
            self.assertEqual((rc, [c["file"] for c in falling["cues"]]), (0, ["whoosh-b.wav"]))     # mood matches too


@unittest.skipUnless(FFMPEG_SKILL_DIR and HAVE_FFMPEG, "needs an ffmpeg-skill checkout and ffmpeg/ffprobe on PATH")
class StockLicenceIntegrationTests(unittest.TestCase):
    """Stock tracks the user downloaded, with their licence records, through the CLI and the real ffmpeg-skill
    (ADR-014)."""

    def test_cc_by_attribution_in_the_receipt_and_a_content_id_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib, dl = os.path.join(tmp, "library"), os.path.join(tmp, "ダウンロード")
            os.mkdir(lib)
            os.mkdir(dl)
            cli = lambda *a: run_cli(*a, "--ffmpeg-skill", str(FFMPEG_SKILL_DIR), "--json")   # noqa: E731
            track = write_tone_wav(os.path.join(dl, "morning.wav"), seconds=8.0, rate=48000)
            text = os.path.join(dl, "LICENSE.txt")
            with open(text, "w", encoding="utf-8") as fh:
                fh.write("Creative Commons Attribution 4.0 International Public License\n")
            rc, doc = cli("library-import", track, "--library", lib, "--licence", "CC-BY-4.0", "--kind", "bgm",
                          "--tags", "morning", "--title", "Morning Light", "--author", "Aiko Tanaka",
                          "--licence-url", "https://creativecommons.org/licenses/by/4.0/", "--licence-file", text)
            self.assertEqual(rc, 0, doc)
            rc, pick = cli("library-pick", lib, "-o", os.path.join(tmp, "bed.wav"), "--kind", "bgm", "--tags", "morning",
                           "--duration", "5")
            self.assertEqual(rc, 0, pick)
            line = '"Morning Light" by Aiko Tanaka, licensed under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)'
            self.assertEqual((pick["licence"]["attribution"], pick["provenance"]["licence"]["attribution"]), (line, line))
            hit = write_tone_wav(os.path.join(dl, "hit.wav"), seconds=8.0, rate=48000, freq=330.0)
            rc, doc = cli("library-import", hit, "--library", lib, "--licence", "Royalty-free (store licence)", "--kind", "bgm",
                          "--tags", "hit", "--provider", "ShopX", "--content-id", "registered")
            self.assertEqual(rc, 0, doc)
            rc, pick = cli("library-pick", lib, "-o", os.path.join(tmp, "hit-bed.wav"), "--kind", "bgm", "--tags", "hit")
            self.assertEqual(rc, 0, pick)
            self.assertEqual(pick["licence"]["content_id"], "registered")
            self.assertTrue(any("Content ID" in w for w in pick["warnings"]), pick["warnings"])
            rc, refused = cli("library-import", track, "--library", lib, "--kind", "bgm", "--name", "again.wav",
                              "--licence-url", "https://example.invalid/licence")
            self.assertEqual((rc, refused["error"]["code"]), (2, "LICENCE_REQUIRED"))


if __name__ == "__main__":
    unittest.main()
