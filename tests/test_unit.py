"""Unit tests: contract, typed errors, the engine registry and EngineSpec, availability checks (mocked package
lookups, a fake ACE-Step install, a fake checkpoints directory and Hugging Face cache), device / GPU memory detection
(mocked nvidia-smi and sysctl), the worker subprocess and its network guard, the ACE-Step adapter against a fake
engine with the verified API (ADR-004), GENERATE_BGM through a stub ffmpeg-skill, doctor, provenance and the OS
registry conformance checks. No engine, no weights, no GPU, no ffmpeg and no network are needed; nothing is
downloaded."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from helpers import (CHECKPOINT_FILES, OS_DIR, ROOT, env, fake_acestep_engine, fake_acestep_install, fake_checkpoints,
                     fake_hf_snapshot, no_engines_env, run_cli)

from sound_design_skill import SKILL_ID, __version__, device
from sound_design_skill.contract import CAPABILITY_IDS, contract
from sound_design_skill.doctor import doctor
from sound_design_skill.engines import DEFAULT_ENGINE, engine_ids, engines_for, get_engine
from sound_design_skill.engines import acestep, lookup, netguard, pkgprobe
from sound_design_skill.engines.base import OPS, EngineRequest, EngineResult, in_range, parse_version
from sound_design_skill.errors import ERROR_TABLE, EXIT_CODES, SoundDesignError
from sound_design_skill.ffmpeg_skill import FLAGS_USED, FfmpegSkill, SkillInfo, parse_contract
from sound_design_skill.ops import BGM_SCHEMA, BgmRequest, caption_from, generate_bgm, tolerance_for
from sound_design_skill.procs import TIMEOUT_EXIT, run_process_group
from sound_design_skill.provenance import build_provenance, sha256_file, sha256_text
from sound_design_skill.service import engine_interpreter, run_engine

ENGINES = ["acestep"]
SRC = ROOT / "src" / "sound_design_skill"
NVIDIA_ROW = "0, NVIDIA GeForce RTX 3060, 12288, 11800, 550.54.15\n"


def pkgs(dists=None, modules=None, error=None):
    """A python_packages() answer."""
    return {"python": "3.11.0", "dists": dict(dists or {}), "modules": dict(modules or {}), "interpreter": "this Python",
            "error": error}


def req(**kw):
    base = dict(op="bgm", prompt="calm piano bed, 70 bpm", duration=30.0, output_path="bed.wav", seed=42)
    base.update(kw)
    return EngineRequest(**base)


class ErrorTests(unittest.TestCase):
    def test_table_shape_and_retryable(self):
        for code, (exit_code, retryable) in ERROR_TABLE.items():
            self.assertTrue(re.fullmatch(r"[A-Z_]+", code))
            self.assertGreaterEqual(exit_code, 2)
            self.assertIsInstance(retryable, bool)
        self.assertEqual({c for c, (_, r) in ERROR_TABLE.items() if r}, {"TOOL_ERROR", "ENGINE_TIMEOUT", "CANCELLED"})

    def test_error_document(self):
        e = SoundDesignError("ENGINE_TIMEOUT", "slow", {"engine": "x"})
        self.assertEqual(e.to_dict(), {"code": "ENGINE_TIMEOUT", "message": "slow", "retryable": True, "details": {"engine": "x"}})
        self.assertEqual(e.exit_code, EXIT_CODES["ENGINE_TIMEOUT"])
        self.assertFalse(SoundDesignError("TOOL_ERROR", "x", retryable=False).retryable)
        with self.assertRaises(ValueError):
            SoundDesignError("NOPE", "x")
        self.assertEqual(SoundDesignError.from_dict({"code": "NOPE", "message": "m"}).code, "INTERNAL_ERROR")


class ContractTests(unittest.TestCase):
    def test_os_registry_keys_and_provides(self):
        c = contract()
        for k in ("skill_id", "id", "skill", "version", "contract_version", "provides", "dependencies", "not_provided"):
            self.assertIn(k, c)
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), (SKILL_ID, __version__, "1.0"))
        self.assertEqual([p["id"] for p in c["provides"]], ["audio.bgm_generate", "audio.sfx_generate", "audio.library_import"])
        tool_ids = {t["tool_id"] for t in c["tools"]}
        for p in c["provides"]:
            self.assertEqual(set(p), {"id", "lifecycle", "tool_id"})
            self.assertIn(p["tool_id"], tool_ids)
        # GENERATE_BGM is implemented since step 2 (ADR-004); SFX (step 3) and library mode (step 4) are not
        self.assertEqual({p["id"]: p["lifecycle"] for p in c["provides"]},
                         {"audio.bgm_generate": "EXPERIMENTAL", "audio.sfx_generate": "PROPOSED", "audio.library_import": "PROPOSED"})
        self.assertEqual({t["tool_id"]: t["implemented"] for t in c["tools"] if t["role"] in ("generation", "ingest")},
                         {"sound-design/generate-bgm": True, "sound-design/generate-sfx": False, "sound-design/library-import": False})
        bgm = next(t for t in c["tools"] if t["tool_id"] == "sound-design/generate-bgm")
        self.assertEqual(bgm["schema"], BGM_SCHEMA)
        self.assertIn("off", bgm["language_model"])
        self.assertFalse(bgm["deterministic"])                                # unverified on the real engine
        for d in c["dependencies"]:
            self.assertIsInstance(d["version_range"], str)
            self.assertFalse(d["version_range"].replace(".", "").isdigit(), d)
        self.assertEqual([d["id"] for d in c["dependencies"]], ["ffmpeg-skill"] + ENGINES)
        self.assertFalse(c["execution"]["network"])
        self.assertTrue(any("audio-production-skill" in n for n in c["not_provided"]))     # MIX owns ミックス
        json.dumps(c, allow_nan=False)

    def test_errors_engines_and_device_mirror_the_code(self):
        c = contract()
        self.assertEqual(c["errors"], {k: {"exit_code": e, "retryable": r} for k, (e, r) in ERROR_TABLE.items()})
        self.assertEqual(sorted(c["engines"]["registered"]), engine_ids())
        self.assertEqual(c["engines"]["default"], {"bgm": "acestep", "sfx": None})
        ff = next(d for d in c["dependencies"] if d["id"] == "ffmpeg-skill")
        self.assertEqual(ff["flags"], {t: list(f) for t, f in FLAGS_USED.items()})
        self.assertEqual(ff["tools"], ["probe"])
        self.assertEqual(len(c["device"]["acestep_tiers"]), len(device.ACESTEP_TIERS))
        self.assertEqual(set(CAPABILITY_IDS), {"bgm", "sfx", "library"})

    def test_version_comes_from_pyproject(self):
        want = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
        self.assertEqual(__version__, want)

    def test_cli_contract_and_skill_alias(self):
        rc, doc = run_cli("contract", "--json")
        rc2, doc2 = run_cli("skill", "--json")
        self.assertEqual((rc, rc2, doc["skill_id"]), (0, 0, SKILL_ID))
        self.assertEqual(doc, doc2)

    def test_json_is_ascii_and_parses_on_a_legacy_code_page_console(self):
        """Windows consoles and pipes default to a code page (cp1252) that cannot encode Japanese (ADR-003)."""
        child = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0",
                     PYTHONPATH=os.pathsep.join([str(ROOT / "src")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        proc = subprocess.run([sys.executable, "-m", "sound_design_skill.cli", "contract", "--json"], capture_output=True, env=child,
                              timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        self.assertTrue(all(b < 128 for b in proc.stdout))
        doc = json.loads(proc.stdout.decode("ascii"))
        self.assertIn("ミックス", doc["not_provided"][0])
        human = subprocess.run([sys.executable, "-m", "sound_design_skill.cli", "contract"], capture_output=True, env=child, timeout=60)
        self.assertEqual(human.returncode, 0, human.stderr.decode("utf-8", "replace"))

    def test_no_shell_no_direct_ffmpeg_no_downloads_no_engine_imports(self):
        for f in SRC.rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("shell=True", text, f.name)
            self.assertNotIn('"ffmpeg"', text.replace('"ffmpeg", "ffprobe"', ""), f.name)   # media goes through ffmpeg-skill
            for banned in ("hf_hub_download", "snapshot_download", "urllib.request", "requests.", "urlopen", "torch.hub",
                           "http.client", "import acestep", "from acestep", "import torch", "from torch", "import diffusers",
                           "import transformers"):
                self.assertNotIn(banned, text, f"{f.name}: {banned}")


class EngineSpecTests(unittest.TestCase):
    def test_registry_and_defaults(self):
        self.assertEqual(engine_ids(), ENGINES)
        self.assertEqual(set(DEFAULT_ENGINE), set(OPS))
        self.assertEqual(DEFAULT_ENGINE, {"bgm": "acestep", "sfx": None})     # OS ADR-012; SFX engine ADR is step 3
        self.assertEqual(engines_for("bgm"), ["acestep"])
        self.assertEqual(engines_for("sfx"), [])
        with self.assertRaises(SoundDesignError) as e:
            get_engine("musicgen")
        self.assertEqual(e.exception.code, "ENGINE_UNAVAILABLE")

    def test_spec_carries_licences_output_use_and_sources(self):
        s = get_engine("acestep").spec
        self.assertTrue(s.code_license.startswith("MIT") and s.weights_license.startswith("MIT"))
        self.assertIn("Apache-2.0", s.weights_license)                       # Qwen3-Embedding-0.6B
        self.assertIn("commercial", s.output_use)
        self.assertIn("unverified", s.training_data)
        self.assertIn("ADR-001", s.licence_checked)
        self.assertTrue(s.sources and all(u.startswith("https://") for u in s.sources))
        self.assertEqual((s.duration_range, s.min_vram_gb, s.kind), ((10.0, 600.0), 4.0, "python"))
        self.assertFalse(s.network_required or s.credit_required)
        self.assertIsNone(s.watermark)
        self.assertIn("SOUND_DESIGN_ACESTEP_PYTHON", s.install)
        d = s.to_dict()
        for k in ("code_license", "weights_license", "output_use", "training_data", "credit", "credit_required", "watermark"):
            self.assertIn(k, d)
        json.dumps(d)

    def test_parse_version_and_range(self):
        self.assertEqual(parse_version("1.5.0"), (1, 5, 0))
        self.assertIsNone(parse_version("unknown"))
        self.assertTrue(in_range((1, 5, 0), *acestep.SUPPORTED))
        self.assertTrue(in_range((1, 5, 9), *acestep.SUPPORTED))
        self.assertFalse(in_range((0, 1, 0), *acestep.SUPPORTED))             # the unrelated PyPI `ace-step`
        self.assertFalse(in_range((1, 6), *acestep.SUPPORTED))

    def test_request_result_round_trip(self):
        r = req(lyrics="[verse]\nこんにちは", params={"bpm": 70})
        self.assertEqual(EngineRequest.from_dict(r.to_dict()), r)
        res = EngineResult(output_path="b.wav", engine_version="1.5.0", model="acestep-v15-turbo", sample_rate=48000, seed=42,
                           duration=30.0, extra={"lm": None})
        self.assertEqual(EngineResult.from_dict(res.to_dict()), res)

    def test_request_validation_needs_no_engine(self):
        eng = get_engine("acestep")
        eng.validate(req())
        for bad in (req(op="sfx"), req(prompt="  "), req(duration=5), req(duration=601), req(duration=True), req(seed=-1),
                    req(seed=2 ** 32), req(seed=1.5)):
            with self.assertRaises(SoundDesignError, msg=bad) as e:
                eng.validate(bad)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        eng.validate(req(seed=None, duration=600))


class AceStepAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = self.tmp.name

    def clean(self, **extra):
        values = dict(no_engines_env(self.cache))
        values.update(extra)
        return env(**values)

    def test_statuses(self):
        eng = get_engine("acestep")
        with self.clean():
            with mock.patch.object(acestep, "python_packages", return_value=pkgs({"ace-step": None}, {"acestep": False})):
                r = eng.report()
                self.assertEqual((r["status"], r["missing"]), ("MISSING", "package"))
                self.assertIn("ACE-Step-1.5", r["install"])
            with mock.patch.object(acestep, "python_packages", return_value=pkgs({"ace-step": "0.1.0"}, {"acestep": True})):
                self.assertEqual(eng.inspect()["status"], "INCOMPATIBLE")
            with mock.patch.object(acestep, "python_packages", return_value=pkgs({"ace-step": "1.5.0"}, {"acestep": True})):
                r = eng.inspect()
                self.assertEqual((r["status"], r["missing"], r["version"]), ("MISSING", "model", "1.5.0"))
                with self.assertRaises(SoundDesignError) as e:
                    eng.require()
                self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")
                fake_hf_snapshot(self.cache, "ACE-Step/Ace-Step1.5", acestep.REQUIRED_FILES[:2])   # partial download
                self.assertIn("vae/", eng.inspect()["detail"])
                fake_hf_snapshot(self.cache, "ACE-Step/Ace-Step1.5", acestep.REQUIRED_FILES)   # no 1.7B LM: ACE-Step would download
                r = eng.inspect()
                self.assertEqual((r["status"], r["missing"]), ("MISSING", "model"))
                self.assertIn("would download", r["detail"])
                self.assertIn("acestep-5Hz-lm-1.7B", r["detail"])
                fake_hf_snapshot(self.cache, "ACE-Step/Ace-Step1.5")
                r = eng.report()
                self.assertEqual((r["status"], r["model"], r["install"], r["lm_models"]),
                                 ("OK", "acestep-v15-turbo", None, ["acestep-5Hz-lm-1.7B"]))
                self.assertIn("not used", r["lm_note"])               # present on disk, never loaded (ADR-004)
                self.assertIn("Qwen3", r["lm_status"])

    def test_explicit_checkpoints_dir_is_never_replaced_by_the_cache(self):
        fake_hf_snapshot(self.cache, "ACE-Step/Ace-Step1.5")
        ckpt = os.path.join(self.cache, "checkpoints")
        with self.clean(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt), \
                mock.patch.object(acestep, "python_packages", return_value=pkgs({"ace-step": "1.5.0"}, {"acestep": True})):
            r = get_engine("acestep").inspect()
            self.assertEqual(r["missing"], "model")
            self.assertIn("$SOUND_DESIGN_ACESTEP_CHECKPOINTS", r["detail"])
            fake_checkpoints(ckpt, CHECKPOINT_FILES + ("acestep-5Hz-lm-0.6B/model.safetensors",))
            r = get_engine("acestep").inspect()
            self.assertEqual((r["status"], r["lm_models"]), ("OK", ["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B"]))
            self.assertIn("not used", r["lm_note"])

    def test_component_check_mirrors_acestep_weights_rule(self):
        """acestep/model_downloader.py _contains_model_weights: any one of its weight file names counts (a sharded
        checkpoint has only an index file)."""
        ckpt = Path(self.cache) / "ckpt"
        fake_checkpoints(str(ckpt), acestep.REQUIRED_FILES + ("acestep-5Hz-lm-1.7B/model.safetensors.index.json",))
        self.assertEqual(acestep.missing_components(ckpt), [])
        (ckpt / "acestep-5Hz-lm-1.7B" / "model.safetensors.index.json").unlink()
        (ckpt / "acestep-5Hz-lm-1.7B" / "config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(acestep.missing_components(ckpt), ["acestep-5Hz-lm-1.7B"])

    def test_real_lookup_finds_a_fake_install_without_importing_it(self):
        ckpt = fake_checkpoints(os.path.join(self.cache, "ckpt"))
        with self.clean(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt), fake_acestep_install(self.cache):
            r = get_engine("acestep").inspect()
        self.assertEqual((r["status"], r["version"]), ("OK", "1.5.0"))
        self.assertNotIn("acestep", sys.modules)

    def test_engine_interpreter_lookup_runs_pkgprobe_under_that_interpreter(self):
        with self.clean(SOUND_DESIGN_ACESTEP_PYTHON=sys.executable):
            found = lookup.python_packages(["ace-step", "no-such-dist-xyz"], ["json", "no_such_module_xyz"],
                                           "SOUND_DESIGN_ACESTEP_PYTHON")
            self.assertIsNone(found["error"])
            self.assertEqual(found["interpreter"], sys.executable)
            self.assertEqual(found["modules"], {"json": True, "no_such_module_xyz": False})
            self.assertIsNone(found["dists"]["no-such-dist-xyz"])
            r = get_engine("acestep").inspect()
            if found["dists"]["ace-step"] is None:
                self.assertEqual((r["status"], r["interpreter"]), ("MISSING", sys.executable))

    def test_engine_interpreter_that_is_not_executable_is_missing(self):
        with self.clean(SOUND_DESIGN_ACESTEP_PYTHON=os.path.join(self.cache, "no-such-python")):
            r = get_engine("acestep").inspect()
        self.assertEqual(r["status"], "MISSING")
        self.assertIn("SOUND_DESIGN_ACESTEP_PYTHON", r["detail"])

    def test_pkgprobe_script(self):
        proc = subprocess.run([sys.executable, "-I", pkgprobe.__file__, "no-such-dist-xyz", "json"], capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual((doc["dists"], doc["modules"]), ({"no-such-dist-xyz": None}, {"json": True}))
        with redirect_stderr(io.StringIO()):
            self.assertEqual(pkgprobe.main(["only-one"]), 2)


class DeviceTests(unittest.TestCase):
    def which(self, present):
        return lambda name: f"/usr/bin/{name}" if name in present else None

    def test_parse_nvidia_smi(self):
        gpus = device.parse_nvidia_smi(NVIDIA_ROW + "1, Tesla T4, 15360, 15000, 550.54.15\ngarbage\n")
        self.assertEqual([g["name"] for g in gpus], ["NVIDIA GeForce RTX 3060", "Tesla T4"])
        self.assertEqual((gpus[0]["vram_total_mb"], gpus[0]["vram_free_mb"], gpus[0]["driver"]), (12288, 11800, "550.54.15"))
        self.assertEqual(device.parse_nvidia_smi(""), [])

    def test_tiers_follow_the_acestep_table(self):
        self.assertIn("DiT only", device.acestep_tier(4.0))
        self.assertIn("lm-0.6B", device.acestep_tier(8.0))
        self.assertIn("1.7B", device.acestep_tier(12.0))
        self.assertIn("lm-4B", device.acestep_tier(80.0))
        self.assertIsNone(device.acestep_tier(None))

    def test_nvidia_gpu(self):
        with mock.patch.object(device.shutil, "which", self.which({"nvidia-smi"})), \
                mock.patch.object(device, "run_process_group", return_value=(0, NVIDIA_ROW, "")) as run:
            d = device.detect()
        argv = run.call_args[0][0]
        self.assertEqual(argv[1:], device.NVIDIA_QUERY)                      # a fixed list argv, never a shell
        self.assertEqual((d["status"], d["accelerator"], d["vram_total_mb"], d["vram_free_mb"]), ("OK", "cuda", 12288, 11800))
        self.assertTrue(d["meets_acestep_minimum"])
        self.assertIn("1.7B", d["acestep_recommendation"])

    def test_nvidia_smi_that_fails_is_unknown(self):
        with mock.patch.object(device.shutil, "which", self.which({"nvidia-smi"})), \
                mock.patch.object(device, "run_process_group", return_value=(9, "", "NVIDIA-SMI has failed")):
            d = device.detect()
        self.assertEqual((d["status"], d["accelerator"], d["vram_total_mb"]), ("UNKNOWN", "cuda", None))
        self.assertIn("NVIDIA-SMI has failed", d["detail"])

    def test_apple_silicon(self):
        with mock.patch.object(device.shutil, "which", self.which({"sysctl"})), \
                mock.patch.object(device.platform, "system", return_value="Darwin"), \
                mock.patch.object(device.platform, "machine", return_value="arm64"), \
                mock.patch.object(device, "run_process_group", return_value=(0, str(16 * 1024 ** 3), "")):
            d = device.detect()
        self.assertEqual((d["accelerator"], d["unified_memory"], d["vram_total_mb"]), ("mps", True, 16384))

    def test_cpu_only_and_rocm(self):
        with mock.patch.object(device.shutil, "which", self.which(set())), \
                mock.patch.object(device.platform, "system", return_value="Linux"):
            d = device.detect()
        self.assertEqual((d["status"], d["accelerator"], d["acestep_recommendation"], d["meets_acestep_minimum"]),
                         ("CPU_ONLY", "cpu", None, None))
        with mock.patch.object(device.shutil, "which", self.which({"rocm-smi"})), \
                mock.patch.object(device.platform, "system", return_value="Linux"):
            self.assertEqual(device.detect()["accelerator"], "rocm")

    def test_real_detection_runs_anywhere(self):
        d = device.detect()
        self.assertIn(d["status"], ("OK", "CPU_ONLY", "UNKNOWN"))
        json.dumps(d)


class WorkerTests(unittest.TestCase):
    """The real worker subprocess path (no engine needed)."""

    def test_process_group_is_killed_on_timeout(self):
        t0 = time.time()
        code, _, _ = run_process_group([sys.executable, "-c", "import time; time.sleep(30)"], 1.0)
        self.assertEqual(code, TIMEOUT_EXIT)
        self.assertLess(time.time() - t0, 15)

    def test_argv_must_be_a_clean_list(self):
        with self.assertRaises(SoundDesignError):
            run_process_group([], 1.0)
        with self.assertRaises(SoundDesignError):
            run_process_group(["a\x00b"], 1.0)
        self.assertEqual(run_process_group([os.path.join(tempfile.gettempdir(), "no-such-exe")], 5.0)[0], 127)

    def test_unknown_engine_is_refused_before_a_process_starts(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SoundDesignError) as e:
            run_engine("nope", req(), 30, tmp)
        self.assertEqual(e.exception.code, "ENGINE_UNAVAILABLE")

    def test_bad_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for t in (0, -1, True):
                with self.assertRaises(SoundDesignError):
                    run_engine("acestep", req(), t, tmp)

    def test_engine_interpreter(self):
        with env(SOUND_DESIGN_ACESTEP_PYTHON="/opt/ACE-Step-1.5/.venv/bin/python"):
            self.assertEqual(engine_interpreter("acestep"), "/opt/ACE-Step-1.5/.venv/bin/python")
        with env(SOUND_DESIGN_ACESTEP_PYTHON=None):
            self.assertEqual(engine_interpreter("acestep"), sys.executable)

    def test_worker_reports_structured_errors(self):
        from sound_design_skill.service import worker_argv, worker_env
        with tempfile.TemporaryDirectory() as tmp:
            rq, out = os.path.join(tmp, "req.json"), os.path.join(tmp, "out.json")
            Path(rq).write_text(json.dumps({"engine_id": "nope", "request": req().to_dict()}), encoding="utf-8")
            code, _, err = run_process_group(worker_argv(rq, out), 60, env=worker_env())
            self.assertEqual(code, 1, err)
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual((payload["ok"], payload["error"]["code"], payload["error"]["retryable"]),
                             (False, "ENGINE_UNAVAILABLE", False))
            self.assertEqual(worker_env()["HF_HUB_OFFLINE"], "1")

    def test_missing_engine_is_reported_through_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp, env(**no_engines_env(tmp)), self.assertRaises(SoundDesignError) as e:
            run_engine("acestep", req(output_path=os.path.join(tmp, "bed.wav")), 60, tmp)
        self.assertIn(e.exception.code, ("ENGINE_UNAVAILABLE", "MODEL_NOT_FOUND"))
        self.assertEqual(e.exception.exit_code, 3)

    def test_invalid_request_is_refused_in_the_worker_before_the_engine_is_looked_up(self):
        with tempfile.TemporaryDirectory() as tmp, env(**no_engines_env(tmp)), self.assertRaises(SoundDesignError) as e:
            run_engine("acestep", req(duration=3), 60, tmp)
        self.assertEqual((e.exception.code, e.exception.details["range"]), ("INVALID_ARGUMENT", [10.0, 600.0]))

    def test_an_engine_without_process_is_still_not_implemented(self):
        class Bare(acestep.AceStepEngine):
            process = acestep.Engine.process
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = fake_checkpoints(os.path.join(tmp, "ckpt"))
            with env(**no_engines_env(tmp)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt), fake_acestep_install(tmp), \
                    self.assertRaises(SoundDesignError) as e:
                Bare().process(req(output_path=os.path.join(tmp, "bed.wav")))
        self.assertEqual((e.exception.code, e.exception.exit_code), ("NOT_IMPLEMENTED", 5))

    def test_engine_timeout_is_retryable(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("sound_design_skill.service.worker_argv", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]), \
                self.assertRaises(SoundDesignError) as e:
            run_engine("acestep", req(), 1.0, tmp)
        self.assertEqual((e.exception.code, e.exception.retryable), ("ENGINE_TIMEOUT", True))

    def test_worker_without_result_is_engine_failed(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("sound_design_skill.service.worker_argv", return_value=[sys.executable, "-c", "raise SystemExit(3)"]), \
                self.assertRaises(SoundDesignError) as e:
            run_engine("acestep", req(), 30, tmp)
        self.assertEqual(e.exception.code, "ENGINE_FAILED")

    def test_malformed_result_is_invalid_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "engine_result.json")
            script = f"import json; json.dump({{'ok': True, 'result': {{'output_path': 'x'}}}}, open({out!r}, 'w'))"
            with mock.patch("sound_design_skill.service.worker_argv", return_value=[sys.executable, "-c", script]), \
                    self.assertRaises(SoundDesignError) as e:
                run_engine("acestep", req(), 30, tmp)
        self.assertEqual(e.exception.code, "INVALID_RESULT")


class NetGuardTests(unittest.TestCase):
    """The worker refuses the network before any engine import (ADR-004): ModelScope is not covered by HF_HUB_OFFLINE."""

    def test_non_loopback_connects_and_lookups_are_refused_loopback_works(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        undo = netguard.install()
        try:
            for host in ("www.google.com", "huggingface.co", "modelscope.cn"):
                with self.assertRaises(netguard.NetworkBlocked):
                    socket.getaddrinfo(host, 443)
                with self.assertRaises(netguard.NetworkBlocked):
                    socket.gethostbyname(host)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with self.assertRaises(netguard.NetworkBlocked):
                    s.connect(("192.0.2.1", 443))                     # TEST-NET-1: refused before any packet
                with self.assertRaises(netguard.NetworkBlocked):
                    s.connect_ex(("www.google.com", 443))
            finally:
                s.close()
            with self.assertRaises(OSError):                          # what ACE-Step's _can_access_google catches
                socket.create_connection(("www.google.com", 443), timeout=1)
            c = socket.create_connection(server.getsockname(), timeout=5)
            c.close()
        finally:
            undo()
            server.close()
        self.assertIs(socket.socket.connect, socket.socket.connect)   # undone: the real methods are back
        self.assertTrue(netguard.is_local_host("::1") and netguard.is_local_host("localhost"))
        self.assertFalse(netguard.is_local_host("10.0.0.1"))

    def test_worker_installs_the_guard_before_the_engine_is_imported(self):
        """In a fresh interpreter with the worker's environment, and in the worker's source order."""
        script = ("import socket, sys; from sound_design_skill.engines import netguard; netguard.install(); "
                  "\ntry:\n    socket.getaddrinfo('modelscope.cn', 443)\nexcept netguard.NetworkBlocked:\n    sys.exit(0)\nsys.exit(1)")
        from sound_design_skill.service import worker_env
        code, _, err = run_process_group([sys.executable, "-c", script], 60, env=worker_env())
        self.assertEqual(code, 0, err)
        text = (SRC / "engines" / "worker.py").read_text(encoding="utf-8")
        self.assertLess(text.index("netguard.install()"), text.index("from . import get_engine"))


class AceStepEngineTests(unittest.TestCase):
    """AceStepEngine.process in the real worker subprocess, against the fake `acestep` package (tests/helpers.py) that
    has the API surface ADR-004 verified and writes a deterministic WAV for a seed and a duration."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
        stack = [env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=self.ckpt)]
        for cm in stack:
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        engine_cm = fake_acestep_engine(self.dir)
        self.fake = engine_cm.__enter__()
        self.addCleanup(engine_cm.__exit__, None, None, None)
        self.n = 0

    def run_engine(self, **kw):
        self.n += 1
        run_dir = os.path.join(self.dir, f"run{self.n}")
        os.mkdir(run_dir)
        kw.setdefault("output_path", os.path.join(run_dir, "bgm.wav"))
        kw.setdefault("duration", 10.0)
        return run_engine("acestep", req(**kw), 60, run_dir)

    def events(self, name):
        return [c for c in self.fake["calls"]() if c["event"] == name]

    def test_generates_through_the_verified_api(self):
        res = self.run_engine(seed=7, prompt="calm piano bed, 70 bpm")
        self.assertTrue(os.path.isfile(res.output_path))
        self.assertEqual((res.seed, res.sample_rate, res.model, res.engine_version), (7, 48000, "acestep-v15-turbo", "1.5.0"))
        with wave.open(res.output_path) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getnframes()), (48000, 2, 480000))
        init = self.events("initialize_service")
        self.assertEqual(len(init), 1)
        self.assertEqual((init[0]["config_path"], init[0]["vae_checkpoint"], init[0]["device"], init[0]["prefer_source"]),
                         ("acestep-v15-turbo", "official", "auto", None))
        self.assertEqual(Path(init[0]["checkpoint_dir"]), Path(self.ckpt).resolve())   # via ACESTEP_CHECKPOINTS_DIR
        self.assertEqual(init[0]["hf_offline"], "1")
        gen = self.events("generate_music")
        self.assertEqual(len(gen), 1)
        p, cfg = gen[0]["params"], gen[0]["config"]
        self.assertTrue(gen[0]["llm_handler_is_none"])                 # the 5 Hz LM is never loaded
        self.assertEqual((p["task_type"], p["caption"], p["duration"], p["seed"]), ("text2music", "calm piano bed, 70 bpm", 10.0, 7))
        self.assertEqual((p["instrumental"], p["lyrics"]), (True, "[Instrumental]"))
        self.assertFalse(p["thinking"] or p["use_cot_metas"] or p["use_cot_caption"] or p["use_cot_language"])
        self.assertEqual((cfg["batch_size"], cfg["use_random_seed"], cfg["seeds"], cfg["audio_format"]), (1, False, [7], "wav"))
        self.assertEqual(res.extra["lm"]["enabled"], False)
        self.assertIn("ca1e85fe", res.extra["api"])
        self.assertEqual(self.events("download") + self.events("google_probe_failed"), [])
        self.assertNotIn("acestep", sys.modules)                      # imported in the worker only

    def test_same_seed_and_duration_give_the_same_file(self):
        digest = lambda r: hashlib.sha256(Path(r.output_path).read_bytes()).hexdigest()   # noqa: E731
        a, b, c = self.run_engine(seed=42), self.run_engine(seed=42), self.run_engine(seed=43)
        self.assertEqual(digest(a), digest(b))
        self.assertNotEqual(digest(a), digest(c))

    def test_lyrics_turn_off_instrumental(self):
        self.run_engine(lyrics="[verse]\nこんにちは")
        p = self.events("generate_music")[0]["params"]
        self.assertEqual((p["instrumental"], p["lyrics"]), (False, "[verse]\nこんにちは"))

    def test_an_omitted_seed_is_drawn_and_reported(self):
        res = self.run_engine(seed=None)
        self.assertEqual(self.events("generate_music")[0]["config"]["seeds"], [res.seed])

    def test_a_download_attempt_is_refused_and_never_reaches_the_network(self):
        self.fake["set_mode"]("force_download")
        with self.assertRaises(SoundDesignError) as e:
            self.run_engine()
        self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")
        self.assertIn("_smart_download", e.exception.message)
        # ACE-Step's own download functions and Google probe were replaced, so none of them ran
        self.assertEqual(self.events("download") + self.events("google_probe_failed"), [])
        self.assertEqual(self.events("generate_music"), [])

    def test_checkpoints_acestep_would_complete_by_download_are_refused_before_import(self):
        (Path(self.ckpt) / "acestep-5Hz-lm-1.7B" / "model.safetensors").unlink()
        with self.assertRaises(SoundDesignError) as e:
            self.run_engine()
        self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")
        self.assertIn("would download", e.exception.message)
        self.assertEqual(self.fake["calls"](), [])

    def test_engine_failures_are_structured(self):
        for mode, code, text in (("init_fail", "ENGINE_FAILED", "CUDA out of memory"), ("fail", "ENGINE_FAILED", "diffusion failed"),
                                 ("wrong_seed", "INVALID_RESULT", "seed")):
            self.fake["set_mode"](mode)
            with self.assertRaises(SoundDesignError, msg=mode) as e:
                self.run_engine(seed=5)
            self.assertEqual(e.exception.code, code, mode)
            self.assertIn(text, e.exception.message, mode)

    def test_arguments_the_engine_cannot_take_are_refused_in_the_worker(self):
        for kw in ({"prompt": "x" * 513}, {"lyrics": "  "}, {"params": {"device": "tpu"}}):
            with self.assertRaises(SoundDesignError, msg=kw) as e:
                self.run_engine(**kw)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)
        self.assertEqual(self.fake["calls"](), [])


class StubFfmpegSkill:
    """ffmpeg-skill stand-in for the unit tests: probe reads the WAV header with the wave module (the integration
    tests use the real ffmpeg-skill)."""

    def __init__(self):
        self.probed = []

    def info(self):
        return SkillInfo(Path("stub"), "2.23.0", "1.0", {"probe": {}}, [])

    def probe(self, path):
        self.probed.append(path)
        try:
            with wave.open(path) as w:
                rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        except (wave.Error, EOFError) as e:
            raise SoundDesignError("UNSUPPORTED_MEDIA", f"stub probe: {e}")
        return {"duration": n / rate, "video": None, "stream_counts": {"video": 0, "audio": 1},
                "audio": {"codec": "pcm_s16le", "sample_rate": rate, "channels": ch}}


class GenerateBgmTests(unittest.TestCase):
    """GENERATE_BGM end to end in-process: checks, the real worker with the fake engine, validation through a stub
    ffmpeg-skill, provenance and placement (ADR-004)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.out_dir = os.path.join(self.dir, "out")
        os.mkdir(self.out_dir)
        self.ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
        for cm in (env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=self.ckpt)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        engine_cm = fake_acestep_engine(self.dir)
        self.fake = engine_cm.__enter__()
        self.addCleanup(engine_cm.__exit__, None, None, None)
        self.skill = StubFfmpegSkill()

    def out(self, name="bed.wav"):
        return os.path.join(self.out_dir, name)

    def bgm(self, **kw):
        kw.setdefault("output_path", self.out())
        kw.setdefault("duration", 12.0)
        kw.setdefault("prompt", "calm piano bed")
        return generate_bgm(BgmRequest(**kw), skill=self.skill)

    def assert_clean(self):
        self.assertEqual([n for n in os.listdir(self.out_dir) if n.startswith(".sound-design-")], [])

    def test_bed_with_provenance(self):
        doc = self.bgm(tags=["ambient", " 70 bpm "], seed=1234)
        self.assertEqual((doc["schema"], doc["ok"], doc["capability"]), (BGM_SCHEMA, True, "audio.bgm_generate"))
        self.assertEqual(doc["output"]["path"], self.out())
        self.assertAlmostEqual(doc["output"]["duration"], 12.0, places=3)
        self.assertEqual((doc["output"]["sample_rate"], doc["output"]["channels"]), (48000, 2))
        self.assertEqual(doc["request"], {"prompt": "calm piano bed, ambient, 70 bpm", "duration": 12.0, "seed": 1234,
                                          "seed_source": "given"})
        p = doc["provenance"]
        self.assertEqual((p["op"], p["seed"], p["prompt"]["text"]), ("bgm", 1234, "calm piano bed, ambient, 70 bpm"))
        self.assertEqual(p["output"]["sha256"], sha256_file(self.out()))
        self.assertEqual(p["output"]["sha256"], doc["output"]["sha256"])
        self.assertEqual(p["duration"]["requested"], 12.0)
        self.assertAlmostEqual(p["duration"]["measured"], 12.0, places=3)
        self.assertEqual(p["duration"]["tolerance"], tolerance_for(12.0))
        e = p["engine"]
        self.assertEqual((e["id"], e["version"], e["model"]), ("acestep", "1.5.0", "acestep-v15-turbo"))
        self.assertTrue(e["code_license"].startswith("MIT") and e["weights_license"].startswith("MIT"))
        self.assertIn("commercial", e["output_use"])
        self.assertIn("Qwen3", e["lm_status"])
        self.assertNotIn("interpreter", e)
        self.assertEqual(p["parameters"]["tags"], ["ambient", "70 bpm"])
        self.assertEqual(p["parameters"]["lm"]["enabled"], False)
        self.assertEqual(p["parameters"]["generation_config"]["seeds"], [1234])
        self.assertEqual(p["ffmpeg_skill"]["version"], "2.23.0")
        self.assertFalse(p["network"])
        self.assertEqual(self.skill.probed[-1].endswith("bgm.wav"), True)   # probed in the run dir, before placement
        self.assert_clean()
        self.assertNotIn("acestep", sys.modules)
        json.dumps(doc, ensure_ascii=True, allow_nan=False)

    def test_seed_is_drawn_when_omitted_and_the_same_seed_repeats_the_file(self):
        first = self.bgm(output_path=self.out("a.wav"))
        self.assertEqual(first["request"]["seed_source"], "drawn")
        self.assertTrue(0 <= first["provenance"]["seed"] < 2 ** 32)
        again = self.bgm(output_path=self.out("b.wav"), seed=first["provenance"]["seed"])
        self.assertEqual(again["output"]["sha256"], first["output"]["sha256"])     # deterministic for a given seed

    def test_a_wrong_duration_is_invalid_result_and_nothing_is_written(self):
        self.fake["set_mode"]("short")
        with self.assertRaises(SoundDesignError) as e:
            self.bgm()
        self.assertEqual(e.exception.code, "INVALID_RESULT")
        self.assertIn("duration", e.exception.message)
        self.assertFalse(os.path.exists(self.out()))
        self.assert_clean()

    def test_checks_run_before_any_work(self):
        Path(self.out("exists.wav")).write_bytes(b"x")
        lyrics = self.out("lyrics.wav")
        Path(lyrics).write_text("[verse]\nla la", encoding="utf-8")
        cases = [
            ({"output_path": self.out("exists.wav")}, "OUTPUT_EXISTS"),
            ({"output_path": self.out("bed.mp3")}, "INVALID_ARGUMENT"),
            ({"output_path": os.path.join(self.dir, "no-such-dir", "bed.wav")}, "INVALID_ARGUMENT"),
            ({"output_path": self.out_dir + os.sep + "sub.wav", "prompt": "  ", "tags": []}, "INVALID_ARGUMENT"),
            ({"prompt": "x" * 600}, "INVALID_ARGUMENT"),
            ({"seed": -1}, "INVALID_ARGUMENT"),
            ({"seed": 2 ** 32}, "INVALID_ARGUMENT"),
            ({"seed": True}, "INVALID_ARGUMENT"),
            ({"duration": 5.0}, "INVALID_ARGUMENT"),
            ({"duration": 601.0}, "INVALID_ARGUMENT"),
            ({"device": "tpu"}, "INVALID_ARGUMENT"),
            ({"timeout": 0}, "INVALID_ARGUMENT"),
            ({"lyrics_path": self.out("no-lyrics.txt")}, "INPUT_NOT_FOUND"),
            ({"lyrics_path": lyrics, "output_path": lyrics, "overwrite": True}, "OUTPUT_IS_INPUT"),
            ({"engine": "musicgen"}, "ENGINE_UNAVAILABLE"),
        ]
        for kw, code in cases:
            with self.assertRaises(SoundDesignError, msg=kw) as e:
                self.bgm(**kw)
            self.assertEqual(e.exception.code, code, kw)
        self.assertEqual(self.fake["calls"](), [])                    # no worker reached the engine
        self.assertEqual(self.skill.probed, [])
        self.assert_clean()

    def test_engine_is_checked_before_ffmpeg_skill(self):
        os.remove(os.path.join(self.ckpt, "vae", "diffusion_pytorch_model.safetensors"))
        with self.assertRaises(SoundDesignError) as e:
            generate_bgm(BgmRequest(output_path=self.out(), duration=12.0, prompt="x",
                                    ffmpeg_skill_dir=os.path.join(self.dir, "no-ffmpeg-skill")))
        self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")     # not FFMPEG_SKILL_UNAVAILABLE
        self.assertEqual(self.fake["calls"](), [])

    def test_overwrite_and_lyrics(self):
        Path(self.out()).write_bytes(b"old")
        lyrics = self.out("words.txt")
        Path(lyrics).write_text("[verse]\n静かな朝", encoding="utf-8")
        doc = self.bgm(overwrite=True, lyrics_path=lyrics, seed=3)
        self.assertNotEqual(Path(self.out()).read_bytes(), b"old")
        self.assertEqual(doc["provenance"]["parameters"]["instrumental"], False)
        self.assertEqual(doc["provenance"]["parameters"]["lyrics_sha256"], sha256_text("[verse]\n静かな朝"))
        self.assertEqual(self.fake["calls"]()[-1]["params"]["lyrics"], "[verse]\n静かな朝")

    def test_timeout_kills_the_engine_and_cleans_up(self):
        self.fake["set_mode"]("sleep")
        t0 = time.time()
        with self.assertRaises(SoundDesignError) as e:
            self.bgm(timeout=3)
        self.assertEqual((e.exception.code, e.exception.retryable), ("ENGINE_TIMEOUT", True))
        self.assertLess(time.time() - t0, 30)
        self.assertFalse(os.path.exists(self.out()))
        self.assert_clean()

    def test_caption_from(self):
        self.assertEqual(caption_from(" lo-fi ", ["piano", " ", "rain"]), "lo-fi, piano, rain")
        self.assertEqual(caption_from(None, ["ambient"]), "ambient")
        self.assertEqual(caption_from("", []), "")
        self.assertEqual((tolerance_for(10.0), tolerance_for(300.0)), (0.1, 3.0))

    def test_cli_reports_a_missing_engine_as_json(self):
        with env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=os.path.join(self.dir, "empty")):
            rc, doc = run_cli("generate-bgm", "--prompt", "calm piano", "--tags", "ambient,70 bpm", "--duration", "12",
                              "-o", self.out(), "--seed", "9", "--json")
        self.assertEqual((rc, doc["ok"], doc["error"]["code"]), (3, False, "MODEL_NOT_FOUND"))
        self.assertTrue(doc["error"]["details"]["install"])
        self.assertFalse(os.path.exists(self.out()))


class FfmpegSkillContractTests(unittest.TestCase):
    def doc(self, drop=None):
        tools = []
        for name, flags in FLAGS_USED.items():
            props = {f: {"type": "string"} for f in flags if (name, f) != drop}
            tools.append({"name": name, "audio_only": True, "input_schema": {"properties": props}})
        return {"skill": {"version": "2.21.0"}, "contract_version": "1.0", "tools": tools}

    def test_capability_detection(self):
        info = parse_contract(self.doc(), Path("."))
        self.assertTrue(info.supported, info.problems)
        self.assertEqual(info.version, "2.21.0")
        self.assertFalse(parse_contract(self.doc(drop=("probe", "inputs")), Path(".")).supported)
        self.assertIn("'probe' is missing", parse_contract(dict(self.doc(), tools=[]), Path(".")).problems[0])
        self.assertTrue(parse_contract(dict(self.doc(), contract_version="2.0"), Path(".")).warnings)

    def test_locate_explicit_dir_is_never_replaced(self):
        with self.assertRaises(SoundDesignError) as e:
            FfmpegSkill.locate(os.path.join(tempfile.gettempdir(), "no-such-ffmpeg-skill"))
        self.assertEqual(e.exception.code, "FFMPEG_SKILL_UNAVAILABLE")
        self.assertEqual(len(e.exception.details["tried"]), 1)


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_engine_device_and_ops(self):
        missing = os.path.join(tempfile.gettempdir(), "no-such-ffmpeg-skill")
        with tempfile.TemporaryDirectory() as tmp, env(**no_engines_env(tmp)), \
                mock.patch.object(acestep, "python_packages", return_value=pkgs({"ace-step": None}, {})), \
                mock.patch.object(device.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None), \
                mock.patch.object(device, "run_process_group", return_value=(0, NVIDIA_ROW, "")):
            d = doctor(missing)
            rc, cli_doc = run_cli("doctor", "--ffmpeg-skill", missing, "--offline", "--json")
        self.assertEqual(sorted(d["checks"]["engines"]), ENGINES)
        e = d["checks"]["engines"]["acestep"]
        self.assertEqual(e["status"], "MISSING")
        self.assertTrue(e["install"])
        for k in ("code_license", "weights_license", "output_use", "credit_required", "watermark"):
            self.assertIn(k, e)
        dev = d["checks"]["device"]
        self.assertEqual((dev["accelerator"], dev["vram_total_mb"], dev["vram_free_mb"]), ("cuda", 12288, 11800))
        self.assertEqual(d["unavailable_engines"], ENGINES)
        self.assertEqual(d["checks"]["ffmpeg_skill"]["status"], "MISSING")
        self.assertEqual((d["status"], d["ok"]), ("fail", False))
        self.assertEqual(set(d["checks"]["operations"]), {"bgm", "sfx", "library"})
        self.assertFalse(any(o["ready"] for o in d["checks"]["operations"].values()))
        self.assertIsNone(d["checks"]["operations"]["sfx"]["engine"])
        self.assertIn("step 3", d["checks"]["operations"]["sfx"]["note"])
        self.assertEqual((rc, cli_doc["status"], cli_doc["offline"]), (1, "fail", True))
        self.assertEqual(cli_doc["checks"]["device"]["vram_total_mb"], 12288)

    def test_cpu_only_is_a_warning_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp, env(**no_engines_env(tmp)), \
                mock.patch.object(device.shutil, "which", lambda name: None), \
                mock.patch.object(device.platform, "system", return_value="Linux"):
            d = doctor(os.path.join(tmp, "none"))
        self.assertEqual(d["checks"]["device"]["status"], "CPU_ONLY")
        self.assertTrue(any(w.startswith("device:") for w in d["warnings"]))
        self.assertFalse(any(p.startswith("device") for p in d["problems"]))

    def test_engines_command(self):
        with tempfile.TemporaryDirectory() as tmp, env(**no_engines_env(tmp)):
            rc, doc = run_cli("engines", "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(doc["engines"]), ENGINES)
        self.assertTrue(all(e["usable_offline"] for e in doc["engines"].values()))


class ProvenanceTests(unittest.TestCase):
    def test_block_carries_licences_seed_and_no_local_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bed.wav")
            Path(out).write_bytes(b"RIFF")
            report = dict(get_engine("acestep").spec.to_dict(), version="1.5.0", model="acestep-v15-turbo",
                          interpreter="/secret/venv/bin/python")
            p = build_provenance(op="bgm", engine=report, prompt="静かなピアノ", seed=42, duration={"requested": 30.0, "measured": 30.02},
                                 output_path=out, parameters={"lm": None})
            self.assertEqual(p["output"]["sha256"], sha256_file(out))
        self.assertEqual(p["prompt"], {"text": "静かなピアノ", "sha256": sha256_text("静かなピアノ")})
        self.assertEqual((p["seed"], p["duration"]["requested"]), (42, 30.0))
        self.assertEqual(p["engine"]["id"], "acestep")
        self.assertIn("commercial", p["engine"]["output_use"])
        self.assertNotIn("interpreter", p["engine"])
        self.assertFalse(p["network"])
        self.assertEqual(p["skill"], {"id": SKILL_ID, "version": __version__})


@unittest.skipUnless(OS_DIR, "needs an AI-video-production-OS checkout (AI_VIDEO_PRODUCTION_OS_DIR or ../AI-video-production-OS)")
class ConformanceTests(unittest.TestCase):
    def test_registry_checks_pass(self):
        sys.path.insert(0, str(OS_DIR))
        try:
            from registry import conformance as conf
        finally:
            sys.path.remove(str(OS_DIR))
        results = conf.run_static_checks(contract())
        results.append(conf.check_no_unsafe_shell_out(str(SRC)))
        results.append(conf.check_doctor_status(lambda: doctor()))
        self.assertEqual([r.check for r in results], ["publishes_contract", "lifecycle_declared", "dependency_version_ranges",
                                                      "no_unsafe_shell_out", "doctor_status"])
        for r in results:
            self.assertEqual(r.status, "PASS", f"{r.check}: {r.detail}")


if __name__ == "__main__":
    unittest.main()
