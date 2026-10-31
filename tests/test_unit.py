"""Unit tests: contract, typed errors, the engine registry and EngineSpec, availability checks (mocked package
lookups, a fake ACE-Step install, a fake checkpoints directory and Hugging Face cache), device / GPU memory detection
(mocked nvidia-smi and sysctl), the worker subprocess, doctor, provenance and the OS registry conformance checks. No
engine, no weights, no GPU, no ffmpeg and no network are needed; nothing is downloaded."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from helpers import (OS_DIR, ROOT, env, fake_acestep_install, fake_checkpoints, fake_hf_snapshot, no_engines_env,
                     run_cli)

from sound_design_skill import SKILL_ID, __version__, device
from sound_design_skill.contract import CAPABILITY_IDS, contract
from sound_design_skill.doctor import doctor
from sound_design_skill.engines import DEFAULT_ENGINE, engine_ids, engines_for, get_engine
from sound_design_skill.engines import acestep, lookup, pkgprobe
from sound_design_skill.engines.base import OPS, EngineRequest, EngineResult, in_range, parse_version
from sound_design_skill.errors import ERROR_TABLE, EXIT_CODES, SoundDesignError
from sound_design_skill.ffmpeg_skill import FLAGS_USED, FfmpegSkill, parse_contract
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
            self.assertEqual(p["lifecycle"], "PROPOSED")       # no op is implemented in this release (ADR-002)
        self.assertFalse(any(t["implemented"] for t in c["tools"] if t["role"] in ("generation", "ingest")))
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
                fake_hf_snapshot(self.cache, "ACE-Step/Ace-Step1.5")
                r = eng.report()
                self.assertEqual((r["status"], r["model"], r["install"], r["lm_models"]), ("OK", "acestep-v15-turbo", None, []))
                self.assertIn("DiT-only", r["lm_note"])

    def test_explicit_checkpoints_dir_is_never_replaced_by_the_cache(self):
        fake_hf_snapshot(self.cache, "ACE-Step/Ace-Step1.5")
        ckpt = os.path.join(self.cache, "checkpoints")
        with self.clean(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt), \
                mock.patch.object(acestep, "python_packages", return_value=pkgs({"ace-step": "1.5.0"}, {"acestep": True})):
            r = get_engine("acestep").inspect()
            self.assertEqual(r["missing"], "model")
            self.assertIn("$SOUND_DESIGN_ACESTEP_CHECKPOINTS", r["detail"])
            fake_checkpoints(ckpt, acestep.REQUIRED_FILES + ("acestep-5Hz-lm-0.6B/model.safetensors",))
            r = get_engine("acestep").inspect()
            self.assertEqual((r["status"], r["lm_models"], r["lm_note"]), ("OK", ["acestep-5Hz-lm-0.6B"], None))

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

    def test_installed_engine_reaches_not_implemented_through_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = fake_checkpoints(os.path.join(tmp, "ckpt"))
            with env(**no_engines_env(tmp)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt), fake_acestep_install(tmp), \
                    self.assertRaises(SoundDesignError) as e:
                run_engine("acestep", req(output_path=os.path.join(tmp, "bed.wav")), 60, tmp)
        self.assertEqual((e.exception.code, e.exception.exit_code), ("NOT_IMPLEMENTED", 5))   # step 2 implements it
        self.assertEqual((e.exception.details["op"], e.exception.details["roadmap_step"]), ("bgm", 2))

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
