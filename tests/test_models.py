"""Model selection for GENERATE_BGM (ADR-016): ACE-Step 1.5 XL turbo (4B DiT) as an opt-in `--model`, the 2B turbo
still the default. The ModelSpec facts, the GPU memory check (mocked nvidia-smi / sysctl), the model's files in the
checkpoints directory, the worker running the fake engine with the XL config_path (the same API), the guided
refusals (MODEL_NOT_FOUND, INSUFFICIENT_VRAM) before anything runs, the provenance, doctor and the contract. No engine,
no weights, no GPU and no network; nothing is downloaded."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import (CHECKPOINT_FILES, ROOT, XL_CHECKPOINT_FILES, env, fake_acestep_engine, fake_acestep_install,
                     fake_checkpoints, fake_hf_snapshot, gpu, no_engines_env, run_cli)
from test_unit import StubFfmpegSkill, req

from sound_design_skill import device
from sound_design_skill.cli import main as cli_main
from sound_design_skill.contract import contract
from sound_design_skill.doctor import doctor
from sound_design_skill.engines import acestep, get_engine
from sound_design_skill.engines.base import ModelSpec
from sound_design_skill.errors import ERROR_TABLE, SoundDesignError
from sound_design_skill.ops import BgmRequest, generate_bgm
from sound_design_skill.service import run_engine

XL = "acestep-v15-xl-turbo"
TURBO = "acestep-v15-turbo"


class ModelSpecTests(unittest.TestCase):
    """What ADR-016 verified about the XL model, carried as data."""

    def test_two_models_and_the_default_is_unchanged(self):
        s = get_engine("acestep").spec
        self.assertEqual(s.model_ids, (TURBO, XL))
        self.assertEqual((s.default_model, acestep.DEFAULT_MODEL, s.model().id), (TURBO, TURBO, TURBO))
        turbo = s.model(TURBO)
        # the default model's facts are the engine's own, so a caller that never passes --model sees no change
        self.assertEqual((turbo.weights_license, turbo.output_use, turbo.training_data, turbo.min_vram_gb),
                         (s.weights_license, s.output_use, s.training_data, s.min_vram_gb))
        self.assertEqual((turbo.default, turbo.vram_check, turbo.repository), (True, "warn", "ACE-Step/Ace-Step1.5"))
        self.assertTrue(set(turbo.required_files) <= set(acestep.REQUIRED_FILES))
        self.assertEqual(get_engine("ezaudio").spec.models, ())
        self.assertIsNone(get_engine("ezaudio").spec.default_model)

    def test_xl_facts(self):
        m = get_engine("acestep").spec.model(XL)
        self.assertIsInstance(m, ModelSpec)
        self.assertEqual((m.repository, m.parameters, m.default), ("ACE-Step/acestep-v15-xl-turbo", "4B DiT", False))
        self.assertEqual(m.revision, "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee")       # the HF commit read for ADR-016
        self.assertTrue(m.weights_license.startswith("MIT (Hugging Face ACE-Step/acestep-v15-xl-turbo"))
        self.assertIn("Apache-2.0", m.weights_license)                                   # the shared text encoder
        self.assertIn("Generated music can be used for commercial purposes", m.output_use)
        self.assertIn("unverified", m.training_data)
        self.assertEqual((m.min_vram_gb, m.vram_check), (20.0, "refuse"))
        self.assertTrue(any(">= 12 GB" in r for r in m.vram_documented))
        self.assertTrue(any(">= 20 GB without offload" in r for r in m.vram_documented))
        self.assertIn("no offload", m.vram_note)
        self.assertAlmostEqual(m.download_gb, 19.95)
        self.assertEqual(len(m.required_files), 7)                                        # config, index, 4 shards, silence latent
        self.assertIn(f"{XL}/silence_latent.pt", m.required_files)
        self.assertTrue(all(f.startswith(XL + "/") for f in m.required_files))
        self.assertIn(f"acestep-download --model {XL}", m.install)
        self.assertIn("SOUND_DESIGN_ACESTEP_CHECKPOINTS", m.install)
        self.assertIn("ADR-016", m.licence_checked)
        self.assertTrue(m.sources and all(u.startswith("https://") for u in m.sources))
        self.assertIn("ADR-016", " ".join(get_engine("acestep").spec.licence_notes))

    def test_unknown_model_is_invalid_argument(self):
        for eid, name in (("acestep", "acestep-v15-xl-sft"), ("acestep", ""), ("ezaudio", XL)):
            with self.assertRaises(SoundDesignError, msg=(eid, name)) as e:
                get_engine(eid).spec.model(name)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    def test_spec_publishes_models_as_plain_json(self):
        d = get_engine("acestep").spec.to_dict()
        self.assertEqual(d["default_model"], TURBO)
        self.assertEqual([m["id"] for m in d["models"]], [TURBO, XL])
        self.assertIsInstance(d["models"][1]["required_files"], list)
        self.assertEqual(json.loads(json.dumps(d)), d)


class VramFitTests(unittest.TestCase):
    def fit(self, **kw):
        with gpu(**kw):
            return device.vram_fit(20.0, device.detect())

    def test_measured_gpus(self):
        self.assertEqual(self.fit(mb=24576)["status"], "OK")
        self.assertEqual(self.fit(mb=20475)["status"], "OK")         # a 20 GB card reports a little less (tolerance)
        low = self.fit(mb=16384)
        self.assertEqual((low["status"], low["vram_total_mb"], low["required_gb"]), ("INSUFFICIENT", 16384, 20.0))
        self.assertIn("below", low["detail"])
        self.assertEqual(self.fit(mb=19400)["status"], "INSUFFICIENT")
        mac = self.fit(apple_mb=16384)
        self.assertEqual((mac["status"], mac["accelerator"]), ("INSUFFICIENT", "mps"))
        self.assertIn("shared with the CPU", mac["detail"])
        self.assertEqual(self.fit(apple_mb=32768)["status"], "OK")

    def test_no_gpu_unknown_and_not_checked(self):
        self.assertEqual(self.fit(cpu_only=True)["status"], "NO_GPU")
        unknown = self.fit(unknown=True)
        self.assertEqual(unknown["status"], "UNKNOWN")
        self.assertIn("not measured", unknown["detail"])
        self.assertEqual(device.vram_fit(None, {"status": "OK", "vram_total_mb": 1})["status"], "NOT_CHECKED")
        self.assertEqual(device.vram_fit(20.0, None)["status"], "NOT_CHECKED")
        self.assertEqual(set(device.REFUSED_FITS), {"INSUFFICIENT", "NO_GPU"})


class ModelFilesTests(unittest.TestCase):
    """The XL model's files: in the one checkpoints directory ACE-Step reads (file checks only, nothing imported)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def test_explicit_checkpoints_directory(self):
        ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
        eng = get_engine("acestep")
        with env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=ckpt):
            self.assertEqual(eng.model_status(TURBO)["status"], "OK")
            xl = eng.model_status(XL)
            self.assertEqual((xl["status"], len(xl["missing"])), ("MISSING", 7))
            self.assertIn("acestep-download --model", xl["install"])
            self.assertNotIn("one checkpoints directory", xl["detail"])               # already explicit
            fake_checkpoints(ckpt, XL_CHECKPOINT_FILES[:-1])                          # no silence latent yet
            self.assertEqual(eng.model_status(XL)["missing"], [f"{XL}/silence_latent.pt"])
            fake_checkpoints(ckpt, XL_CHECKPOINT_FILES)
            self.assertEqual(eng.model_status(XL), {"status": "OK", "missing": [], "install": None,
                                                    "detail": f"{XL} ($SOUND_DESIGN_ACESTEP_CHECKPOINTS)"})
            with fake_acestep_install(self.dir):
                r = eng.report()
            self.assertEqual((r["status"], r["default_model"]), ("OK", TURBO))
            self.assertEqual({k: v["status"] for k, v in r["model_status"].items()}, {TURBO: "OK", XL: "OK"})

    def test_hugging_face_cache_says_to_use_a_checkpoints_directory(self):
        fake_hf_snapshot(self.dir, "ACE-Step/Ace-Step1.5")
        fake_hf_snapshot(self.dir, "ACE-Step/acestep-v15-xl-turbo", [f.split("/", 1)[1] for f in XL_CHECKPOINT_FILES])
        with env(**no_engines_env(self.dir)):
            xl = get_engine("acestep").model_status(XL)
        self.assertEqual(xl["status"], "MISSING")                     # its own cache folder is not where ACE-Step looks
        self.assertIn("SOUND_DESIGN_ACESTEP_CHECKPOINTS", xl["detail"])

    def test_without_the_main_model_nothing_is_ready(self):
        with env(**no_engines_env(self.dir)):
            xl = get_engine("acestep").model_status(XL)
        self.assertEqual(xl["status"], "MISSING")
        self.assertIn("main model", xl["detail"])


class XlEngineTests(unittest.TestCase):
    """AceStepEngine.process with the XL model in the real worker, against the fake engine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"), CHECKPOINT_FILES + XL_CHECKPOINT_FILES)
        for cm in (env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=self.ckpt)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
        engine_cm = fake_acestep_engine(self.dir)
        self.fake = engine_cm.__enter__()
        self.addCleanup(engine_cm.__exit__, None, None, None)
        self.n = 0

    def run_xl(self, model=XL, **kw):
        self.n += 1
        run_dir = os.path.join(self.dir, f"run{self.n}")
        os.mkdir(run_dir)
        return run_engine("acestep", req(output_path=os.path.join(run_dir, "bgm.wav"), duration=10.0,
                                         params={"device": "auto", "model": model}, **kw), 60, run_dir)

    def events(self, name):
        return [c for c in self.fake["calls"]() if c["event"] == name]

    def test_the_same_api_with_the_xl_config_path(self):
        res = self.run_xl(seed=7)
        self.assertEqual((res.model, res.seed, res.sample_rate), (XL, 7, 48000))
        init = self.events("initialize_service")
        self.assertEqual(len(init), 1)
        self.assertEqual((init[0]["config_path"], init[0]["vae_checkpoint"], init[0]["device"]), (XL, "official", "auto"))
        self.assertEqual(Path(init[0]["checkpoint_dir"]), Path(self.ckpt).resolve())
        gen = self.events("generate_music")
        self.assertEqual(len(gen), 1)
        self.assertTrue(gen[0]["llm_handler_is_none"])
        self.assertEqual((gen[0]["params"]["task_type"], gen[0]["config"]["seeds"]), ("text2music", [7]))
        self.assertEqual(self.events("download") + self.events("google_probe_failed"), [])
        self.assertNotIn("acestep", sys.modules)

    def test_missing_xl_files_are_refused_before_import(self):
        os.remove(os.path.join(self.ckpt, XL, "model-00003-of-00004.safetensors"))
        with self.assertRaises(SoundDesignError) as e:
            self.run_xl()
        self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")
        self.assertEqual(e.exception.details["missing"], [f"{XL}/model-00003-of-00004.safetensors"])
        self.assertIn("acestep-download --model", e.exception.details["install"])
        self.assertEqual(self.fake["calls"](), [])

    def test_a_dit_download_attempt_is_refused(self):
        """ACE-Step fetches a missing XL DiT through ensure_dit_model -> download_submodel -> _smart_download."""
        self.fake["set_mode"]("force_dit_download")
        with self.assertRaises(SoundDesignError) as e:
            self.run_xl()
        self.assertEqual(e.exception.code, "MODEL_NOT_FOUND")
        self.assertIn("_smart_download", e.exception.message)
        self.assertIn("ACE-Step/acestep-v15-xl-turbo", e.exception.message)
        self.assertEqual(self.events("download") + self.events("google_probe_failed") + self.events("generate_music"), [])

    def test_an_unknown_model_is_refused_in_the_worker(self):
        with self.assertRaises(SoundDesignError) as e:
            self.run_xl(model="acestep-v15-xl-sft")
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        self.assertEqual(self.fake["calls"](), [])


class GenerateBgmModelTests(unittest.TestCase):
    """generate_bgm with --model, in-process with the real worker and a stub ffmpeg-skill."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.out_dir = os.path.join(self.dir, "out")
        os.mkdir(self.out_dir)
        self.ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"), CHECKPOINT_FILES + XL_CHECKPOINT_FILES)
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

    def configs(self):
        return [c["config_path"] for c in self.fake["calls"]() if c["event"] == "initialize_service"]

    def assert_nothing_ran(self):
        self.assertEqual(self.fake["calls"](), [])
        self.assertEqual(self.skill.probed, [])
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_the_default_needs_no_gpu_check_and_is_unchanged(self):
        with mock.patch("sound_design_skill.ops.detect", side_effect=AssertionError("the default model checks no GPU")):
            doc = self.bgm(seed=5)
        self.assertEqual((doc["request"]["model"], doc["engine"]["model"], self.configs()), (TURBO, TURBO, [TURBO]))
        m = doc["provenance"]["parameters"]["model"]
        self.assertEqual((m["id"], m["selected_by"], m["default"], m["vram"]), (TURBO, "default", True, None))
        self.assertEqual(doc["engine"]["weights_license"], acestep.SPEC.weights_license)
        self.assertEqual(doc["warnings"], [])

    def test_xl_on_a_24_gb_gpu(self):
        with gpu(mb=24576):
            doc = self.bgm(model=XL, seed=5)
        self.assertEqual(self.configs(), [XL])
        self.assertEqual((doc["request"]["model"], doc["engine"]["model"]), (XL, XL))
        spec = get_engine("acestep").spec.model(XL)
        for block in (doc["engine"], doc["provenance"]["engine"]):           # the model that ran, with its own terms
            self.assertEqual((block["model"], block["weights_license"], block["output_use"]),
                             (XL, spec.weights_license, spec.output_use))
        self.assertEqual(doc["provenance"]["engine"]["training_data"], spec.training_data)
        m = doc["provenance"]["parameters"]["model"]
        self.assertEqual((m["id"], m["selected_by"], m["repository"], m["revision"], m["parameters"]),
                         (XL, "request", "ACE-Step/acestep-v15-xl-turbo", spec.revision, "4B DiT"))
        self.assertEqual((m["vram"]["status"], m["vram"]["vram_total_mb"], m["min_vram_gb"]), ("OK", 24576, 20.0))
        self.assertIn("ADR-016", m["licence_checked"])
        self.assertEqual(doc["warnings"], [])
        self.assertTrue(os.path.isfile(self.out()))
        json.dumps(doc, ensure_ascii=True, allow_nan=False)

    def test_same_api_same_generation_parameters(self):
        with gpu(mb=24576):
            xl = self.bgm(model=XL, seed=11, output_path=self.out("xl.wav"))
        turbo = self.bgm(seed=11, output_path=self.out("turbo.wav"))
        for k in ("generation_params", "generation_config", "engine_api", "lm"):
            self.assertEqual(xl["provenance"]["parameters"][k], turbo["provenance"]["parameters"][k], k)

    def test_too_little_gpu_memory_is_refused_before_anything_runs(self):
        for kw, fit in (({"mb": 12288}, "INSUFFICIENT"), ({"mb": 16384}, "INSUFFICIENT"), ({"cpu_only": True}, "NO_GPU"),
                        ({"apple_mb": 16384}, "INSUFFICIENT")):
            with self.subTest(kw), gpu(**kw), self.assertRaises(SoundDesignError) as e:
                self.bgm(model=XL, seed=1)
            err = e.exception
            self.assertEqual((err.code, err.exit_code, err.retryable), ("INSUFFICIENT_VRAM", 3, False))
            self.assertEqual((err.details["model"], err.details["required_gb"], err.details["vram"]["status"]), (XL, 20.0, fit))
            self.assertEqual(err.details["default_model"], TURBO)
            self.assertIn(f"omit --model to use {TURBO}", err.message)
            self.assertIn(">= 20 GB", err.message)
        self.assert_nothing_ran()

    def test_unmeasured_gpu_memory_runs_with_a_warning(self):
        with gpu(unknown=True):
            doc = self.bgm(model=XL, seed=2)
        self.assertEqual(doc["engine"]["model"], XL)
        self.assertEqual(doc["provenance"]["parameters"]["model"]["vram"]["status"], "UNKNOWN")
        self.assertTrue(any("not measured" in w and ">= 20 GB" in w for w in doc["warnings"]), doc["warnings"])

    def test_missing_xl_files_are_model_not_found_with_the_download_line(self):
        for name in XL_CHECKPOINT_FILES:
            os.remove(os.path.join(self.ckpt, name))
        with gpu(mb=24576), self.assertRaises(SoundDesignError) as e:
            self.bgm(model=XL)
        err = e.exception
        self.assertEqual((err.code, err.exit_code), ("MODEL_NOT_FOUND", 3))
        self.assertIn("acestep-download --model acestep-v15-xl-turbo", err.details["install"])
        self.assertEqual((len(err.details["missing"]), err.details["download_gb"], err.details["default_model"]), (7, 19.95, TURBO))
        self.assert_nothing_ran()
        doc = self.bgm(seed=3)                                     # the default still works without XL
        self.assertEqual(doc["engine"]["model"], TURBO)

    def test_an_unknown_model_is_refused_first(self):
        Path(self.ckpt, "vae", "diffusion_pytorch_model.safetensors").unlink()       # the engine is not OK either
        for kw in ({"model": "acestep-v15-xl-sft"}, {"model": ""}, {"model": XL, "engine": "ezaudio"}):
            with self.subTest(kw), self.assertRaises(SoundDesignError) as e:
                self.bgm(**kw)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)
        self.assert_nothing_ran()

    def test_the_default_named_explicitly_is_the_same_bed(self):
        a = self.bgm(seed=21, output_path=self.out("a.wav"))
        b = self.bgm(seed=21, output_path=self.out("b.wav"), model=TURBO)
        self.assertEqual(a["output"]["sha256"], b["output"]["sha256"])
        self.assertEqual(b["provenance"]["parameters"]["model"]["selected_by"], "request")

    def test_cli(self):
        with gpu(mb=24576):
            rc, doc = run_cli("generate-bgm", "--prompt", "calm piano", "--duration", "12", "-o", self.out("cli.wav"),
                              "--seed", "4", "--model", XL, "--ffmpeg-skill", os.path.join(self.dir, "no-ffmpeg-skill"),
                              "--json")
        self.assertEqual((rc, doc["error"]["code"]), (3, "FFMPEG_SKILL_UNAVAILABLE"))   # every model check passed first
        with gpu(mb=8192):
            rc, doc = run_cli("generate-bgm", "--prompt", "calm piano", "--duration", "12", "-o", self.out("cli.wav"),
                              "--model", XL, "--json")
        self.assertEqual((rc, doc["ok"], doc["error"]["code"]), (3, False, "INSUFFICIENT_VRAM"))
        self.assertEqual(doc["error"]["details"]["default_model"], TURBO)
        with self.assertRaises(SystemExit) as e, mock.patch("sys.stderr"):
            cli_main(["generate-bgm", "--prompt", "x", "--duration", "12", "-o", self.out("x.wav"), "--model", "xl", "--json"])
        self.assertEqual(e.exception.code, 2)
        self.assertEqual(self.fake["calls"](), [])


class DoctorModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.ckpt = fake_checkpoints(os.path.join(self.dir, "ckpt"))
        for cm in (env(**no_engines_env(self.dir)), env(SOUND_DESIGN_ACESTEP_CHECKPOINTS=self.ckpt), fake_acestep_install(self.dir)):
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)

    def models(self, **kw):
        with gpu(**kw):
            d = doctor(os.path.join(self.dir, "no-ffmpeg-skill"))
        return d, d["checks"]["operations"]["bgm"]["models"]

    def test_xl_absent_is_not_a_problem(self):
        d, models = self.models(mb=24576)
        self.assertEqual(d["checks"]["operations"]["bgm"]["default_model"], TURBO)
        self.assertEqual(list(models), [TURBO, XL])
        t, x = models[TURBO], models[XL]
        self.assertEqual((t["status"], t["default"], t["flag"], t["install"]), ("OK", True, None, None))
        self.assertEqual((x["status"], x["ready"], x["flag"], x["refusal"]["code"]), ("MISSING", False, f"--model {XL}", "MODEL_NOT_FOUND"))
        self.assertIn("acestep-download --model", x["install"])
        self.assertEqual(d["checks"]["engines"]["acestep"]["status"], "OK")
        self.assertFalse(any(XL in p for p in d["problems"]))
        self.assertFalse(any(XL in w for w in d["warnings"]))
        self.assertEqual(sorted(d["checks"]["engines"]), ["acestep", "ezaudio"])      # unchanged keys (CI checks them)

    def test_xl_present(self):
        fake_checkpoints(self.ckpt, XL_CHECKPOINT_FILES)
        _, models = self.models(mb=24576)
        self.assertEqual((models[XL]["status"], models[XL]["vram"]["status"], models[XL]["refusal"]), ("OK", "OK", None))
        d, models = self.models(mb=12288)
        x = models[XL]
        self.assertEqual((x["status"], x["vram"]["status"], x["refusal"]["code"]), ("OK", "INSUFFICIENT", "INSUFFICIENT_VRAM"))
        self.assertTrue(any(w.startswith(f"model {XL}: on disk") for w in d["warnings"]), d["warnings"])
        self.assertTrue(models[TURBO]["vram"]["status"] == "OK")
        d, models = self.models(unknown=True)
        self.assertEqual(models[XL]["vram"]["status"], "UNKNOWN")
        self.assertTrue(any("not measured" in w for w in d["warnings"] if XL in w), d["warnings"])

    def test_human_output_lists_the_models(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with gpu(mb=24576), redirect_stdout(buf):
            cli_main(["doctor", "--ffmpeg-skill", os.path.join(self.dir, "none")])
        text = buf.getvalue()
        self.assertIn(f"model {TURBO}", text)
        self.assertRegex(text, rf"model {XL} +MISSING")
        self.assertIn(f"acestep-download --model {XL}", text)


class ContractModelTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        self.bgm = next(t for t in self.c["tools"] if t["tool_id"] == "sound-design/generate-bgm")

    def test_flag_request_key_and_specs(self):
        self.assertIn(f"[--model {TURBO}|{XL}]", self.bgm["argv"])
        m = self.bgm["models"]
        self.assertEqual((m["flag"], m["request_key"], m["default"], m["choices"]), ("--model", "model", TURBO, [TURBO, XL]))
        self.assertIn("provenance.parameters.model", m["recorded_in"])
        self.assertIn("ADR-016", m["adr"])
        self.assertIn("model", BgmRequest.__dataclass_fields__)
        reg = self.c["engines"]["registered"]
        self.assertEqual(([x["id"] for x in reg["acestep"]["models"]], reg["acestep"]["default_model"]), ([TURBO, XL], TURBO))
        self.assertEqual((reg["ezaudio"]["models"], reg["ezaudio"]["default_model"]), ([], None))
        self.assertEqual(self.c["errors"]["INSUFFICIENT_VRAM"], {"exit_code": 3, "retryable": False})
        self.assertEqual(ERROR_TABLE["INSUFFICIENT_VRAM"], (3, False))
        self.assertEqual(self.c["device"]["model_fit"]["refused"], ["INSUFFICIENT", "NO_GPU"])
        self.assertIn("ADR-016", self.c["policy"]["models"])
        self.assertTrue(any(f.startswith("parameters.model") for f in self.c["provenance"]["fields"]))

    def test_docs_give_the_vram_guidance(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        adr = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
        for text in (skill, readme):
            self.assertIn(f"--model {XL}", text)
            self.assertIn("20 GB", text)
            self.assertIn("INSUFFICIENT_VRAM", text)
        self.assertRegex(adr, r"## ADR-016 ")
        self.assertIn("d4a0b288", adr)
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
        self.assertIn(f"## {version}", changelog)


if __name__ == "__main__":
    unittest.main()
