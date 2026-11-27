"""The licence record's published shape (roadmap step 12): schemas/licence-record.schema.json is the base of the
Velisa shared licence-record spec (docs/specs/licence-record.md). library-import's stored entry and the record a
reader gets are validated against it, so a field that is renamed, added or dropped fails here and goes through the
spec. The validator below covers only the JSON Schema keywords the file uses; no third-party package is needed."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from helpers import OS_DIR, ROOT, write_tone_wav

from sound_design_skill import library
from sound_design_skill.library import MANIFEST_NAME, LibraryImportRequest, library_import, licence_info
from test_library import StubLibrarySkill

SCHEMA_PATH = ROOT / "schemas" / "licence-record.schema.json"
_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list, "null": type(None)}


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _json_equal(a, b):
    """JSON equality: true and false are not the numbers 1 and 0 (Python's == says they are)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def errors(value, schema, root, where="$"):
    """Every place `value` breaks `schema` (a subset of JSON Schema 2020-12: $ref into $defs, type, const, enum,
    minLength, maxLength, pattern, required, properties, additionalProperties: false, items, allOf, if / then / else;
    the last three and maxLength are used by Velisa's registry copy of the schema)."""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[1]
        return errors(value, root["$defs"][name], root, where)
    out = []
    for sub in schema.get("allOf", []):
        out += errors(value, sub, root, where)
    if "if" in schema:
        branch = schema.get("then") if not errors(value, schema["if"], root, where) else schema.get("else")
        if branch is not None:
            out += errors(value, branch, root, where)
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(value, _TYPES[k]) and not (k != "boolean" and isinstance(value, bool)) for k in kinds):
            return [f"{where}: {value!r} is not {'/'.join(kinds)}"]
    if "const" in schema and not _json_equal(value, schema["const"]):
        out.append(f"{where}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            out.append(f"{where}: empty")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{where}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append(f"{where}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        out += [f"{where}: {k} is missing" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            out += [f"{where}: {k} is not in the schema" for k in value if k not in props]
        for k, v in value.items():
            if k in props:
                out += errors(v, props[k], root, f"{where}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            out += errors(v, schema["items"], root, f"{where}[{i}]")
    return out


class LicenceRecordSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.lib = os.path.join(self.dir, "library")
        os.mkdir(self.lib)

    def check(self, value, definition):
        self.assertEqual(errors(value, {"$ref": f"#/$defs/{definition}"}, self.schema), [])

    def imp(self, name, **kw):
        src = write_tone_wav(os.path.join(self.dir, name), seconds=1.0, rate=8000)
        return library_import(LibraryImportRequest(src, self.lib, kw.pop("licence", "CC-BY-4.0"), kw.pop("kind", "bgm"), **kw),
                              skill=StubLibrarySkill())

    def test_library_import_with_every_field_matches_the_schema(self):
        text = os.path.join(self.dir, "LICENSE.txt")
        with open(text, "w", encoding="utf-8") as fh:
            fh.write("Creative Commons Attribution 4.0 International\n")
        doc = self.imp("full.wav", tags=["piano"], mood=["calm"], loopable=True, source="https://example.invalid/track",
                       author="A. Composer", credit="Music: A. Composer", title="Morning", provider="Example Library",
                       licence_url="https://creativecommons.org/licenses/by/4.0/", licence_file=text,
                       attribution="\"Morning\" by A. Composer, CC BY 4.0", content_id="not-registered")
        self.check(doc["entry"], "entry")
        self.check(doc["licence"], "record")
        self.assertEqual(set(doc["entry"]), set(self.schema["$defs"]["entry"]["properties"]))   # every field written
        with open(os.path.join(self.lib, MANIFEST_NAME), encoding="utf-8") as fh:
            self.check(json.load(fh)["files"]["full.wav"], "entry")

    def test_library_import_with_only_the_licence_matches_the_schema(self):
        doc = self.imp("plain.wav", licence="CC0-1.0", kind="sfx", loudness=False)
        self.check(doc["entry"], "entry")
        self.check(doc["licence"], "record")

    def test_the_record_without_a_declared_licence_matches_the_schema(self):
        record = licence_info(None)
        self.assertEqual((record["licence"], record["status"], record["usable"]), ("unknown", "unknown", False))
        self.check(record, "record")

    def test_the_record_fields_are_the_ones_the_schema_names(self):
        # field names are stable: the record a reader gets has exactly the schema's fields, all required
        record = licence_info({"licence": "CC-BY-4.0"})
        props = self.schema["$defs"]["record"]["properties"]
        self.assertEqual(set(record), set(props))
        self.assertEqual(set(self.schema["$defs"]["record"]["required"]), set(props))

    def test_the_validator_rejects_what_the_schema_does_not_allow(self):
        doc = self.imp("strict.wav")
        renamed = dict(doc["entry"], license=doc["entry"]["licence"])
        del renamed["licence"]
        self.assertTrue(errors(renamed, {"$ref": "#/$defs/entry"}, self.schema))
        self.assertTrue(errors(dict(doc["licence"], verified=True), {"$ref": "#/$defs/record"}, self.schema))
        self.assertTrue(errors(dict(doc["licence"], verified=0), {"$ref": "#/$defs/record"}, self.schema))   # 0 is not false
        self.assertTrue(errors(dict(doc["entry"], kind="music"), {"$ref": "#/$defs/entry"}, self.schema))

    def test_the_schema_names_the_velisa_spec(self):
        self.assertIn("docs/specs/licence-record.md", self.schema["description"])
        self.assertEqual(self.schema["$defs"]["entry"]["properties"]["content_id"]["enum"], list(library.CONTENT_ID_VALUES))


VELISA_SCHEMA = None if OS_DIR is None else OS_DIR / "registry" / "schemas" / "licence-record.schema.json"
SCHEMA_URL = "https://github.com/kajisho5/sound-design-skill/blob/main/schemas/licence-record.schema.json"


@unittest.skipUnless(VELISA_SCHEMA is not None and VELISA_SCHEMA.is_file(),
                     "needs a Velisa checkout with registry/schemas/licence-record.schema.json (AI_VIDEO_PRODUCTION_OS_DIR)")
class VelisaSharedSpecTests(unittest.TestCase):
    """Roadmap step 12: the shared spec lives in Velisa (docs/specs/licence-record.md, with its machine-readable copy in
    registry/schemas/). The spec links this repository's schema, and what library-import writes is valid under Velisa's
    copy too, so the base and the shared spec cannot drift apart unnoticed. Runs in the integration job (Velisa main)."""

    def setUp(self):
        with open(VELISA_SCHEMA, encoding="utf-8") as fh:
            self.shared = json.load(fh)
        self.base = load_schema()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.lib = os.path.join(self.dir, "library")
        os.mkdir(self.lib)

    def test_the_velisa_spec_links_this_schema(self):
        with open(OS_DIR / "docs" / "specs" / "licence-record.md", encoding="utf-8") as fh:
            self.assertIn(SCHEMA_URL, fh.read())

    def test_library_import_output_is_valid_under_the_shared_schema(self):
        src = write_tone_wav(os.path.join(self.dir, "full.wav"), seconds=1.0, rate=8000)
        doc = library_import(LibraryImportRequest(src, self.lib, "CC-BY-4.0", "bgm", tags=["piano"], mood=["calm"], loopable=True,
                                                  source="https://example.invalid/track", author="A. Composer", credit="Music: A. Composer",
                                                  title="Morning", provider="Example Library", attribution="\"Morning\" by A. Composer, CC BY 4.0",
                                                  licence_url="https://creativecommons.org/licenses/by/4.0/", content_id="not-registered"),
                             skill=StubLibrarySkill())
        for value, definition in ((doc["entry"], "entry"), (doc["licence"], "record"), (licence_info(None), "record")):
            self.assertEqual(errors(value, {"$ref": f"#/$defs/{definition}"}, self.shared), [], definition)
            self.assertEqual(errors(value, self.shared, self.shared), [], definition)    # the top-level entry / record switch

    def test_the_shared_schema_keeps_every_base_field(self):
        # the shared spec extends the base (video, images, AI-generated assets); it never renames or drops a field
        for definition in ("entry", "record"):
            base = set(self.base["$defs"][definition]["properties"])
            shared = set(self.shared["$defs"][definition]["properties"])
            self.assertEqual(base - shared, set(), definition)
        self.assertEqual(self.shared["$defs"]["record"]["required"], self.base["$defs"]["record"]["required"])


if __name__ == "__main__":
    unittest.main()
