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

from helpers import ROOT, write_tone_wav

from sound_design_skill import library
from sound_design_skill.library import MANIFEST_NAME, LibraryImportRequest, library_import, licence_info
from test_library import StubLibrarySkill

SCHEMA_PATH = ROOT / "schemas" / "licence-record.schema.json"
_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list, "null": type(None)}


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def errors(value, schema, root, where="$"):
    """Every place `value` breaks `schema` (a subset of JSON Schema 2020-12: $ref into $defs, type, const, enum,
    minLength, pattern, required, properties, additionalProperties: false, items)."""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[1]
        return errors(value, root["$defs"][name], root, where)
    out = []
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(value, _TYPES[k]) and not (k != "boolean" and isinstance(value, bool)) for k in kinds):
            return [f"{where}: {value!r} is not {'/'.join(kinds)}"]
    if "const" in schema and value is not schema["const"] and value != schema["const"]:
        out.append(f"{where}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            out.append(f"{where}: empty")
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
        self.assertTrue(errors(dict(doc["entry"], kind="music"), {"$ref": "#/$defs/entry"}, self.schema))

    def test_the_schema_names_the_velisa_spec(self):
        self.assertIn("docs/specs/licence-record.md", self.schema["description"])
        self.assertEqual(self.schema["$defs"]["entry"]["properties"]["content_id"]["enum"], list(library.CONTENT_ID_VALUES))


if __name__ == "__main__":
    unittest.main()
