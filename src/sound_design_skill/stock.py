"""Stock music and SFX sources (roadmap step 10, ADR-014), as doctor reports them. This skill downloads nothing: the
user downloads a track and its licence (text or URL) from the source, then registers them with library-import, which
records the licence, the attribution line and the source's Content ID statement. A source whose API needs a key is
listed `MISSING` until its key is set and `KEY_SET` after; no source is called either way (fetching is not
implemented). A source with no API this skill knows of is `MANUAL`."""
from __future__ import annotations

import os
from typing import Any, Dict

FETCH_NOTE = ("not implemented: this skill downloads nothing; download the file and its licence yourself and run "
              "library-import with --licence and --licence-url / --licence-file")
SOURCES = (
    {"id": "freesound", "name": "Freesound", "kind": "sfx", "api_key_env": "FREESOUND_API_KEY",
     "licences": "per sound: CC0, CC BY, CC BY-NC (the sound's page states it)", "site": "https://freesound.org"},
    {"id": "jamendo", "name": "Jamendo", "kind": "bgm", "api_key_env": "JAMENDO_CLIENT_ID",
     "licences": "per track: Creative Commons variants, or a paid Jamendo Licensing licence", "site": "https://www.jamendo.com"},
    {"id": "youtube-audio-library", "name": "YouTube Audio Library", "kind": "bgm / sfx", "api_key_env": None,
     "licences": "per track: free to use in videos; some tracks require attribution (the library states the line)",
     "site": "https://studio.youtube.com"},
)


def report() -> Dict[str, Any]:
    out = {}
    for s in SOURCES:
        env = s["api_key_env"]
        status = "MANUAL" if env is None else ("KEY_SET" if os.environ.get(env) else "MISSING")
        out[s["id"]] = {"status": status, "name": s["name"], "kind": s["kind"], "needs": env, "licences": s["licences"],
                        "site": s["site"], "fetch": FETCH_NOTE}
    return out
