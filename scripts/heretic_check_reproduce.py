"""Check that Heretic reads the reproduce.json files the IDE writes (GET /api/patches/{pid}/record/heretic), with
Heretic's own code, offline and without a model.

Run with Heretic's own interpreter from an empty scratch directory (Settings() reads a config.toml in the current
directory when there is one):

    python heretic_check_reproduce.py <out.json> <IDE reproduce.json> ...

For each file, the steps heretic --reproduce takes before it loads a model (heretic/main.py, heretic/reproduce.py):

    S-D1  load_reproduction_information(path) succeeds
    S-D2  info["version"] == "4"
    S-D3  Settings.model_validate_init_only(info["settings"], Settings()) succeeds (the command line set to a dummy
          model, so that Settings() needs no config.toml)
    S-D4  load_plugin(name=info["settings"]["modifiers"][0]["plugin"], base_class=Modifier) succeeds
    S-D5  that class's parameters class (Modifier.parameters_class, read from the class's generic base as the
          property does) .from_dict(info["parameters"]) succeeds
    S-D6  format_version_information(info["environment"]["heretic"]) succeeds; its text is printed and recorded
          (assumption A7: a pip install from git records the same URL; a mismatch only adds Heretic's "critical" line)

Also reported, not asserted: the modifier's own settings validated by the plugin (get_settings_raw and
validate_settings, as Heretic builds a plugin), and the settings Heretic would take from the file.
Prints a PASS or FAIL line per assertion; writes <out.json> with every result and a pass flag per assertion; exits 1
when an assertion fails. Imports nothing from wsp.
"""
import json
import sys
import traceback
from pathlib import Path
from typing import get_args

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
out, files = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
sys.argv = [sys.argv[0], "--model", "dummy/model"]           # Settings() parses the command line

from heretic.config import Settings
from heretic.modifier import Modifier
from heretic.plugin import load_plugin
from heretic.reproduce import format_version_information, load_reproduction_information

current = Settings()
result = {"files": {}}


def step(entry, name, fn):
    """Run one assertion: -> its value, or None when it raised (recorded as a failure with the error)."""
    try:
        value = fn()
        entry[name] = {"pass": True}
        print(f"PASS {entry['_label']} {name}", flush=True)
        return value
    except Exception as e:
        entry[name] = {"pass": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        print(f"FAIL {entry['_label']} {name}: {type(e).__name__}: {e}", flush=True)
        return None


for path in files:
    entry = {"_label": path.name}
    info = step(entry, "S-D1 load_reproduction_information", lambda: load_reproduction_information(str(path)))
    if info is not None:
        ok = info.get("version") == "4"
        entry["S-D2 version 4"] = {"pass": ok, "version": info.get("version")}
        print(f"{'PASS' if ok else 'FAIL'} {path.name} S-D2 version 4 (version {info.get('version')!r})", flush=True)
        settings = step(entry, "S-D3 Settings.model_validate_init_only",
                        lambda: Settings.model_validate_init_only(info["settings"], current))
        cls = step(entry, "S-D4 load_plugin",
                   lambda: load_plugin(name=info["settings"]["modifiers"][0]["plugin"], base_class=Modifier))
        if cls is not None:
            params = step(entry, "S-D5 parameters_class.from_dict",
                          lambda: get_args(cls.__orig_bases__[0])[0].from_dict(info["parameters"]))
            if params is not None:
                entry["parameters"] = repr(params)
            if settings is not None:
                try:
                    raw = cls.get_settings_raw(settings.model_extra, "modifier",
                                               info["settings"]["modifiers"][0].get("instance_name"))
                    entry["reported_modifier_settings"] = {"ok": True, "value": cls.validate_settings(raw).model_dump(
                        mode="json")}
                except Exception as e:
                    entry["reported_modifier_settings"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                entry["reported_settings"] = settings.model_dump(mode="json")
        text = step(entry, "S-D6 format_version_information",
                    lambda: format_version_information(info["environment"]["heretic"]))
        if text is not None:
            entry["version_information"] = text
            print(f"     {path.name} Heretic version information: {text}", flush=True)
    entry["pass"] = all(v["pass"] for k, v in entry.items() if k.startswith("S-D"))
    result["files"][str(path)] = {k: v for k, v in entry.items() if k != "_label"}

result["pass"] = bool(result["files"]) and all(e["pass"] for e in result["files"].values())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1, default=str))
print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
sys.exit(0 if result["pass"] else 1)
