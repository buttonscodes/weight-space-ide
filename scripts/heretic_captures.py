"""Heretic's module captures in place of the IDE's: the "IDE from Heretic's captures" code path of the ARA comparison.

Shared by scripts/heretic_compare_ara.py (its stages 3 and 4b) and scripts/heretic_compare_adapter.py (its
ide-captures stage), so that both run the same fit on the same inputs. Not a script.

    rows, heretic_io = heretic_io_of(ws, name, info, tens, n, d, modules=None)
    use_heretic_io(ws, name, info, heretic_io, n, d)

``info`` is DR.load_io(ws, name, meta) of a measurement made with module_io; ``tens`` the tensors of Heretic's
dump (dump/dump.safetensors, keys io.<good|bad>.L<layer>.<component>.<module index>.<x|y>). The first compares
the two tools' captures per layer, component and module and keeps Heretic's where both captured the module; the
second writes those into the measurement's saved arrays, so that the next ablate_ara call on the measurement fits
from Heretic's captures.

Modules are matched by Heretic's module names when ``modules`` (dump.json["modules"], heretic_dump.py) is given:
each IDE module (its path in the layer, from io.json version 2, or from the component's matrix for a version-1
measurement) takes the capture of the Heretic module whose name, through wsp.runtime_layouts.module_path, is the same
path. Without it, as before: a component Heretic captured as exactly one module takes that module (index 0).
``heretic_io`` is keyed (set, layer, component, module path or None).
"""
import math
import os

import numpy as np

from wsp import directions as DR
from wsp import runtime_layouts as RL

import formats_shared as FS


def rel(a, b):
    d = float((a.double() - b.double()).norm())
    n = float(b.double().norm())
    return d / n if n > 0 else (0.0 if d == 0 else math.inf)


def _ide_path(ws, layer, comp, path):
    """The module path in the layer of an io.json entry; for a version-1 entry, the component matrix's tensor name."""
    if path:
        return path
    views = DR.component_views(ws, layer, comp)
    if not views:
        return None
    name = getattr(views[0], "name", None)
    if not name:
        return None
    name = name[:-len(".weight")] if name.endswith(".weight") else name
    return FS.layer_relative(RL.module_path(ws.hf_config, name), layer)


def heretic_paths(cfg, modules, layer, comp):
    """Heretic's module index -> its path in the layer (through module_path), for one layer and component."""
    names = (modules[layer] if layer < len(modules) else {}).get(comp) or []
    return {i: (FS.layer_relative(RL.module_path(cfg, nm), layer) if nm else None) for i, nm in enumerate(names)}


def heretic_io_of(ws, name, info, tens, n, d, modules=None):
    """-> (a row per layer, component and IDE module: what each tool captured and how far apart; {(set, layer,
    component, module path): (Heretic's x, Heretic's y)} for every module both captured)."""
    io_rows, heretic_io = [], {}
    for l in range(n):
        arrays = dict(zip(("good", "bad"), DR.io_arrays(ws, name, info, l)))
        for comp, path, c in FS.io_entries(info["layers"][l]):
            keys = [k for k in tens if k.startswith(f"io.good.L{l}.{comp}.") and k.endswith(".x")]
            row = {"layer": l, "component": comp, "ide_captured": bool(c.get("captured")), "heretic_modules": len(keys)}
            index = None
            if modules is None:
                index = 0 if row["heretic_modules"] == 1 else None
            else:
                p = _ide_path(ws, l, comp, path)
                row["module"] = p
                hits = [i for i, hp in heretic_paths(ws.hf_config, modules, l, comp).items() if hp == p]
                index = hits[0] if len(hits) == 1 and f"io.good.L{l}.{comp}.{hits[0]}.x" in tens else None
                row["heretic_index"] = index
            if c.get("captured") and index is not None:
                for s in ("good", "bad"):
                    hx, hy = tens[f"io.{s}.L{l}.{comp}.{index}.x"], tens[f"io.{s}.L{l}.{comp}.{index}.y"]
                    x = arrays[s][:, c["x"]:c["x"] + c["d_in"]]
                    y = arrays[s][:, c["y"]:c["y"] + d]
                    assert tuple(hx.shape) == tuple(x.shape) and tuple(hy.shape) == tuple(y.shape), (hx.shape, x.shape)
                    row[f"{s}_x_rel"], row[f"{s}_y_rel"] = rel(x, hx), rel(y, hy)
                    heretic_io[(s, l, comp, path)] = (hx, hy)
            io_rows.append(row)
    return io_rows, heretic_io


def ide_io_of(ws, name, info, n, d):
    """The IDE's own captures, {(set, layer, component, module path): (x, y)} for every captured module (copies,
    so they survive use_heretic_io)."""
    out = {}
    for l in range(n):
        arrays = dict(zip(("good", "bad"), DR.io_arrays(ws, name, info, l)))
        for comp, path, c in FS.io_entries(info["layers"][l]):
            if not c.get("captured"):
                continue
            for s in ("good", "bad"):
                out[(s, l, comp, path)] = (arrays[s][:, c["x"]:c["x"] + c["d_in"]].clone(),
                                           arrays[s][:, c["y"]:c["y"] + d].clone())
    return out


def use_heretic_io(ws, name, info, heretic_io, n, d):
    """Write Heretic's captures into the measurement's saved module arrays (in place)."""
    folder = DR._paths(ws, name)["io"]
    for l in range(n):
        L = info["layers"][l]
        if not L.get("files"):
            continue
        for s in ("good", "bad"):
            arr = np.load(os.path.join(folder, L["files"][s]), allow_pickle=False).copy()
            for comp, path, c in FS.io_entries(L):
                if (s, l, comp, path) in heretic_io:
                    hx, hy = heretic_io[(s, l, comp, path)]
                    arr[:, c["x"]:c["x"] + c["d_in"]] = hx.numpy()
                    arr[:, c["y"]:c["y"] + d] = hy.numpy()
            np.save(os.path.join(folder, L["files"][s]), arr.astype("<f4"), allow_pickle=False)
