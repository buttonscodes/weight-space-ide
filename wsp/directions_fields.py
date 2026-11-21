"""The fields of the edit methods' parameters for the open workspace, for a form (GET /api/directions, ``parameters``).

Each field gives its kind, its default, its bounds and Heretic's search range for this model, so a client builds the
form from the server's own tables and keeps no copy of a default or a bound. Nothing here reads a weight: only the
model's structure (its layer count and which output components its layers have).

- The ablation's scalar defaults are those ``directions.normalize`` applies; its distribution fields have no API
  default (they are required), so their defaults are Heretic's draw at the middle of each range of
  ``directions_search.space``, transformed as Heretic transforms a draw (``to_parameters``): ``max_weight`` clamped
  at 0, ``min_weight`` the middle fraction (0.5) of ``max_weight``. Every default from a midpoint is rounded to four
  decimals; the ranges are ``space()``'s bounds, unrounded (Heretic's arithmetic).
- The ARA defaults are ``directions_ara.DEFAULTS``; the bounds are its checks'. ``neighbor_count``'s range is
  Heretic's with sets of the largest size (``min(15, good prompts, bad prompts)``): a measurement with fewer prompts
  narrows it.

Nothing imports this module but ``wsp.server``.
"""
from __future__ import annotations

from . import directions as DR
from . import directions_ara as ARA
from . import directions_search as DSR

DIGITS = 4                             # a default taken from the middle of a range is rounded to this


def _field(name, kind, default, *, nullable=False, null_label=None, min=None, max=None, min_exclusive=False,
           choices=None, applies_when=None, heretic_range=None, log=False) -> dict:
    return {"name": name, "kind": kind, "default": default, "nullable": nullable, "null_label": null_label,
            "min": min, "max": max, "min_exclusive": min_exclusive, "choices": choices, "applies_when": applies_when,
            "heretic_range": heretic_range, "log": log}


def _ranges(rows) -> dict:
    """space() rows -> {name: ([low, high], log)}."""
    out = {}
    for r in rows:
        a = r["distribution"]["attributes"]
        if "low" in a:
            out[r["name"]] = ([a["low"], a["high"]], bool(a.get("log")))
    return out


def _ablation(n, comps) -> dict:
    L = n - 1
    ranges = _ranges(DSR.space("ablation", n_layers=n, components=comps))
    draws = {"direction_scope": "per layer"}
    for key, ((lo, hi), _) in ranges.items():
        draws[key] = (lo + hi) / 2
    drawn = DSR.to_parameters("ablation", draws, DSR.MODIFIERS["Abliteration"]["settings"], direction="x",
                              components=comps, seed=0)
    wd = {c: {k: round(v, DIGITS) for k, v in d.items()} for c, d in drawn["weight_distributions"].items()}
    api = DR.normalize({"direction": "x", "weight_distributions": wd})     # the API's defaults for the rest
    fields = [
        _field("direction_index", "number", drawn["direction_index"], nullable=True, null_label="a direction per layer",
               min=-1, max=L, heretic_range=ranges["direction_index"][0]),
        _field("orthogonalize_direction", "boolean", api["orthogonalize_direction"]),
        _field("row_normalization", "choice", api["row_normalization"], choices=list(DR.ROW_NORMALIZATIONS)),
        _field("full_normalization_lora_rank", "integer", api["full_normalization_lora_rank"], min=1, max=DR.MAX_RANK,
               applies_when={"row_normalization": "full"}),
    ]
    components = []
    for c in comps:
        d = wd[c]
        components.append({"name": c, "fields": [
            _field("max_weight", "number", d["max_weight"], min=-DR.MAX_WEIGHT, max=DR.MAX_WEIGHT,
                   heretic_range=ranges[f"{c}.max_weight"][0]),
            _field("max_weight_position", "number", d["max_weight_position"], min=-n, max=2 * n,
                   heretic_range=ranges[f"{c}.max_weight_position"][0]),
            # Heretic draws a fraction of max_weight, not min_weight itself: no range of its own
            _field("min_weight", "number", d["min_weight"], min=-DR.MAX_WEIGHT, max=DR.MAX_WEIGHT),
            _field("min_weight_distance", "number", d["min_weight_distance"], min=0, min_exclusive=True,
                   heretic_range=ranges[f"{c}.min_weight_distance"][0]),
        ]})
    return {"fields": fields, "components": components}


def _ara(n) -> dict:
    L = n - 1
    D = ARA.DEFAULTS
    ranges = _ranges(DSR.space("ara", n_layers=n, n_good=DR.MAX_PROMPTS, n_bad=DR.MAX_PROMPTS))

    def drawn(name):
        return {"heretic_range": ranges[name][0], "log": ranges[name][1]}
    return {"fields": [
        _field("start_layer_index", "integer", D["start_layer_index"], min=0, max=L, **drawn("start_layer_index")),
        _field("end_layer_index", "integer", D["end_layer_index"], nullable=True, null_label="to the last layer",
               min=1, max=n, **drawn("end_layer_index")),
        _field("preserve_good_behavior_weight", "number", D["preserve_good_behavior_weight"], min=0,
               max=ARA.MAX_LOSS_WEIGHT, **drawn("preserve_good_behavior_weight")),
        _field("steer_bad_behavior_weight", "number", D["steer_bad_behavior_weight"], min=0, max=ARA.MAX_LOSS_WEIGHT,
               **drawn("steer_bad_behavior_weight")),
        _field("overcorrect_relative_weight", "number", D["overcorrect_relative_weight"], min=0,
               max=ARA.MAX_OVERCORRECT, **drawn("overcorrect_relative_weight")),
        _field("neighbor_count", "integer", D["neighbor_count"], min=1, max=ARA.MAX_NEIGHBORS,
               **drawn("neighbor_count")),
        _field("preserve_row_magnitudes", "boolean", D["preserve_row_magnitudes"]),
        _field("lora_rank", "integer", D["lora_rank"], min=1, max=ARA.MAX_LORA_RANK),
        _field("n_optimization_steps", "integer", D["n_optimization_steps"], min=1, max=ARA.MAX_STEPS),
        _field("learning_rate", "number", D["learning_rate"], min=0, max=ARA.MAX_LEARNING_RATE, min_exclusive=True),
        _field("max_iter", "integer", D["max_iter"], min=1, max=ARA.MAX_ITER),
        _field("history_size", "integer", D["history_size"], min=1, max=ARA.MAX_HISTORY),
        _field("seed", "integer", D["seed"], min=0, max=ARA.MAX_SEED),
    ]}


def parameter_fields(ws) -> dict | None:
    """The fields of both methods' parameters for the open workspace, for a form: kind, default, bounds and
    Heretic's search range. None when the model's layers or output components are not known."""
    s = getattr(ws, "struct", None)
    n = getattr(s, "n_layers", 0)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or len(getattr(s, "layers", None) or []) < n:
        return None
    try:
        comps = DSR._model_components(ws)
    except DR.DirectionError:                       # a component of an unexpected shape: not a form we can offer
        return None
    if not comps:
        return None
    return {"n_layers": n, "ablation": _ablation(n, list(comps)), "ara": _ara(n)}
