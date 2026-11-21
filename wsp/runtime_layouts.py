"""How a checkpoint's tensor names become the names of the model `transformers` loads.

A workspace addresses tensors by their names in the checkpoint. The runner holds the model that
``AutoModelForCausalLM`` builds from it, and for some layouts those are not the same names:

    Qwen3.5 (and its MoE)    a multimodal checkpoint (``model.language_model.*``, ``model.visual.*``,
                             ``mtp.*``) is loaded as its text model: ``model.language_model.`` becomes
                             ``model.``, the vision tower and the multi-token prediction head are not loaded
    Qwen2/Qwen3/Qwen3.5 MoE  the checkpoint keeps one tensor per routed expert
                             (``mlp.experts.<e>.down_proj.weight``); transformers 5.x fuses them at load into
                             ``mlp.experts.down_proj [E, d, I]`` and ``mlp.experts.gate_up_proj [E, 2I, d]``

``runtime_name`` gives, for a checkpoint name, the loaded parameter's name and the index of the
checkpoint tensor inside it. A model type that is not listed here is the identity, so every other
layout keeps the runner's existing name resolution. Pure: no torch import.
"""
from __future__ import annotations

import math
import re

HYBRID_TEXT_TYPES = ("qwen3_5_text", "qwen3_5_moe_text")
EXPERT_RULES = (            # transformers' qwen2_moe WeightConverter, written out (conversion_mapping.py)
    (r"^(?P<m>.+\.mlp\.experts)\.(?P<e>\d+)\.down_proj\.weight$", "{m}.down_proj",    "whole"),   # [E, d, I][e]
    (r"^(?P<m>.+\.mlp\.experts)\.(?P<e>\d+)\.gate_proj\.weight$", "{m}.gate_up_proj", "first"),   # [E, 2I, d][e, :I]
    (r"^(?P<m>.+\.mlp\.experts)\.(?P<e>\d+)\.up_proj\.weight$",   "{m}.gate_up_proj", "second"),  # [E, 2I, d][e, I:]
)
LAYOUTS = {                 # text model type -> how checkpoint names become the loaded model's names
    "qwen3_5_text":     {"prefix": (("model.language_model.", "model."),),
                         "skip": (("model.visual.", "the vision tower"), ("mtp.", "the multi-token prediction head")),
                         "experts": ()},
    "qwen3_5_moe_text": {"prefix": (("model.language_model.", "model."),),
                         "skip": (("model.visual.", "the vision tower"), ("mtp.", "the multi-token prediction head")),
                         "experts": EXPERT_RULES},
    "qwen3_moe":        {"prefix": (), "skip": (), "experts": EXPERT_RULES},
    "qwen2_moe":        {"prefix": (), "skip": (), "experts": EXPERT_RULES},
}


def text_config(cfg: dict) -> dict:
    """The text model's config: ``cfg["text_config"]`` when it is a dict (a multimodal checkpoint), else ``cfg``."""
    sub = (cfg or {}).get("text_config")
    return sub if isinstance(sub, dict) else (cfg or {})


def text_model_type(cfg: dict) -> str:
    """The text model's ``model_type`` (``qwen3_5_text`` for a Qwen3.5 checkpoint), else the top-level one, else ``""``."""
    cfg = cfg or {}
    return text_config(cfg).get("model_type") or cfg.get("model_type") or ""


class Skipped(KeyError):
    """A checkpoint tensor that the text model the runner loads does not have; ``reason`` says why, ``part`` names
    the part it belongs to (``the vision tower``, ``the multi-token prediction head``)."""

    def __init__(self, reason: str, part: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.part = part

    def __str__(self) -> str:
        return self.reason


def _prefixed(layout: dict, name: str) -> str:
    for old, new in layout["prefix"]:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def runtime_name(cfg: dict, name: str, shape) -> tuple[str, tuple]:
    """Checkpoint tensor ``name`` of ``shape`` -> (the loaded model's parameter name, the index of the tensor in it).

    The first matching prefix rule is applied (once); a name under a skipped prefix (tested on the checkpoint
    name) raises ``Skipped``; a routed expert's tensor becomes its slice of the fused parameter: down ``(e,)``,
    gate ``(e, slice(0, I))``, up ``(e, slice(I, 2I))`` with ``I = shape[0]``. Otherwise the index is ``()``.
    A model type not in ``LAYOUTS`` returns ``(name, ())``."""
    layout = LAYOUTS.get(text_model_type(cfg))
    if layout is None:
        return name, ()
    runtime = _prefixed(layout, name)
    for prefix, reason in layout["skip"]:
        if name.startswith(prefix):
            raise Skipped(f"{name} belongs to {reason}, which the text model the IDE runs does not load", reason)
    for pattern, target, part in layout["experts"]:
        m = re.match(pattern, runtime)
        if m:
            e, rows = int(m.group("e")), int(shape[0])
            index = {"whole": (e,), "first": (e, slice(0, rows)), "second": (e, slice(rows, 2 * rows))}[part]
            return target.format(m=m.group("m")), index
    return runtime, ()


def module_path(cfg: dict, path: str) -> str:
    """A module path of the checkpoint's naming (a hook module) -> the loaded model's; the prefix rules only."""
    layout = LAYOUTS.get(text_model_type(cfg))
    return _prefixed(layout, path) if layout is not None and path else path


def loaded_params(ws) -> int:
    """The number of values in the checkpoint tensors the runner's model loads (skipped tensors left out)."""
    total = 0
    for name in ws.names():
        shape = tuple(ws.shape(name))
        try:
            runtime_name(ws.hf_config, name, shape)
        except Skipped:
            continue
        total += math.prod(shape)
    return total
