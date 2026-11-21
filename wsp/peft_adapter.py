"""A PEFT LoRA adapter directory read as PEFT 0.21.2 merges it, without importing ``peft`` (plan item H4b).

A directory with ``adapter_config.json`` and ``adapter_model.safetensors`` (Heretic's ``export_strategy = "adapter"``,
the IDE's own adapter export, wsp.adapter_export) changes each target module's weight as PEFT's merge does
(peft/tuners/lora/layer.py: ``Linear.merge`` with ``safe_merge`` false, ``get_delta_weight``):

    weight += transpose(lora_B @ lora_A, fan_in_fan_out) * scaling       scaling = alpha / r   (use_rslora: alpha / sqrt(r))

- ``r`` and ``alpha`` of a module: the first ``rank_pattern`` / ``alpha_pattern`` key k (in the file's order) with
  ``re.match(rf"(.*\\.)?({k})$", module)``, else ``r`` / ``lora_alpha`` (peft/utils/other.py get_pattern_key). The
  factors' rank must be that ``r`` (PEFT refuses to load them otherwise).
- ``fan_in_fan_out`` is what PEFT sets it to whatever the file says (tuners/lora/layer.py dispatch_default): false on
  an ``nn.Linear`` (a true one is reset with a warning), true on a ``Conv1D`` (GPT-2 style, weight stored [in, out]).
- Factors stored in bfloat16 or float16 are upcast to float32 first, as ``PeftModel.from_pretrained`` does by default
  (``autocast_adapter_dtype``); the product is then float32, as is the change it adds.
- A module is changed only when ``target_modules`` selects it (tuners_utils.check_target_module_exists: a list
  matches a module named as an entry or ending in ``.`` + an entry, a string is a full regular-expression match;
  ``exclude_modules`` and ``layers_to_transform`` / ``layers_pattern`` as PEFT applies them). A selected module the file
  has no factors for changes nothing (``init_lora_weights`` true or ``"gaussian"`` start ``lora_B`` at zero).

Only a plain LoRA is read; everything else is refused with what was found (``REFUSED``, ``peft_type``, ``bias``,
``init_lora_weights``, a file key that is not a factor). The module names are the loaded model's
(``base_model.model.`` + the module path); ``tensor_of_module`` maps one to the checkpoint tensor it changes.
"""
from __future__ import annotations

import json
import math
import os
import re

import torch

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"
PICKLE_FILE = "adapter_model.bin"
PREFIX = "base_model.model."
FACTORS = (".lora_A.weight", ".lora_B.weight")
FLOAT_DTYPES = ("F32", "BF16", "F16")
WHOLE_ELEMS = 16_000_000              # a module's product computed whole up to this many values (else by rows)

# Config keys that make an adapter something other than a plain LoRA when set (truthy), with what they are. The
# LoRA variants are the fields PEFT 0.21.2 marks is_lora_variant (tuners/lora/config.py, layer.py lora_variants).
REFUSED = (
    ("use_dora", "a DoRA adapter (a magnitude vector beside the factors)"),
    ("modules_to_save", "whole modules saved beside the factors"),
    ("lora_bias", "a bias trained on lora_B"),
    ("alora_invocation_tokens", "an activated LoRA (aLoRA)"),
    ("arrow_config", "an Arrow routing of several LoRAs"),
    ("use_bdlora", "a block-diagonal LoRA"),
    ("velora_config", "a VeLoRA adapter"),
    ("monteclora_config", "a MonteCLoRA adapter"),
    ("kasa_config", "a KaSA adapter"),
    ("use_qalora", "a quantization-aware LoRA (QALoRA)"),
    ("target_parameters", "LoRA on parameters rather than modules"),
    ("trainable_token_indices", "trainable tokens of an embedding"),
    ("layer_replication", "replicated layers"),
    ("megatron_config", "Megatron parallel layers"),
)


class AdapterError(ValueError):
    pass


def _json(v) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def read_config(directory) -> dict:
    """adapter_config.json, checked as a plain LoRA (no weight is read)."""
    p = os.path.join(directory, CONFIG_FILE)
    try:
        with open(p) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise AdapterError(f"the adapter's {CONFIG_FILE} cannot be read: {e}") from None
    if not isinstance(cfg, dict):
        raise AdapterError(f"the adapter's {CONFIG_FILE} cannot be read: it is not an object")
    kind = cfg.get("peft_type")
    if kind != "LORA":
        raise AdapterError(f"the adapter's peft_type is {_json(kind)}; only a LoRA adapter (peft_type \"LORA\") is read")
    for key, what in REFUSED:
        if cfg.get(key):
            raise AdapterError(f"the adapter's {key} is {_json(cfg[key])}: {what}, not a plain LoRA")
    if cfg.get("bias", "none") not in ("none", None):
        raise AdapterError(f"the adapter's bias is {_json(cfg['bias'])}: it trains bias terms, not a plain LoRA (bias "
                           f"\"none\")")
    init = cfg.get("init_lora_weights", True)
    if not (init is True or init == "gaussian"):
        raise AdapterError(f"the adapter's init_lora_weights is {_json(init)}: only true and \"gaussian\" are read (lora_B "
                           f"starts at zero, so a selected module the file lacks changes nothing)")
    r, alpha = cfg.get("r"), cfg.get("lora_alpha")
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise AdapterError(f"the adapter's r is {_json(r)}; it must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise AdapterError(f"the adapter's lora_alpha is {_json(alpha)}; it must be a number")
    for key, number in (("rank_pattern", "a positive integer"), ("alpha_pattern", "a number")):
        pat = cfg.get(key) or {}
        if not isinstance(pat, dict):
            raise AdapterError(f"the adapter's {key} is {_json(pat)}; it must be an object")
        for k, v in pat.items():
            bad = isinstance(v, bool) or not isinstance(v, (int, float)) or (key == "rank_pattern" and
                                                                              (not isinstance(v, int) or v <= 0))
            if bad or not math.isfinite(v):
                raise AdapterError(f"the adapter's {key} gives {k} {_json(v)}; it must be {number}")
    if not isinstance(cfg.get("use_rslora", False), bool):
        raise AdapterError(f"the adapter's use_rslora is {_json(cfg.get('use_rslora'))}; it must be true or false")
    tm = cfg.get("target_modules")
    if tm == "all-linear":
        raise AdapterError("the adapter's target_modules is \"all-linear\", which PEFT expands from the loaded model's "
                           "layers; give an adapter whose target_modules lists its modules")
    if not (isinstance(tm, str) or isinstance(tm, list) and tm and all(isinstance(t, str) for t in tm)):
        raise AdapterError(f"the adapter's target_modules is {_json(tm)}; it must be a list of module names or a "
                           f"regular expression")
    ex = cfg.get("exclude_modules")
    if ex is not None and not (isinstance(ex, str) or isinstance(ex, list) and all(isinstance(t, str) for t in ex)):
        raise AdapterError(f"the adapter's exclude_modules is {_json(ex)}; it must be a list of module names or a "
                           f"regular expression")
    lt = cfg.get("layers_to_transform")
    if lt is not None and not (isinstance(lt, int) and not isinstance(lt, bool) or
                               isinstance(lt, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in lt)):
        raise AdapterError(f"the adapter's layers_to_transform is {_json(lt)}; it must be a layer index or a list of them")
    lp = cfg.get("layers_pattern")
    if lp is not None and not (isinstance(lp, str) or isinstance(lp, list) and all(isinstance(t, str) for t in lp)):
        raise AdapterError(f"the adapter's layers_pattern is {_json(lp)}; it must be a name or a list of names")
    return cfg


# ------------------------------------------------------------------ PEFT's matching of module names
def selected(cfg: dict, module: str) -> bool:
    """Whether PEFT attaches the adapter to ``module`` (tuners_utils.check_target_module_exists)."""
    ex = cfg.get("exclude_modules")
    if ex:
        if isinstance(ex, str):
            if re.fullmatch(ex, module):
                return False
        elif module in ex or any(module.endswith(f".{k}") for k in ex):
            return False
    tm = cfg["target_modules"]
    if isinstance(tm, str):
        return re.fullmatch(tm, module) is not None
    if module in tm:
        return True
    found = any(module.endswith(f".{t}") for t in tm)
    lt, lp = cfg.get("layers_to_transform"), cfg.get("layers_pattern")
    if found and lt is not None and (len(lt) != 0 if isinstance(lt, list) else True):
        match = None
        if lp is None or len(lp) == 0:
            match = re.match(r".*?\.[^.]*\.(?P<idx>\d+)\.", module)
        else:
            for pattern in [lp] if isinstance(lp, str) else lp:
                match = re.match(rf"(?:^|.*?\.){pattern}\.(?P<idx>\d+)\.", module)
                if match is not None:
                    break
        if match is None:
            return False
        i = int(match.group("idx"))
        return i == lt if isinstance(lt, int) else i in lt
    return found


def pattern_value(patterns: dict, module: str, default):
    """PEFT's get_pattern_key: the value of the first key (in order) that matches the end of ``module``."""
    for key in patterns or {}:
        if re.match(rf"(.*\.)?({key})$", module):
            return patterns[key]
    return default


def scaling_of(cfg: dict, module: str) -> tuple[int, float, float]:
    """(r, alpha, scaling) of one module, as LoraLayer.update_layer sets them."""
    r = pattern_value(cfg.get("rank_pattern"), module, cfg["r"])
    alpha = pattern_value(cfg.get("alpha_pattern"), module, cfg["lora_alpha"])
    return r, alpha, (alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r)


# ------------------------------------------------------------------ module names: the loaded model's to the checkpoint's
def tensor_of_module(cfg, module, tensor_names):
    """A module name of the loaded model (PEFT's, Heretic's) -> the checkpoint tensor ``<module>.weight`` it changes:
    the exact name when the checkpoint has it, else the tensor whose module path maps to the same loaded-model path
    (wsp.runtime_layouts.module_path on both sides); None when there is none or more than one."""
    from . import runtime_layouts as RL
    if f"{module}.weight" in tensor_names:
        return f"{module}.weight"
    want = RL.module_path(cfg, module)
    hits = [t for t in tensor_names if t.endswith(".weight") and RL.module_path(cfg, t[:-len(".weight")]) == want]
    return hits[0] if len(hits) == 1 else None


# ------------------------------------------------------------------ the file
class Adapter:
    """The adapter's config and its factors' headers; ``bind`` maps its modules to a model's tensors and ``delta``
    reads a region of one tensor's change."""

    def __init__(self, directory):
        from safetensors import safe_open
        self.dir = directory
        self.config = read_config(directory)
        wp = os.path.join(directory, WEIGHTS_FILE)
        if not os.path.isfile(wp):
            if os.path.isfile(os.path.join(directory, PICKLE_FILE)):
                raise AdapterError(f"the adapter's weights are {PICKLE_FILE} (a pickle), which is not read; save the "
                                   f"adapter with safetensors ({WEIGHTS_FILE})")
            raise AdapterError(f"the adapter has no {WEIGHTS_FILE}")
        try:
            self.file = safe_open(wp, "pt")
            keys = list(self.file.keys())
        except Exception as e:
            raise AdapterError(f"the adapter's {WEIGHTS_FILE} cannot be read: {type(e).__name__}: {e}") from None
        modules: dict[str, dict] = {}
        for k in sorted(keys):
            suffix = next((s for s in FACTORS if k.endswith(s)), None)
            if not k.startswith(PREFIX) or suffix is None or len(k) <= len(PREFIX) + len(suffix):
                raise AdapterError(f"the adapter's {WEIGHTS_FILE} holds {k}, which is not a LoRA factor "
                                   f"({PREFIX}<module>.lora_A.weight or .lora_B.weight)")
            modules.setdefault(k[len(PREFIX):-len(suffix)], {})[suffix[1:7]] = k
        if not modules:
            raise AdapterError(f"the adapter's {WEIGHTS_FILE} holds no LoRA factors")
        self.modules = {}
        for m, ks in modules.items():
            if set(ks) != {"lora_A", "lora_B"}:
                lacks = "lora_B" if "lora_A" in ks else "lora_A"
                raise AdapterError(f"the adapter's module {m} has no {lacks} factor")
            if not selected(self.config, m):
                raise AdapterError(f"the adapter's file has factors for {m}, which its target_modules do not select: "
                                   f"PEFT would not load them")
            sa, sb = self.file.get_slice(ks["lora_A"]), self.file.get_slice(ks["lora_B"])
            da, db = str(sa.get_dtype()), str(sb.get_dtype())
            if da not in FLOAT_DTYPES or db not in FLOAT_DTYPES:
                raise AdapterError(f"the adapter's module {m} has factors stored as {da} and {db}; only F32, BF16 and F16 "
                                   f"factors are read")
            a, b = list(sa.get_shape()), list(sb.get_shape())
            r, alpha, scaling = scaling_of(self.config, m)
            if len(a) != 2 or len(b) != 2 or a[0] != b[1]:
                raise AdapterError(f"the adapter's module {m} has factors lora_A {_json(a)} and lora_B {_json(b)}, which "
                                   f"do not multiply as lora_B @ lora_A")
            if a[0] != r:
                raise AdapterError(f"the adapter's module {m} has factors of rank {a[0]}, but its config gives it r {r}")
            self.modules[m] = {"module": m, "keys": ks, "r": r, "alpha": alpha, "scaling": scaling, "out": b[0],
                               "in": a[1], "dtype": da if da == db else f"{da}/{db}"}

    def bind(self, cfg, names, shape, *, tied=lambda n: False, conv1d=lambda n: False) -> dict[str, dict]:
        """{tensor name: module entry with 'param' and 'transposed'} for a model with these tensor ``names``
        (``shape(name)``, ``cfg`` its config.json); a module that maps to no tensor, to a tied head, to the tensor of
        another module, or whose product does not have the tensor's shape is refused."""
        names = list(names)
        out = {}
        for m in sorted(self.modules):
            e = self.modules[m]
            t = tensor_of_module(cfg, m, names)
            if t is None:
                raise AdapterError(f"the adapter's target module {m} maps to no tensor of this model (no {m}.weight)")
            if tied(t):
                raise AdapterError(f"the adapter's target module {m} is the output head, which this model ties to the "
                                   f"token embedding")
            if t in out:
                raise AdapterError(f"the adapter's target modules {out[t]['module']} and {m} both map to {t}")
            transposed = bool(conv1d(t))
            want = [e["in"], e["out"]] if transposed else [e["out"], e["in"]]
            have = list(shape(t))
            if have != want:
                stored = " (a Conv1D weight, stored [in, out])" if transposed else ""
                raise AdapterError(f"the adapter's module {m} changes {t} by a [{want[0]}, {want[1]}] product{stored}, "
                                   f"but {t} has shape {_json(have)} in this model")
            out[t] = {**e, "param": t, "transposed": transposed}
        return out

    def factors(self, entry) -> tuple[torch.Tensor, torch.Tensor]:
        """(lora_B, lora_A) of one module as float32 (PeftModel.from_pretrained's autocast of bf16 / fp16 factors)."""
        ks = entry["keys"]
        return self.file.get_tensor(ks["lora_B"]).float(), self.file.get_tensor(ks["lora_A"]).float()

    def delta(self, entry, region=None) -> torch.Tensor:
        """The change of one tensor (a ``bind`` entry), float32, on ``region`` (a wsp.store Region of the checkpoint
        tensor; None for all of it): get_delta_weight's ``transpose(B @ A, fan_in_fan_out) * scaling``. A module up to
        WHOLE_ELEMS values is multiplied whole, as PEFT multiplies it, and then indexed; a larger one by its rows."""
        from .store import norm_index, region_index
        B, A = self.factors(entry)
        shape = (entry["in"], entry["out"]) if entry["transposed"] else (entry["out"], entry["in"])
        region = norm_index((), shape) if region is None else region
        s = entry["scaling"]
        if entry["out"] * entry["in"] <= WHOLE_ELEMS:
            P = B @ A
            P = P.T if entry["transposed"] else P
            return (P * s)[region_index(region)]
        ix = region_index(region)
        rows, cols = (ix[1], ix[0]) if entry["transposed"] else (ix[0], ix[1])
        b = B[rows] if not isinstance(rows, int) else B[rows:rows + 1]
        a = A[:, cols] if not isinstance(cols, int) else A[:, cols:cols + 1]
        P = b @ a
        P = P.T if entry["transposed"] else P
        P = P * s
        if isinstance(ix[0], int):
            P = P[0]
        if isinstance(ix[1], int):
            P = P[..., 0]
        return P
