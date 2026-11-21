"""Dump what Heretic computed in a finished run, for comparison with the IDE.

Run with Heretic's own interpreter, with the run directory (config.toml, checkpoints/, model/) as the
current directory:   python heretic_dump.py <out directory> [<merged model directory>]

The merged model directory defaults to model/ (a run exported with export_strategy "merge"). A run exported as an
adapter (export_strategy "adapter") has none: give the directory scripts/formats_shared.py merge wrote from the
run's adapter (PEFT's merge, materialised), for example ../heretic-merged.

Writes <out>/dump.json (prompts, token ids as Heretic tokenized them, the response prefix, every trial's
parameters and scores, which trial was exported) and <out>/dump.safetensors (the mean residuals of the
good and bad prompts, [layers + 1, d], and the base model's first-token logits on the KL prompts).
It also writes <out>/scores.json and <out>/scores.safetensors for the scores comparison
(scripts/heretic_compare_scores.py): the dummy prompt of Heretic's response-prefix detection, the
prefix-test responses without and with the run's prefix, and on the base model and then on the run's
merged model (model/) the responses, keyword flags and scores of Heretic's own KeywordRate,
ResponseLength (on the KL prompts) and KLDivergence scorers, with every input and response token id.
For a run of the fitted low-rank ablation (a [modifier.ARA] section) it also writes the last-token input
and output of every module Heretic fits, as Heretic captures them: tensors
io.<good|bad>.L<layer>.<component>.<module index>.<x|y>, [prompts, width].
dump.json also records, per layer, the full name of every module get_layer_modules lists, by component and module
index (`modules`, [layers] of {component: [name, ...]}: the names apply_lora targets, from named_modules), and the
model's dtype (`dtype`). Dumps written before `modules` existed stay readable: the comparisons then match modules by
index.
Uses Heretic's own classes for all of it, so nothing here is a reimplementation.
"""
import json
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0 if len(sys.argv) >= 2 else 2)
out = Path(sys.argv[1])
merged_dir = sys.argv[2] if len(sys.argv) > 2 else "model"
if not Path(merged_dir).is_dir():
    sys.exit(f"{merged_dir}: no merged model directory; for an adapter export, materialise it first with "
             f"scripts/formats_shared.py merge and give its directory as the second argument")
sys.argv = sys.argv[:1]                      # Settings() parses the command line

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from safetensors.torch import save_file

from heretic.config import Settings, SingleDatasetSpecification
from heretic.model import Model
from heretic.utils import load_prompts

settings = Settings()
cfg = tomllib.loads(Path("config.toml").read_text())

ck = next(Path(settings.study_checkpoint_dir).glob("*.jsonl"))
storage = JournalStorage(JournalFileBackend(str(ck), lock_obj=JournalFileOpenLock(str(ck))))
study = optuna.load_study(study_name="heretic", storage=storage)
saved = json.loads(study.user_attrs["settings"])
settings.response_prefix = saved.get("response_prefix")

complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
names = [s["name"] for s in complete[0].user_attrs["scores"]]            # both scorers are objectives here


def key(t):
    return tuple(next((s["score"]["value"] for s in t.user_attrs["scores"] if s["name"] == n), None) for n in names)


front = sorted(study.best_trials, key=key)                               # as Heretic sorts the Pareto front
exported = front[settings.trial_index]


def spec(d):
    return SingleDatasetSpecification(**d)


modifier = "ARA" if "ARA" in cfg["modifier"] else "Abliteration"
sets = {
    "good": spec(cfg["modifier"][modifier]["good_prompts"]),
    "bad": spec(cfg["modifier"][modifier]["bad_prompts"]),
    "kl": spec(cfg["scorer"]["KLDivergence"]["prompts"]),
    "refusal": spec(cfg["scorer"]["KeywordRate"]["prompts"]),
}
prompts = {k: load_prompts(settings, v) for k, v in sets.items()}

model = Model(settings)


def module_names():
    """Per layer {component: [full module name per module index]}, as apply_lora maps modules to names."""
    root = model.model
    if hasattr(root, "base_model") and hasattr(root.base_model, "model"):   # a PeftModel: its wrapped model's names
        root = root.base_model.model
    by_id = {id(m): name for name, m in root.named_modules()}
    return [{component: [by_id.get(id(m)) for m in modules]
             for component, modules in model.get_layer_modules(layer).items()}
            for layer in range(len(model.get_layers()))]


modules_by_layer = module_names()
recorded = []
responded = []                               # each row's newly generated token ids
original = model.generate


def generate(batch, **kw):
    inputs, outputs = original(batch, **kw)
    ids, mask = inputs["input_ids"].cpu(), inputs["attention_mask"].cpu()
    for row, m in zip(ids, mask):
        recorded.append([int(t) for t, keep in zip(row.tolist(), m.tolist()) if keep])
    seq = outputs.sequences if hasattr(outputs, "sequences") else outputs
    for row in range(seq.shape[0]):
        responded.append([int(t) for t in seq[row, inputs["input_ids"].shape[1]:].tolist()])
    return inputs, outputs


model.generate = generate


def ids_of(fn, *a, **kw):
    recorded.clear()
    value = fn(*a, **kw)
    return value, [list(x) for x in recorded]


def with_responses(fn, *a, **kw):
    """-> (value, input ids, response ids) of the generations ``fn`` makes."""
    recorded.clear()
    responded.clear()
    value = fn(*a, **kw)
    return value, [list(x) for x in recorded], [list(x) for x in responded]


good_mean, good_ids = ids_of(model.get_residuals_mean, prompts["good"], winsorization_quantile=1.0)
bad_mean, bad_ids = ids_of(model.get_residuals_mean, prompts["bad"], winsorization_quantile=1.0)
kl_logits, kl_ids = ids_of(model.get_logits_batched, prompts["kl"])

tensors = {"good_mean": good_mean.float().contiguous(), "bad_mean": bad_mean.float().contiguous(),
           "kl_logits": kl_logits.float().cpu().contiguous()}
io_ids = {}
if modifier == "ARA":
    for label in ("good", "bad"):
        module_io, io_ids[label] = ids_of(model.get_module_io_batched, prompts[label])
        for layer, components in enumerate(module_io):
            for component, modules in components.items():
                for index, (x, y) in modules.items():
                    tensors[f"io.{label}.L{layer}.{component}.{index}.x"] = x.float().contiguous()
                    tensors[f"io.{label}.L{layer}.{component}.{index}.y"] = y.float().contiguous()

# ------------------------------------------------------------------ scores (scripts/heretic_compare_scores.py)
from heretic.plugin import Context
from heretic.scorers.keyword_rate import KeywordRate
from heretic.scorers.kl_divergence import KLDivergence
from heretic.scorers.response_length import ResponseLength

run_prefix = settings.response_prefix
dummy_messages = ([{"role": "system", "content": "You are a helpful assistant."}] if model.supports_system_role else []) \
    + [{"role": "user", "content": "This is a dummy prompt."}]
dummy = model.tokenizer.apply_chat_template(dummy_messages, add_generation_prompt=True, tokenize=False)


def responses_record(prompt_list):
    texts, inputs, responses = with_responses(model.get_responses_batched, prompt_list)
    return {"texts": texts, "input_ids": inputs, "response_ids": responses}


prefix_prompts = load_prompts(settings, settings.response_prefix_test_prompts)
settings.response_prefix = None
prefix_test = {"prompts": [{"system": p.system, "user": p.user} for p in prefix_prompts],
               "without_prefix": responses_record(prefix_prompts), "with_prefix": None}
if run_prefix:
    settings.response_prefix = run_prefix
    prefix_test["with_prefix"] = responses_record(prefix_prompts)
settings.response_prefix = run_prefix


def scorer(cls, overrides=None):
    # As heretic.evaluator.Evaluator._load_and_init_scorers builds each scorer (no instance name).
    raw = cls.get_settings_raw(settings.model_extra, "scorer", None)
    if overrides:
        raw = {**raw, **overrides}
    return cls(heretic_settings=settings, settings=cls.validate_settings(raw))


def as_dict(score):
    return {"value": float(score.value), "rich_display": score.rich_display, "md_display": score.md_display}


def scored(ctx, scorers):
    out = {}
    for label, s in scorers.items():
        value, inputs, responses = with_responses(s.get_score, ctx)
        entry = {"score": as_dict(value), "input_ids": inputs, "response_ids": responses}
        if label != "kl":
            texts = ctx.get_responses(s.prompts)          # the context's cached texts of this scorer's prompts
            entry["texts"] = texts
            if label == "refusal":
                entry["flags"] = [bool(s._is_match(t)) for t in texts]
        out[label] = entry
    return out


ctx = Context(settings, model)
scorers = {"refusal": scorer(KeywordRate),
           "length": scorer(ResponseLength, {"prompts": cfg["scorer"]["KLDivergence"]["prompts"]}),
           "kl": scorer(KLDivergence)}
for s in scorers.values():
    s.init(ctx)                                           # KLDivergence.init computes its baseline on the base model
base_scores = scored(ctx, scorers)
base_scores["kl"]["baseline"] = as_dict(scorers["kl"].get_baseline_score(ctx))
base_model = settings.model
settings.model = merged_dir                               # heretic.main's evaluate_model path
model.reset_model()
merged_scores = scored(Context(settings, model), scorers)  # the KL baseline stays the base model's
merged_kl_logits = model.get_logits_batched(prompts["kl"])
settings.model = base_model                               # dump.json names the base model, as before
generation = {"generation_config": model.model.generation_config.to_dict(),
              "clean_up_tokenization_spaces": getattr(model.tokenizer, "clean_up_tokenization_spaces", None),
              "attn_implementation": getattr(model.model.config, "_attn_implementation", None)}

out.mkdir(parents=True, exist_ok=True)
save_file({"merged_kl_logits": merged_kl_logits.float().cpu().contiguous()}, str(out / "scores.safetensors"))
(out / "scores.json").write_text(json.dumps({
    "response_prefix": run_prefix, "system_prompt": settings.system_prompt,
    "max_response_length": settings.max_response_length, "batch_size": settings.batch_size,
    "chain_of_thought_blocks": [list(b) for b in settings.chain_of_thought_blocks],
    "dummy": {"rendering": dummy, "system_role": bool(model.supports_system_role)},
    "prefix_test": prefix_test, "base": base_scores, "merged": merged_scores, **generation,
    "versions": {p: version(p) for p in ("heretic-llm", "torch", "transformers", "tokenizers")},
}, indent=1, default=str))

save_file(tensors, str(out / "dump.safetensors"))
(out / "dump.json").write_text(json.dumps({
    "model": settings.model, "model_commit": settings.model_commit, "dtype": str(model.dtype),
    "device": str(model.model.device), "layers": len(model.get_layers()),
    "components": model.get_modifiable_components(), "modules": modules_by_layer,
    "system_prompt": settings.system_prompt, "response_prefix": settings.response_prefix,
    "batch_size": settings.batch_size, "seed": saved.get("seed", settings.seed), "modifier": modifier,
    "modifier_settings": saved.get("modifier", cfg.get("modifier")),
    "prompts": {k: [{"system": p.system, "user": p.user} for p in v] for k, v in prompts.items()},
    "ids": {"good": good_ids, "bad": bad_ids, "kl": kl_ids}, "io_ids": io_ids,
    "objective_names": names,
    "trials": [{"number": t.number, "index": t.user_attrs.get("index"), "parameters": t.user_attrs["parameters"],
                "scores": t.user_attrs["scores"]} for t in complete],
    "exported_trial_index": exported.user_attrs.get("index"),
    "versions": {p: version(p) for p in ("heretic-llm", "torch", "transformers", "peft", "optuna", "tokenizers")},
}, indent=1))
print("dumped", out, "exported trial", exported.user_attrs.get("index"), "prefix", repr(settings.response_prefix))
