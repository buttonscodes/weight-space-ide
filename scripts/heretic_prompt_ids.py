"""The token ids Heretic generates from, for every prompt set of the config.toml in the current directory.

Run with Heretic's own interpreter, in a directory that holds a Heretic config.toml (model, model_commit, the
Abliteration modifier's good_prompts and bad_prompts, the KLDivergence and KeywordRate scorers' prompts and
response_prefix_test_prompts, each one dataset specification or a list of them):

    python heretic_prompt_ids.py <out.json>

Heretic's own classes load the prompts (load_prompts) and build the model inputs (Model.generate), so nothing
here is a reimplementation. No search is run and no weight is changed. It loads the model: run it under the
memory guard. The answer holds the specifications, the {system, user} prompts Heretic built and, per prompt,
the token ids it generated from; scripts/heretic_compare_prompt_ids.py compares the IDE's with them.

A ``response_prefix`` in config.toml is kept and appended to the good, bad, KL and refusal prompts, as Heretic
appends it; without one the prefix is empty (detection is never run). The response-prefix test prompts are always
rendered with an empty prefix, as Heretic renders them while it detects the prefix (its prefix is still unset then).
The answer also records whether Heretic found a system role in the chat template (``supports_system_role``).
"""
import json
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

out = Path(sys.argv[1])
sys.argv = sys.argv[:1]                      # Settings() parses the command line

from pydantic import TypeAdapter

from heretic.config import DatasetSpecification, Settings
from heretic.model import Model
from heretic.utils import load_prompts

settings = Settings()
if settings.response_prefix is None:          # no detection: an unset prefix is empty
    settings.response_prefix = ""
cfg = tomllib.loads(Path("config.toml").read_text())
raw = {
    "good": cfg["modifier"]["Abliteration"]["good_prompts"],
    "bad": cfg["modifier"]["Abliteration"]["bad_prompts"],
    "kl": cfg["scorer"]["KLDivergence"]["prompts"],
    "refusal": cfg["scorer"]["KeywordRate"]["prompts"],
    "prefix": cfg["response_prefix_test_prompts"],
}
adapter = TypeAdapter(DatasetSpecification)
prompts = {k: load_prompts(settings, adapter.validate_python(v)) for k, v in raw.items()}

model = Model(settings)
recorded = []
original = model.generate


def generate(batch, **kw):
    inputs, outputs = original(batch, **kw)
    ids, mask = inputs["input_ids"].cpu(), inputs["attention_mask"].cpu()
    for row, m in zip(ids, mask):
        recorded.append([int(t) for t, keep in zip(row.tolist(), m.tolist()) if keep])
    return inputs, outputs


model.generate = generate


def ids_of(fn, *a, **kw):
    recorded.clear()
    fn(*a, **kw)
    return [list(x) for x in recorded]


ids = {}
for k in ("good", "bad"):
    ids[k] = ids_of(model.get_residuals_mean, prompts[k], winsorization_quantile=1.0)
for k in ("kl", "refusal"):
    ids[k] = ids_of(model.get_logits_batched, prompts[k])
prefix, settings.response_prefix = settings.response_prefix, ""     # the test prompts render before a prefix exists
ids["prefix"] = ids_of(model.get_logits_batched, prompts["prefix"])
settings.response_prefix = prefix
assert all(len(ids[k]) == len(prompts[k]) for k in prompts), {k: (len(ids[k]), len(prompts[k])) for k in prompts}
out.write_text(json.dumps({
    "model": settings.model, "model_commit": settings.model_commit, "system_prompt": settings.system_prompt,
    "response_prefix": settings.response_prefix, "batch_size": settings.batch_size,
    "supports_system_role": model.supports_system_role,
    "specifications": raw,
    "prompts": {k: [{"system": p.system, "user": p.user} for p in v] for k, v in prompts.items()},
    "ids": ids,
    "versions": {p: version(p) for p in ("heretic-llm", "torch", "transformers", "tokenizers", "datasets", "pyarrow")},
}, indent=1))
print("written", out, {k: len(v) for k, v in ids.items()})
