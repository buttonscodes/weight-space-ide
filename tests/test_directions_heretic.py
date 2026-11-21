"""Running a Heretic reproduce.json natively (wsp.directions_heretic and POST /api/directions/reproduce), model-free.

The workspace is the tiny random chat model of tests/test_directions_datasets.py (a stub chat template that writes
the system message before the user turn); the Hub datasets are fake cache snapshots written here; nothing reaches
the network. Where the expectations come from: the Heretic files are typed here in the shape Heretic 64e9991 writes
them (heretic/utils.py generate_reproduce_json; the field names typed from its config.py and modifiers); the
expected patch is made by the existing code paths with the rows and parameters typed here (DR.measure of the texts,
a weight script typed by hand, Session.apply) and compared piece by piece with torch.equal; the reasons and notes
are typed; Heretic's severities are typed from heretic/reproduce.py; hashes are computed with hashlib.
"""
import hashlib
import json
import platform

import pytest
import torch

from wsp import directions as DR
from wsp import directions_datasets as DD
from wsp import directions_heretic as DH
from wsp import directions_record as REC
from wsp import directions_search as DSR

from test_directions_datasets import J, chat_ws, hub_card, session_on, snapshot, use_template
from test_directions_reproduce import client, same_pieces

GOOD_REPO, BAD_REPO = "org/good", "org/bad"
C_GOOD, C_BAD = "a1" * 20, "b2" * 20
GOOD_ROWS, BAD_ROWS = ["ab", "cd", "ef"], ["gh", "ij", "kl"]
MODEL, MODEL_SHA = "org/model", "c3" * 20
SYSTEM = "s"
WD = {"attn.o_proj": {"max_weight": 1.0, "max_weight_position": 0.0, "min_weight": 0.5, "min_weight_distance": 2.0},
      "mlp.down_proj": {"max_weight": 0.8, "max_weight_position": 1.0, "min_weight": 0.2, "min_weight_distance": 1.0}}
ARA_DRAWN = {"start_layer_index": 0, "end_layer_index": 2, "preserve_good_behavior_weight": 0.5,
             "steer_bad_behavior_weight": 0.01, "overcorrect_relative_weight": 0.25, "neighbor_count": 1}
ABLATION_SCRIPT = ('ablate_direction("{name}", direction_index=0.8, weight_distributions={{"attn.o_proj": '
                   '{{"max_weight": 1.0, "max_weight_position": 0.0, "min_weight": 0.5, "min_weight_distance": 2.0}}, '
                   '"mlp.down_proj": {{"max_weight": 0.8, "max_weight_position": 1.0, "min_weight": 0.2, '
                   '"min_weight_distance": 1.0}}}}, orthogonalize_direction=True, row_normalization="{mode}", '
                   'full_normalization_lora_rank=3)')
ARA_SCRIPT = ('ablate_ara("{name}", start_layer_index=0, end_layer_index=2, preserve_good_behavior_weight=0.5, '
              'steer_bad_behavior_weight=0.01, overcorrect_relative_weight=0.25, neighbor_count=1, '
              'preserve_row_magnitudes=True, lora_rank=2, n_optimization_steps=1, learning_rate=1.0, max_iter=2, '
              'history_size=10, seed=7)')


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """The Hugging Face cache is tmp_path/hub with the two datasets; a fetch or a listing fails the test."""
    cache = tmp_path / "hub"
    monkeypatch.setattr(DD, "hub_cache_dir", lambda: str(cache))

    def refuse(*a, **k):
        raise AssertionError("network: DD.hub_info / DD.hub_fetch")
    monkeypatch.setattr(DD, "hub_info", refuse)
    monkeypatch.setattr(DD, "hub_fetch", refuse)
    for repo, commit, rows in ((GOOD_REPO, C_GOOD, GOOD_ROWS), (BAD_REPO, C_BAD, BAD_ROWS)):
        snapshot(cache, repo, commit, {"README.md": hub_card(train=3, test=1),
                                       "data/train-00000-of-00001.jsonl": J(rows).encode(),
                                       "data/test-00000-of-00001.jsonl": J(["zz"]).encode()})
    return cache


@pytest.fixture
def sess(tmp_path, monkeypatch, hub):
    use_template(monkeypatch)
    return session_on(tmp_path, chat_ws)


def spec(repo, commit, split="train[:3]"):
    return {"dataset": repo, "commit": commit, "config": None, "split": split, "column": "text", "prefix": "",
            "suffix": "", "system_prompt": None}


def heretic_file(method="ablation", mode="none"):
    """A reproduce.json as Heretic 64e9991 writes it (generate_reproduce_json, without system information)."""
    if method == "ablation":
        plugin, cls, params = "heretic.modifiers.abliteration.Abliteration", "Abliteration", \
            {"direction_index": 0.8, "weight_distributions": json.loads(json.dumps(WD))}
        table = {"orthogonalize_direction": True, "row_normalization": mode, "full_normalization_lora_rank": 3,
                 "winsorization_quantile": 1.0}
    else:
        plugin, cls, params = "heretic.modifiers.ara.ARA", "ARA", dict(ARA_DRAWN)
        table = {"preserve_row_magnitudes": True, "lora_rank": 2, "n_optimization_steps": 1, "learning_rate": 1.0,
                 "max_iter": 2, "history_size": 10, "print_loss": True}
    settings = {
        "model": MODEL, "model_commit": MODEL_SHA, "dtypes": ["float32"], "quantization": "none", "device_map": "cpu",
        "max_memory": None, "offload_outputs_to_cpu": True, "batch_size": 1, "max_response_length": 10,
        "response_prefix": "",
        "scorers": [{"plugin": "heretic.scorers.keyword_rate.KeywordRate", "optimization": "minimize", "instance_name": None},
                    {"plugin": "heretic.scorers.kl_divergence.KLDivergence", "optimization": "minimize",
                     "instance_name": None}],
        "modifiers": [{"plugin": plugin, "instance_name": None}], "n_trials": 2, "n_startup_trials": 1, "seed": 7,
        "modified_model_type": "decensored", "max_shard_size": "5GB", "export_strategy": "adapter",
        "checkpoint_action": "restart", "trial_index": 0, "n_additional_trials": None, "model_action": "save",
        "upload_repo_private": None, "upload_reproducibility_information": None,
        "upload_tags": ["uncensored", "decensored", "abliterated"], "ignore_mismatches": None, "system_prompt": SYSTEM,
        "scorer": {"KLDivergence": {"prompts": spec(GOOD_REPO, C_GOOD, "test[:1]")}},
        "modifier": {cls: {"good_prompts": spec(GOOD_REPO, C_GOOD), "bad_prompts": spec(BAD_REPO, C_BAD), **table}}}
    return {"version": "4", "timestamp": "2026-11-19T12:00:00",
            "environment": {"heretic": {"version": "2.2.0.dev0", "is_standard_pypi": False, "metadata": {"type": "local"}},
                            "pytorch_version": "0.0.1",
                            "requirements": {"transformers": "0.0.2", "peft": "0.0.3", "tokenizers": "0.0.4",
                                             "zz-not-installed": "1.0"}},
            "settings": settings, "parameters": params,
            "scores": [{"name": "Refusals", "score": {"value": 0.0, "rich_display": "0/1", "md_display": "0/1"},
                        "baseline": None}],
            "hashes": {"adapter_model.safetensors": "de" * 32}}


def default_name(data):
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "heretic-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def committed_reference(sess, script, module_io=False):
    """The same edit made by hand: the dataset rows measured as texts, the script typed, Session.apply."""
    ws = sess.ws
    DR.measure(ws, "ref", GOOD_ROWS, BAD_ROWS, chat=True, system=SYSTEM, module_io=module_io)
    r = sess.apply(script.format(name="ref"))
    assert r["committed"], r
    return r["patch"]


def reason(ws, data, **kw):
    with pytest.raises(REC.RecordError) as e:
        DH.check_heretic_reproduce(ws, data, **kw)
    prefix = "this reproduce.json cannot be run in the IDE: "
    assert str(e.value).startswith(prefix), str(e.value)
    return str(e.value)[len(prefix):]


# ------------------------------------------------------------------ HR1 the ablation from the file is the hand-made one
@pytest.mark.parametrize("mode", ["none", "full"])
def test_an_ablation_file_makes_the_hand_made_patch(sess, mode):
    ws = sess.ws
    data = heretic_file(mode=mode)
    out = DH.heretic_reproduce(sess, data, dry_run=False)
    assert out["ok"] and out["committed"], out
    name = default_name(data)
    assert out["ran"] == {"measurement": name, "measured": True, "script": ABLATION_SCRIPT.format(name=name, mode=mode),
                          "dry_run": False}
    patch = ws.patch(out["patch"])
    sha = hashlib.sha256(json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    assert name == f"heretic-{sha[:12]}"
    assert patch.source["origin"] == {
        "kind": "heretic_reproduce", "file_sha256": sha, "version": "4", "timestamp": "2026-11-19T12:00:00", "model": MODEL, "model_commit": MODEL_SHA,
        "modifier": "Abliteration", "measurement": name, "digest": DSR.piece_digest(patch.deltas), "file_digest": None}
    assert out["digest"] == DSR.piece_digest(patch.deltas) and out["same_as_file"] is None
    # the measurement: Heretic's chat rendering of the dataset rows, from the file's specifications
    meta = DR.load(ws, name)[0]
    assert meta["options"] == {"chat": True, "system": SYSTEM, "prefix": None, "chat_options": {},
                               "winsorization_quantile": 1.0}
    assert [p["commit"] for p in meta["datasets"]["good"]["parts"]] == [C_GOOD]
    assert DR.prompts(ws, name) == {"good": GOOD_ROWS, "bad": BAD_ROWS}
    ws.set_patch(out["patch"], enabled=False)
    ref = committed_reference(sess, ABLATION_SCRIPT.replace('"{mode}"', f'"{mode}"'))
    assert torch.equal(DR.load(ws, name)[1], DR.load(ws, "ref")[1]) and torch.equal(DR.load(ws, name)[2], DR.load(ws, "ref")[2])
    same_pieces(patch.deltas, ws.patch(ref).deltas)
    # the patch has an ordinary reproduction record
    rec, _ = REC.read(ws, patch)
    assert rec["calls"][0]["function"] == "ablate_direction" and rec["rebuildable"] is True
    assert rec["origin"]["kind"] == "heretic_reproduce"


# ------------------------------------------------------------------ HR2 ARA, seeded with the file's seed
def test_an_ara_file_makes_the_hand_made_fit(sess):
    ws = sess.ws
    data = heretic_file("ara")
    out = DH.heretic_reproduce(sess, data, dry_run=False)
    assert out["ok"] and out["committed"], out
    name = default_name(data)
    assert out["ran"]["script"] == ARA_SCRIPT.format(name=name)
    assert DR.load(ws, name)[0]["module_io"]
    ws.set_patch(out["patch"], enabled=False)
    ref = committed_reference(sess, ARA_SCRIPT, module_io=True)
    same_pieces(ws.patch(out["patch"]).deltas, ws.patch(ref).deltas)
    assert REC.HERETIC_DIFFERENCES["ara"] in out["heretic"]["notes"]


# ------------------------------------------------------------------ HR3 a dry run, the reuse of the measurement
def test_a_dry_run_commits_nothing_and_a_second_run_reuses_the_measurement(sess):
    ws = sess.ws
    data = heretic_file()
    before = [p.id for p in ws.patches]
    dry = DH.heretic_reproduce(sess, data)                       # dry_run is the default
    assert dry["ok"] and not dry.get("committed") and [p.id for p in ws.patches] == before
    assert dry["ran"]["measured"] is True and dry["ran"]["dry_run"] is True
    created = DR.load(ws, default_name(data))[0]["created"]
    again = DH.heretic_reproduce(sess, data, dry_run=False)
    assert again["committed"] and again["ran"]["measured"] is False and again["digest"] == dry["digest"]
    assert DR.load(ws, default_name(data))[0]["created"] == created
    ws.set_patch(again["patch"], enabled=False)
    assert DH.check_heretic_reproduce(ws, data)["measurement"]["state"] == "reuse"


def test_a_measurement_of_that_name_made_otherwise_is_refused(sess):
    ws = sess.ws
    DR.measure(ws, "taken", GOOD_ROWS, BAD_ROWS, chat=True, system="other")
    assert reason(ws, heretic_file(), measurement="taken") == (
        "a measurement named 'taken' exists and was not made from this file's prompts and settings (options differs): "
        "pass another measurement name")
    DR.measure(ws, "plain", GOOD_ROWS, BAD_ROWS, chat=True, system=SYSTEM)
    assert DH.check_heretic_reproduce(ws, heretic_file(), measurement="plain")["measurement"]["state"] == "reuse"
    assert reason(ws, heretic_file("ara"), measurement="plain") == (
        "a measurement named 'plain' exists and was not made from this file's prompts and settings (module_io "
        "differs): pass another measurement name")
    DR.measure(ws, "fewer", GOOD_ROWS[:2], BAD_ROWS, chat=True, system=SYSTEM)
    assert "(prompts_sha256 differs)" in reason(ws, heretic_file(), measurement="fewer")


# ------------------------------------------------------------------ HR4 what cannot be honoured, one reason per item
def test_each_refusal_with_its_reason(sess):
    ws = sess.ws

    def edited(change, method="ablation"):
        d = heretic_file(method)
        change(d)
        return d
    assert reason(ws, edited(lambda d: d.update(version="3"))) == (
        'file format version "3": the IDE reads version 4 (plugin-based) reproduce.json files, as Heretic at 64e9991 '
        'does (older files were written before Heretic\'s plugins)')
    ws.meta["repo"], ws.meta["sha"] = "org/other", MODEL_SHA
    assert reason(ws, heretic_file()) == "the file's model is org/model; this workspace's base is org/other"
    ws.meta["repo"], ws.meta["sha"] = MODEL, "d4" * 20
    assert reason(ws, heretic_file()) == (f"the file's model commit is {MODEL_SHA}; this workspace's base is org/model "
                                          f"at {'d4' * 20}")
    ws.meta["sha"] = MODEL_SHA
    assert DH.check_heretic_reproduce(ws, heretic_file())["model"]["repository"] == "same"
    assert reason(ws, edited(lambda d: d["settings"].update(quantization="bnb_4bit"))) == (
        'settings.quantization is "bnb_4bit": the IDE does not load quantized transformers models, and the measurement '
        'and the edit would be of other weights')
    assert reason(ws, edited(lambda d: d["settings"]["modifiers"][0].update(plugin="my.plugins.Steer"))) == (
        "the modifier my.plugins.Steer is not one the IDE has (heretic.modifiers.abliteration.Abliteration, "
        "heretic.modifiers.ara.ARA); it makes the edit, so the file cannot be reproduced without it")
    assert reason(ws, edited(lambda d: d["settings"]["modifiers"].append(d["settings"]["modifiers"][0]))) == (
        "settings.modifiers lists 2 modifiers; Heretic applies exactly one")
    assert reason(ws, edited(lambda d: d["settings"].update(response_prefix=None))) == (
        "settings.response_prefix is not set: Heretic would detect it by generating responses; detect it with POST "
        "/api/directions/prefix and put it into the file")
    assert reason(ws, edited(lambda d: d["parameters"].update(direction_index="x"))).startswith("parameters: ")
    assert reason(ws, edited(lambda d: d["parameters"].update(extra=1))) == "parameters: extra: not a parameter of Abliteration"
    assert reason(ws, edited(lambda d: d["settings"].update(seed=None), "ara")) == (
        "settings.seed must be an integer: ARA's fit is seeded with it")
    assert reason(ws, edited(lambda d: d.pop("parameters"))) == (
        "parameters: missing (a reproduce.json holds the chosen trial's parameters)")
    # a dataset that is not in the cache: the dataset code's own error, naming the file's field
    gone = reason(ws, edited(lambda d: d["settings"]["modifier"]["Abliteration"]["good_prompts"].update(commit="e5" * 20)))
    assert gone.startswith(f"settings.modifier.Abliteration.good_prompts: org/good at {'e5' * 6} is not in the local "
                           f"Hugging Face cache"), gone
    assert "bad_prompts" not in gone                                          # one reason: the good prompts only
    # a patch switched on: Heretic edits the base
    r = sess.apply("mlp(0).W_out *= 2")
    assert r["committed"]
    assert reason(ws, heretic_file()) == (f"patches {r['patch']} are switched on; Heretic edits the base model: switch "
                                          f"them off to reproduce the file")
    ws.set_patch(r["patch"], enabled=False)
    # several at once: every reason, in order, joined
    both = edited(lambda d: (d.update(version="5"), d["settings"].update(quantization="bnb_4bit")))
    assert reason(ws, both) == (
        'file format version "5": the IDE reads version 4 (plugin-based) reproduce.json files, as Heretic at 64e9991 '
        'does (older files were written before Heretic\'s plugins); settings.quantization is "bnb_4bit": the IDE does '
        'not load quantized transformers models, and the measurement and the edit would be of other weights')


def test_nothing_runs_when_a_reason_refuses(sess):
    ws = sess.ws
    d = heretic_file()
    d["version"] = "3"
    with pytest.raises(REC.RecordError):
        DH.heretic_reproduce(sess, d, dry_run=False)
    assert not DR.exists(ws, default_name(d)) and ws.patches == []


# ------------------------------------------------------------------ HR5 reported, not refused
def test_the_model_is_not_verified_when_the_workspace_does_not_know_it(sess):
    plan = DH.check_heretic_reproduce(sess.ws, heretic_file())
    assert plan["model"]["repository"] == plan["model"]["commit"] == "not verified"
    assert plan["model"]["notes"] == ["the workspace does not know its Hugging Face repository (it was opened from a local "
                                      "directory), so the base is not verified to be the file's model"]
    sess.ws.meta["repo"] = MODEL
    plan = DH.check_heretic_reproduce(sess.ws, heretic_file())
    assert (plan["model"]["repository"], plan["model"]["commit"]) == ("same", "not verified")
    assert plan["model"]["notes"] == ["the workspace does not know its commit, so the base is not verified to be the "
                                      "file's commit"]


def test_library_differences_are_graded_as_heretic_grades_them(sess):
    import importlib.metadata as md
    d = heretic_file()
    d["system"] = {"python": {"version": platform.python_version()}, "os": {"platform": "Other-OS-1"},
                   "cpu": {"brand": "Some CPU"}, "accelerators": {"type": "NoSuchAccelerator"}}
    env = DH.check_heretic_reproduce(sess.ws, d)["environment"]

    def installed(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None
    assert env["severity"] == "critical" and env["same"] == 1                       # the Python version
    got = [(r["name"], r["severity"], r["file"], r["ide"]) for r in env["differences"]]
    assert got == [
        ("heretic-llm", "critical", "2.2.0.dev0-local",
         "2.2.0.dev0-git+https://github.com/p-e-w/heretic@64e999152ff7918efc73576814709b6b6a71f15c"),
        ("Accelerator type", "high", "NoSuchAccelerator", got[1][3]),
        ("torch", "high", "0.0.1", str(torch.__version__)),
        ("transformers", "high", "0.0.2", installed("transformers")),
        ("peft", "medium", "0.0.3", installed("peft")),
        ("tokenizers", "medium", "0.0.4", installed("tokenizers")),
        ("CPU", "low", "Some CPU", None),
        ("Operating system", "low", "Other-OS-1", platform.platform()),
        ("zz-not-installed", "low", "1.0", None)]
    assert [r["same"] for r in env["differences"]] == [False] * 6 + [None, False, False]
    # an IDE-written file names the commit the IDE implements: no Heretic row
    d["environment"]["heretic"] = {"version": "2.2.0.dev0", "is_standard_pypi": False,
                                   "metadata": {"type": "git", "url": "https://github.com/p-e-w/heretic",
                                                "commit_hash": "64e999152ff7918efc73576814709b6b6a71f15c"}}
    rows = DH.check_heretic_reproduce(sess.ws, d)["environment"]["differences"]
    assert "heretic-llm" not in [r["name"] for r in rows]
    d.pop("system")
    assert DH.NO_SYSTEM_NOTE in DH.check_heretic_reproduce(sess.ws, d)["notes"]


def test_the_settings_honoured_reported_and_not_used(sess):
    d = heretic_file()
    d["settings"]["made_up"] = 1
    d["settings"]["batch_size"] = 4
    d["settings"]["dtypes"] = ["bfloat16"]
    s = DH.check_heretic_reproduce(sess.ws, d)["settings"]
    assert s["honoured"] == ["model", "model_commit", "quantization", "response_prefix", "system_prompt", "modifiers",
                             "modifier"]
    rep = {r["field"]: (r["file"], r["ide"], r["same"]) for r in s["reported"]}
    assert rep == {"dtypes": (["bfloat16"], "float32", False), "device_map": ("cpu", "cpu", True),
                   "batch_size": (4, 1, False)}
    why = {r["field"]: r["why"] for r in s["not_used"]}
    assert why["n_trials"] == "Heretic's search: a reproduction applies the file's one trial"
    assert why["made_up"] == "not one of Heretic's settings at 64e9991"
    assert set(why) == {"max_memory", "offload_outputs_to_cpu", "max_response_length", "scorers", "n_trials",
                        "n_startup_trials", "modified_model_type", "max_shard_size", "export_strategy",
                        "checkpoint_action", "trial_index", "n_additional_trials", "model_action",
                        "upload_repo_private", "upload_reproducibility_information", "upload_tags",
                        "ignore_mismatches", "scorer", "made_up", "seed"}
    assert why["seed"] == ("Heretic's directional ablation uses the seed only for row_normalization full, and the IDE "
                           "computes the best rank-r one, which needs no seed")
    ara = DH.check_heretic_reproduce(sess.ws, heretic_file("ara"))["settings"]
    assert ara["honoured"][-1] == "seed" and "seed" not in [r["field"] for r in ara["not_used"]]


def test_the_statements(sess, monkeypatch):
    d = heretic_file()
    d["settings"]["scorers"].append({"plugin": "heretic.scorers.bert_classifier.BertClassifier",
                                     "optimization": "minimize", "instance_name": None})
    d["settings"]["modifier"]["Abliteration"]["junk"] = 1
    del d["settings"]["modifier"]["Abliteration"]["winsorization_quantile"]
    d["extra_top"] = {}
    plan = DH.check_heretic_reproduce(sess.ws, d)
    assert plan["notes"] == [
        "the file has no system information (Heretic's \"basic\"): the Python version, operating system, CPU and "
        "accelerator are not compared",
        "the scorer heretic.scorers.bert_classifier.BertClassifier is not one the IDE has; a reproduction scores nothing, "
        "so the patch is the same without it",
        "the file describes one trial (its parameters and scores) of a search of 2 trials (seed 7); the search's other "
        "trials are in Heretic's study journal beside the file, not in reproduce.json, and nothing of the search is run "
        "again",
        "settings.seed is not used: Heretic's directional ablation uses the seed only for row_normalization full",
        "settings.modifier.Abliteration.junk is not a setting of Abliteration; Heretic ignores it too",
        "settings.modifier.Abliteration does not set winsorization_quantile: Heretic's defaults are used, as Heretic "
        "uses them",
        "extra_top is not part of Heretic's reproduce.json; not used"]
    assert plan["file"]["hashes"] == {"adapter_model.safetensors": "de" * 32} and plan["file"]["hashes_note"]
    full = DH.check_heretic_reproduce(sess.ws, heretic_file(mode="full"))["notes"]
    assert REC.HERETIC_DIFFERENCES["full"] in full and not any(n.startswith("settings.seed") for n in full)
    # a model with routed experts: Heretic at 64e9991 under transformers 5.x ablates the shared expert only
    monkeypatch.setattr(DR, "has_routed_experts", lambda ws: True)
    d = heretic_file()
    d["environment"]["requirements"]["transformers"] = "5.18.0"
    plan = DH.check_heretic_reproduce(sess.ws, d)
    assert plan["script"].endswith(', expert_modules="shared")')
    assert plan["notes"][0] == (REC.EXPERT_MODULES_NOTE["ablation"] + '; the reproduction passes expert_modules="shared" (the '
                                'file\'s transformers is 5.18.0)')
    d["environment"]["requirements"]["transformers"] = "4.57.0"
    assert "expert_modules" not in DH.check_heretic_reproduce(sess.ws, d)["script"]


def test_the_winsorization_quantile_is_the_measurement_s(sess):
    ws = sess.ws
    d = heretic_file()
    d["settings"]["modifier"]["Abliteration"]["winsorization_quantile"] = 0.5
    assert DH.check_heretic_reproduce(ws, d)["measurement"]["request"]["winsorization_quantile"] == 0.5
    out = DH.heretic_reproduce(sess, d, dry_run=False)
    assert out["committed"]
    assert DR.load(ws, out["ran"]["measurement"])[0]["options"]["winsorization_quantile"] == 0.5
    ws.set_patch(out["patch"], enabled=False)
    DR.measure(ws, "ref", GOOD_ROWS, BAD_ROWS, chat=True, system=SYSTEM, winsorization_quantile=0.5)
    ref = sess.apply(ABLATION_SCRIPT.format(name="ref", mode="none"))
    same_pieces(ws.patch(out["patch"]).deltas, ws.patch(ref["patch"]).deltas)
    ws.set_patch(ref["patch"], enabled=False)
    # the ARA table has no such setting: ARA measures without winsorization
    assert DH.check_heretic_reproduce(ws, heretic_file("ara"))["measurement"]["request"]["winsorization_quantile"] == 1.0


def test_an_instance_table_overrides_the_class_table(sess):
    d = heretic_file()
    d["settings"]["modifiers"][0]["instance_name"] = "x"
    d["settings"]["modifier"]["Abliteration_x"] = {"row_normalization": "pre"}
    plan = DH.check_heretic_reproduce(sess.ws, d)
    assert 'row_normalization="pre"' in plan["script"] and plan["file"]["instance_name"] == "x"


# ------------------------------------------------------------------ HR6 the round trip: an IDE edit written and read
def test_an_ide_edit_written_as_reproduce_json_reads_back_to_the_same_patch(tmp_path, monkeypatch, hub):
    use_template(monkeypatch)
    one = session_on(tmp_path / "one", chat_ws)
    ws = one.ws
    DR.measure(ws, "m", spec(GOOD_REPO, C_GOOD), spec(BAD_REPO, C_BAD), chat=True, system=SYSTEM)
    r = one.apply(ABLATION_SCRIPT.format(name="m", mode="full"))
    assert r["committed"]
    rec, _ = REC.read(ws, ws.patch(r["patch"]))
    # the built-in tiny model stands in for a checkpoint: to_heretic's rule H2 asks for the kind "hf"
    data = REC.to_heretic(ws, {**rec, "base": {**rec["base"], "kind": "hf"}}, r["patch"], model=MODEL)
    data = json.loads(json.dumps(data))
    two = session_on(tmp_path / "two", chat_ws)
    out = DH.heretic_reproduce(two, data, dry_run=False)
    assert out["committed"] and out["same_as_file"] is True
    assert out["digest"] == out["file_digest"] == rec["pieces"]["digest"] == DSR.piece_digest(ws.patch(r["patch"]).deltas)
    same_pieces(ws.patch(r["patch"]).deltas, two.ws.patch(out["patch"]).deltas)


# ------------------------------------------------------------------ HR7 the route
def test_the_route(sess):
    ws = sess.ws
    c = client(sess)
    url = "/api/directions/reproduce"
    r = c.post(url, json={"reproduce": heretic_file(), "nope": 1})
    assert (r.status_code, r.json()) == (400, {"error": "RecordError: nope: unknown field; the fields are reproduce, "
                                                        "measurement, check, dry_run, message and enforce_scope"})
    assert c.post(url, json={}).json() == {"error": "RecordError: reproduce: required (the content of a Heretic "
                                                    "reproduce.json)"}
    assert c.post(url, json={"reproduce": heretic_file(), "dry_run": "yes"}).status_code == 400
    bad = heretic_file()
    bad["version"] = "3"
    r = c.post(url, json={"reproduce": bad})
    assert r.status_code == 400 and r.json()["error"].startswith(
        "RecordError: this reproduce.json cannot be run in the IDE: file format version \"3\"")
    # check: the plan, nothing measured, nothing committed
    r = c.post(url, json={"reproduce": heretic_file(), "check": True, "measurement": "mine"})
    assert r.status_code == 200 and r.json()["ran"] is None
    plan = r.json()["heretic"]
    assert plan["measurement"]["name"] == "mine" and plan["measurement"]["state"] == "new"
    assert plan["script"] == ABLATION_SCRIPT.format(name="mine", mode="none")
    assert plan["measurement"]["request"] == {"name": "mine", "good": spec(GOOD_REPO, C_GOOD),
                                              "bad": spec(BAD_REPO, C_BAD), "chat": True, "system": SYSTEM,
                                              "prefix": None, "variant": "base", "winsorization_quantile": 1.0,
                                              "module_io": False}
    assert not DR.exists(ws, "mine") and ws.patches == []
    # a search running: 409, nothing measured
    with DSR.search_lock(ws):
        r = c.post(url, json={"reproduce": heretic_file(), "measurement": "mine", "dry_run": False})
        assert (r.status_code, r.json()) == (409, {"error": "a search is running on this workspace; stop it before "
                                                            "running a reproduce.json"})
    assert not DR.exists(ws, "mine")
    r = c.post(url, json={"reproduce": heretic_file(), "measurement": "mine"})           # a dry run by default
    assert r.status_code == 200 and r.json()["ok"] and not r.json().get("committed") and ws.patches == []
    assert r.json()["ran"]["dry_run"] is True and r.json()["ran"]["measured"] is True
    r = c.post(url, json={"reproduce": heretic_file(), "measurement": "mine", "dry_run": False, "message": "from file"})
    assert r.status_code == 200 and r.json()["committed"] and "status" in r.json(), r.text
    assert ws.patch(r.json()["patch"]).message == "from file"
