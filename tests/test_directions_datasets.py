"""Prompt sets from datasets (wsp.directions_datasets), model-free and without the network.

Where the expectations come from (nothing here computes an expectation with the code under test):

- the split ranges of T1 were printed by ``datasets.ReadInstruction.from_spec(s).to_absolute(lengths)`` itself
  (datasets 5.0.1, the version Heretic 64e9991 locks) and are typed here; the refusals of T2 are the strings
  datasets refuses (arrow_reader.py), with the IDE's messages;
- the rows of the local layouts (T4, T6, T7, T8) are the rows Heretic's own code paths returned for the same
  layouts (its text-file branch, ``load_from_disk`` and ``load_dataset`` on a directory with Heretic's arguments),
  typed as lists;
- the text-file rules (T3, T14) are applied by hand from heretic/src/heretic/utils.py ``_load_prompts_single``;
- hashes are computed in the test with ``hashlib`` and ``json`` over bytes and texts typed in the test;
- Heretic's defaults (T34) are typed from Heretic's sources (config.py, the abliteration and ARA modifiers and the
  three scorers).

Tests marked [pyarrow] need pyarrow (requirements-datasets.txt) and skip without it.
"""
import builtins
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from wsp import directions as DR
from wsp import directions_datasets as DD

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = [f"prompt {i} é中" for i in range(30)]
SYS = "You are a helpful assistant."


# ------------------------------------------------------------------ guards and helpers
REAL_HUB_INFO, REAL_HUB_FETCH = DD.hub_info, DD.hub_fetch


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """T0: any test that reaches huggingface_hub's download or listing fails, unless it installs a recorder; a test
    that reaches the Hugging Face cache without the ``hub`` fixture fails too."""
    import huggingface_hub

    def download(*a, **k):
        raise AssertionError("network: huggingface_hub.hf_hub_download")

    def info(*a, **k):
        raise AssertionError("network: huggingface_hub.HfApi.dataset_info")

    def no_cache():
        raise AssertionError("the real Hugging Face cache: use the hub fixture")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", info)
    monkeypatch.setattr(DD, "hub_cache_dir", no_cache)


def w(root, rel, text, newline=None):
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    return p


def wb(root, rel, data):
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


def J(rows):
    return "".join(json.dumps({"text": r, "id": i}, ensure_ascii=False) + "\n" for i, r in enumerate(rows))


def spec(dataset, **kw):
    return {"dataset": str(dataset), **kw}


def resolve(value, system=None, label="good", field="good", allow_system=True, state=None):
    its = DD.items(value, label, field)
    assert its is not None
    return DD.resolve(its, label, field, system=system, state=state, allow_system=allow_system)


def users(value, **kw):
    return [DD.user_of(p) for p in resolve(value, **kw)[0]]


def refused(value, **kw):
    with pytest.raises(DD.DatasetError) as e:
        resolve(value, **kw)
    return str(e.value)


def shape_refused(value, label="good", field="good"):
    with pytest.raises(DD.DatasetError) as e:
        DD.items(value, label, field)
    return str(e.value)


def sha_hex(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def typed_sha(prompts):
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def pq_write(root, rel, rows, row_group_size=None):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pq.write_table(pa.table({"text": rows, "n": list(range(len(rows)))}), p, row_group_size=row_group_size)
    return p


def arrow_stream(root, rel, rows):
    pa = pytest.importorskip("pyarrow")
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    table = pa.table({"text": rows})
    with pa.OSFile(p, "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return p


def saved(root, rows):
    d = os.path.join(str(root), "saved")
    arrow_stream(d, "data-00000-of-00001.arrow", rows)
    w(d, "state.json", json.dumps({"_data_files": [{"filename": "data-00000-of-00001.arrow"}]}))
    return d


# ------------------------------------------------------------------ T0
def test_t0_the_guard_refuses_the_network():
    import huggingface_hub
    with pytest.raises(AssertionError, match="network"):
        huggingface_hub.hf_hub_download("mlabonne/harmless_alpaca", "README.md", repo_type="dataset")
    with pytest.raises(AssertionError, match="network"):
        huggingface_hub.HfApi().dataset_info("mlabonne/harmless_alpaca")
    with pytest.raises(AssertionError, match="network"):
        DD.hub_info("mlabonne/harmless_alpaca", "main")
    with pytest.raises(AssertionError, match="hub fixture"):
        resolve({"dataset": "mlabonne/harmless_alpaca", "split": "train[:4]", "column": "text"})


# ------------------------------------------------------------------ T1, T2: the split grammar
T1 = [("train[:400]", {"train": 25058}, [("train", 0, 400)]),
      ("train[:400]", {"train": 300}, [("train", 0, 300)]),
      ("train[200:250]", {"train": 416}, [("train", 200, 250)]),
      ("train[-7:]", {"train": 416}, [("train", 409, 416)]),
      ("train[:-5]", {"train": 10}, [("train", 0, 5)]),
      ("train[:10%]", {"train": 25}, [("train", 0, 2)]),
      ("train[:10%]", {"train": 35}, [("train", 0, 4)]),
      ("train[:50%]", {"train": 7}, [("train", 0, 4)]),
      ("train[10%:12%]", {"train": 25058}, [("train", 2506, 3007)]),
      ("train[:10%](pct1_dropremainder)", {"train": 250}, [("train", 0, 20)]),
      ("train[-10%:]", {"train": 104}, [("train", 94, 104)]),
      ("test[:-5%]+train[40%:60%]", {"test": 200, "train": 50}, [("test", 0, 190), ("train", 20, 30)]),
      ("train + test[:3]", {"train": 4, "test": 9}, [("train", 0, 4), ("test", 0, 3)]),
      ("train[1_000:]", {"train": 1500}, [("train", 1000, 1500)]),
      ("train[7:3]", {"train": 10}, [("train", 7, 3)]),
      ("train[-100:-50]", {"train": 30}, [("train", 0, 0)]),
      ("train", {"train": 6}, [("train", 0, 6)])]


@pytest.mark.parametrize("text,lengths,want", T1)
def test_t1_split_ranges_equal_datasets(text, lengths, want):
    assert DD.absolute(DD.parse_split(text), lengths) == want


E20_TAIL = (" is not a split instruction: a split name, optionally with a slice such as [:400], [10:20], [-50:] or "
            "[:10%], parts joined by +")
E21 = "rounding is (closest) or (pct1_dropremainder), and only with percent slices"


@pytest.mark.parametrize("text,message", [
    ("train[1]", "'train[1]'" + E20_TAIL),
    ("train[::2]", "'train[::2]'" + E20_TAIL),
    ("[:5]", "'[:5]'" + E20_TAIL),
    ("tr ain", "'tr ain'" + E20_TAIL),
    ("train[_:3]", "'train[_:3]'" + E20_TAIL),
    ("train[:10](closest)", E21),
    ("train[:10%](foo)", E21),
    ("train[:101%]", "percent bounds must be from -100% to 100%"),
    ("train[:10%]+test[:10%](pct1_dropremainder)", "every percent part must use the same rounding"),
])
def test_t2_split_refusals(text, message):
    with pytest.raises(DD.DatasetError) as e:
        DD.parse_split(text)
    assert str(e.value) == message


def test_t2_split_refusals_name_the_field(tmp_path):
    d = tmp_path / "d"
    w(d, "train.jsonl", J([f"r{i}" for i in range(99)]))
    assert refused(spec(d, split="[:5]", column="text")) == "good.split: '[:5]'" + E20_TAIL
    assert refused([spec(d, split="train[:10%](foo)", column="text")]) == "good[0].split: " + E21
    assert refused(spec(d, split="train[:10%](pct1_dropremainder)", column="text")) == (
        f"good.split: (pct1_dropremainder) needs a split of at least 100 rows; train of {d} has 99")
    assert refused(spec(d, split="test", column="text")) == (
        f"good.split: {d} (config default) has no split 'test'; its splits are train")


# ------------------------------------------------------------------ T3: a plain text file
def test_t3_text_file(tmp_path):
    p = w(tmp_path, "prompts.txt", "  alpha \n\n beta\r\ngamma\n   \n", newline="")
    assert users(spec(p)) == ["alpha", "beta", "gamma"]
    assert users(spec(p, split="[1:]")) == ["beta", "gamma"]
    assert users(spec(p, split="train[:2]")) == ["alpha", "beta"]
    assert refused(spec(p, split="[:1]+[2:]")) == (
        "good.split: a plain text file takes one slice, not parts joined by + (Heretic would read only the first part)")
    canon, rec = resolve(spec(p, column="ignored"))
    assert canon == ["alpha", "beta", "gamma"]
    assert rec["parts"][0]["specification"]["column"] == "ignored"
    assert refused(spec(p, split="[1]")) == "good.split: '[1]'" + E20_TAIL
    assert refused(spec(p, config="x")) == (
        "good.config applies to datasets with configs (not to a plain text file or a save_to_disk directory)")
    assert refused(spec(p, commit="main")) == "good.commit applies to Hugging Face datasets only"


# ------------------------------------------------------------------ T4, T5: directory layouts
def layouts(F):
    w(F, "flat/train.jsonl", J(ROWS[:20]))
    w(F, "flat/test.jsonl", J(ROWS[20:]))
    csvtext = 'text,id\n"has, comma",1\n"has ""quote""",2\n"multi\nline",3\nplain,4\n'
    w(F, "dirs/train/b.csv", csvtext)
    w(F, "dirs/train/a.csv", "text,id\nfirst,0\n")
    w(F, "dirs/validation/x.csv", csvtext)
    w(F, "onejson/prompts.json", json.dumps([{"text": r} for r in ROWS[:7]]))
    w(F, "txtdir/train.txt", "one\r\n\r\n  two  \r\nthree\n\n", newline="")
    w(F, "cfg/README.md", "---\nconfigs:\n- config_name: en\n  data_files:\n  - split: train\n    path: en/train*.jsonl\n"
                          "  - split: test\n    path: en/test.jsonl\n- config_name: de\n  default: true\n"
                          "  data_files: de/*.jsonl\n---\n# card\n")
    w(F, "cfg/en/train1.jsonl", J(ROWS[:3]))
    w(F, "cfg/en/train2.jsonl", J(ROWS[3:5]))
    w(F, "cfg/en/test.jsonl", J(ROWS[5:6]))
    w(F, "cfg/de/x.jsonl", J(["hallo", "welt"]))
    w(F, "mixed/a.csv", "text\nc1\nc2\n")
    w(F, "mixed/b.csv", "text\nc3\n")
    w(F, "mixed/c.json", json.dumps([{"text": "j"}]))
    w(F, "hidden/train.jsonl", J(["visible"]))
    w(F, "hidden/.h/train.jsonl", J(["hidden"]))
    w(F, "hidden/__pycache__/train.jsonl", J(["special"]))
    w(F, "tsv/train.tsv", "text\tid\na, b\t1\nc\t2\n")
    w(F, "keywords/my_train_data.jsonl", J(["tr"]))
    w(F, "keywords/dev.jsonl", J(["dv"]))
    w(F, "keywords/eval-set.jsonl", J(["ev"]))
    w(F, "blank/train.jsonl", J(["x"]) + "\n" + J(["y"]))
    w(F, "bom/train.jsonl", "﻿" + J(["bom"]))


T4 = [("flat", "train", None, ROWS[:20]),
      ("flat", "test[:3]", None, ROWS[20:23]),
      ("dirs", "train", None, ["first", "has, comma", 'has "quote"', "multi\nline", "plain"]),
      ("dirs", "validation[1:3]", None, ['has "quote"', "multi\nline"]),
      ("onejson", "train", None, ROWS[:7]),
      ("txtdir", "train[2:4]", None, ["  two  ", "three"]),
      ("cfg", "train", None, ["hallo", "welt"]),
      ("cfg", "train", "en", ROWS[:5]),
      ("cfg", "test", "en", ROWS[5:6]),
      ("mixed", "train", None, ["c1", "c2", "c3"]),
      ("hidden", "train", None, ["visible"]),
      ("tsv", "train", None, ["a, b", "c"]),
      ("keywords", "train", None, ["tr"]),
      ("keywords", "validation", None, ["dv"]),
      ("keywords", "test", None, ["ev"]),
      ("blank", "train", None, ["x", "y"]),
      ("bom", "train", None, ["bom"])]


@pytest.mark.parametrize("name,split,config,want", T4)
def test_t4_directory_layouts(tmp_path, name, split, config, want):
    layouts(tmp_path)
    kw = {"config": config} if config else {}
    assert users(spec(tmp_path / name, split=split, column="text", **kw)) == want


def test_t4_records_of_a_directory(tmp_path):
    layouts(tmp_path)
    d = tmp_path / "cfg"
    _, rec = resolve(spec(d, split="train", column="text", config="en"))
    part = rec["parts"][0]
    assert (part["kind"], part["format"], part["config"], part["listing"], part["local"]) == (
        "directory", "json", "en", "directory", True)
    assert [f["path"] for f in part["files"]] == ["en/train1.jsonl", "en/train2.jsonl"]
    assert part["split"] == [{"split": "train", "rows": 5, "from": 0, "to": 5}]
    _, rec = resolve(spec(tmp_path / "tsv", split="train", column="text"))
    assert rec["parts"][0]["format"] == "csv"


def test_t5_an_empty_line_of_a_txt_dataset_is_refused(tmp_path):
    layouts(tmp_path)
    d = tmp_path / "txtdir"
    assert refused(spec(d, split="train", column="text")) == (
        f"good prompt 2 ({d}, split train row 2) must be text of 1..20000 characters")
    assert refused(spec(d, split="train", column="line")) == "good.column: a txt dataset has one column, 'text'"


# ------------------------------------------------------------------ T6, T7: parquet, arrow, save_to_disk [pyarrow]
def test_t6_sharded_parquet_in_order_and_concatenated(tmp_path):
    d = tmp_path / "sharded"
    pq_write(d, "data/train-00001-of-00002.parquet", ROWS[10:20])
    pq_write(d, "data/train-00000-of-00002.parquet", ROWS[:10])
    pq_write(d, "data/test-00000-of-00001.parquet", ROWS[20:])
    assert users(spec(d, split="train", column="text")) == ROWS[:20]
    assert users(spec(d, split="test[-2:]", column="text")) == ROWS[28:30]
    assert users(spec(d, split="train+test", column="text")) == ROWS[:30]
    _, rec = resolve(spec(d, split="train+test", column="text"))
    part = rec["parts"][0]
    assert part["format"] == "parquet"
    assert part["split"] == [{"split": "train", "rows": 20, "from": 0, "to": 20},
                             {"split": "test", "rows": 10, "from": 0, "to": 10}]
    assert [f["path"] for f in part["files"]] == ["data/train-00000-of-00002.parquet",
                                                  "data/train-00001-of-00002.parquet",
                                                  "data/test-00000-of-00001.parquet"]
    for f in part["files"]:
        assert f["sha256"] == sha_hex(d / f["path"])
        assert f["bytes"] == os.path.getsize(d / f["path"])


def test_t6_only_the_row_groups_used_are_read(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    d = tmp_path / "groups"
    pq_write(d, "train.parquet", ["a", "b", "c", "d", "e", "f"], row_group_size=2)
    assert pq.ParquetFile(str(d / "train.parquet")).metadata.num_row_groups == 3
    seen = []
    real = pq.ParquetFile.read_row_group

    def spy(self, i, *a, **k):
        seen.append(i)
        return real(self, i, *a, **k)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy)
    assert users(spec(d, split="train[:2]", column="text")) == ["a", "b"]
    assert seen == [0]
    seen.clear()
    assert users(spec(d, split="train[3:5]", column="text")) == ["d", "e"]
    assert seen == [1, 2]


def test_t6_arrow_files(tmp_path):
    d = tmp_path / "arrowdir"
    arrow_stream(d, "train.arrow", ["a", "b", "c"])
    assert users(spec(d, split="train[1:]", column="text")) == ["b", "c"]
    assert refused(spec(d, split="train", column="prompt")) == (
        f"good.column: split train of {d} has no column 'prompt'; its columns are text")


def test_t7_save_to_disk(tmp_path):
    d = saved(tmp_path, ROWS[:12])
    assert users(spec(d, split="train[2:5]", column="text")) == ROWS[2:5]
    assert refused(spec(d, split="train[:3]+train[5:6]", column="text")) == (
        "good.split: a dataset saved with save_to_disk takes one slice, not parts joined by + (Heretic would read only "
        "the first part)")
    _, rec = resolve(spec(d, split="anything[:1]", column="text"))
    part = rec["parts"][0]
    assert (part["kind"], part["format"], part["listing"], part["config"]) == ("saved_to_disk", "arrow", None, None)
    assert part["split"] == [{"split": "anything", "rows": 12, "from": 0, "to": 1}]
    assert part["files"] == [{"path": "data-00000-of-00001.arrow",
                              "bytes": os.path.getsize(os.path.join(d, "data-00000-of-00001.arrow")),
                              "sha256": sha_hex(os.path.join(d, "data-00000-of-00001.arrow"))}]
    assert refused(spec(d, split="train", column="text", config="x")) == (
        "good.config applies to datasets with configs (not to a plain text file or a save_to_disk directory)")


def test_t7_state_json_without_data_files(tmp_path):
    d = tmp_path / "s"
    w(d, "state.json", json.dumps({"_fingerprint": "x"}))
    assert refused(spec(d, split="train", column="text")) == (
        f"good: {d}/state.json does not list its data files (_data_files)")


# ------------------------------------------------------------------ T8: datasets' file rules, measured
def test_t8_a_file_matched_twice_is_read_twice(tmp_path):
    d = tmp_path / "overlap"
    w(d, "train-train.jsonl", '{"text": "a"}\n{"text": "b"}\n')
    assert users(spec(d, split="train", column="text")) == ["a", "b", "a", "b"]


def test_t8_a_literal_card_path_without_a_file(tmp_path):
    d = tmp_path / "literal"
    w(d, "README.md", "---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n"
                      "    path: [x.jsonl, missing.jsonl]\n---\n")
    w(d, "x.jsonl", '{"text": "x"}\n')
    assert refused(spec(d, split="train", column="text")) == (
        f"good: the dataset card of {d} names 'missing.jsonl', which matches no file")


# ------------------------------------------------------------------ T9: configs
def card(d, body):
    w(d, "README.md", "---\n" + body + "---\n")
    w(d, "a.jsonl", J(["A"]))
    w(d, "b.jsonl", J(["B"]))


def test_t9_config_rules(tmp_path):
    d = tmp_path / "none"
    card(d, "configs:\n- config_name: a\n  data_files: a.jsonl\n- config_name: b\n  data_files: b.jsonl\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good.config: required: {d} has several configs (a, b) and none is the default")
    assert refused(spec(d, split="train", column="text", config="c")) == (
        f"good.config: {d} has no config 'c'; its configs are a, b")
    assert users(spec(d, split="train", column="text", config="b")) == ["B"]
    d = tmp_path / "two"
    card(d, "configs:\n- config_name: default\n  data_files: a.jsonl\n- config_name: b\n  default: true\n"
            "  data_files: b.jsonl\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good.config: {d} has several default configs ('default' and 'b'); give config")
    d = tmp_path / "nocard"
    w(d, "train.jsonl", J(["t"]))
    assert refused(spec(d, split="train", column="text", config="x")) == (
        f"good.config: {d} has no config 'x'; its configs are default")
    assert users(spec(d, split="train", column="text", config="default")) == ["t"]
    d = tmp_path / "sep"
    card(d, "configs:\n- config_name: a\n  data_files: a.jsonl\n  sep: ';'\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good: the dataset card of {d} sets 'sep' for config 'a', which changes how the files are read; the IDE "
        f"reads config_name, data_files, data_dir, default and features")
    d = tmp_path / "shape"
    card(d, "configs:\n- config_name: a\n  data_files:\n    train: a.jsonl\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good: the dataset card of {d} gives data_files for config 'a' in a form datasets does not accept either")
    d = tmp_path / "noname"
    card(d, "configs:\n- data_files: a.jsonl\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good: the dataset card of {d} lists a config without a config_name")
    d = tmp_path / "bad"
    card(d, "configs: [a, b\n")
    assert refused(spec(d, split="train", column="text")).startswith(
        f"good: the dataset card (README.md) of {d} cannot be read (")


def test_t9_data_dir_and_features(tmp_path):
    d = tmp_path / "dd"
    w(d, "README.md", "---\nconfigs:\n- config_name: default\n  data_dir: sub\n  features:\n  - name: text\n---\n")
    w(d, "sub/train.jsonl", J(["in sub"]))
    w(d, "train.jsonl", J(["outside"]))
    assert users(spec(d, split="train", column="text")) == ["in sub"]
    d = tmp_path / "dd2"
    w(d, "README.md", "---\nconfigs:\n- config_name: default\n  data_dir: sub/\n  data_files:\n  - split: test\n"
                      "    path: '*.jsonl'\n---\n")
    w(d, "sub/x.jsonl", J(["x"]))
    w(d, "x.jsonl", J(["no"]))
    assert users(spec(d, split="test", column="text")) == ["x"]


# ------------------------------------------------------------------ T10: formats
READS = "the IDE reads parquet, arrow, json, jsonl, ndjson, csv, tsv and txt"


def test_t10_refused_formats(tmp_path):
    for ext in ("zip", "png", "lance"):
        d = tmp_path / ext
        wb(d, f"train.{ext}", b"\x00\x01")
        assert refused(spec(d, split="train", column="text")) == f"good: split train of {d} is .{ext} files; {READS}"
    d = tmp_path / "script"
    w(d, "script.py", "print(1)\n")
    w(d, "README.md", "# a card\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good: {d} has no data files (a loading script, which datasets 4 and later do not run either)")
    d = tmp_path / "cardonly"
    w(d, "README.md", "# a card\n")
    assert refused(spec(d, split="train", column="text")) == f"good: {d} has no data files"
    d = tmp_path / "gz"
    wb(d, "train.jsonl.gz", b"\x1f\x8b")
    assert refused(spec(d, split="train", column="text")) == (
        "good: train.jsonl.gz is compressed (.gz); the IDE reads uncompressed files")
    d = tmp_path / "mixedsplits"
    w(d, "train.csv", "text\na\n")
    w(d, "test.jsonl", J(["b"]))
    assert refused(spec(d, split="train+test", column="text")) == (
        f"good: the splits of {d} are in different formats (train: csv, test: json)")
    assert users(spec(d, split="train", column="text")) == ["a"]


# ------------------------------------------------------------------ T11: without pyarrow
def test_t11_without_pyarrow(tmp_path, monkeypatch):
    real = builtins.__import__

    def refuse(name, *a, **k):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("no pyarrow (test)")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", refuse)
    d = tmp_path / "pq"
    wb(d, "train.parquet", b"PAR1 not really parquet")
    assert refused(spec(d, split="train", column="text")) == (
        "good: reading parquet files needs pyarrow, which is not installed here: python -m pip install -r "
        "requirements-datasets.txt")
    d = tmp_path / "ar"
    wb(d, "train.arrow", b"not really arrow")
    assert refused(spec(d, split="train", column="text")) == (
        "good: reading arrow files needs pyarrow, which is not installed here: python -m pip install -r "
        "requirements-datasets.txt")
    layouts(tmp_path)
    assert users(spec(tmp_path / "flat", split="train[:2]", column="text")) == ROWS[:2]
    assert users(spec(tmp_path / "dirs", split="train[:1]", column="text")) == ["first"]
    assert users(spec(tmp_path / "txtdir", split="train[:1]", column="text")) == ["one"]
    assert users(spec(w(tmp_path, "p.txt", "x\ny\n"))) == ["x", "y"]


# ------------------------------------------------------------------ T12: reading refusals
def test_t12_reading_refusals(tmp_path):
    d = tmp_path / "csv"
    w(d, "train.csv", "prompt,id\na,1\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good.column: split train of {d} has no column 'text'; its columns are prompt, id")
    d = tmp_path / "jrow"
    w(d, "train.jsonl", '{"text": "a"}\n{"other": "b"}\n')
    assert refused(spec(d, split="train", column="text")) == "good.column: row 2 of train.jsonl has no 'text'"
    assert users(spec(d, split="train[:1]", column="text")) == ["a"]
    d = tmp_path / "nontext"
    w(d, "train.jsonl", '{"text": "a"}\n{"text": 3}\n')
    assert refused(spec(d, split="train", column="text")) == (
        f"good: row 2 of split train of {d} is not text in column 'text' (it is int)")
    d = tmp_path / "utf8"
    wb(d, "train.jsonl", b'{"text": "\xff"}\n')
    assert refused(spec(d, split="train", column="text")) == "good: train.jsonl is not valid UTF-8 text"
    d = tmp_path / "badline"
    w(d, "train.jsonl", '{"text": "a"}\n\n{"text": \n')
    assert refused(spec(d, split="train", column="text")) == (
        "good: train.jsonl line 3 is not valid JSON (Expecting value: line 1 column 10 (char 9))")
    d = tmp_path / "dict"
    w(d, "train.json", '{\n  "text": ["a", "b"]\n}\n')
    assert refused(spec(d, split="train", column="text")) == (
        "good: train.json must hold a JSON list of objects or one JSON object per line")
    d = tmp_path / "list"
    w(d, "train.json", '["a", "b"]')
    assert refused(spec(d, split="train", column="text")) == (
        "good: train.json must hold a JSON list of objects or one JSON object per line")
    d = tmp_path / "badarray"
    w(d, "train.json", '[{"text": "a"},\n {"text": }]')
    assert refused(spec(d, split="train", column="text")) == (
        "good: train.json line 2 is not valid JSON (Expecting value: line 2 column 11 (char 26))")
    d = tmp_path / "shortcsv"
    w(d, "train.csv", "id,text\n1\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good: row 1 of split train of {d} is not text in column 'text' (it is NoneType)")
    d = tmp_path / "emptycell"
    w(d, "train.csv", "text,id\n,1\n")
    assert refused(spec(d, split="train", column="text")) == (
        f"good prompt 1 ({d}, split train row 1) must be text of 1..20000 characters")


def test_t12_parquet_column_missing(tmp_path):
    d = tmp_path / "pq"
    pq_write(d, "train.parquet", ["a"])
    assert refused(spec(d, split="train", column="prompt")) == (
        f"good.column: split train of {d} has no column 'prompt'; its columns are text, n")
    wb(d, "test.parquet", b"PAR1 not really parquet PAR1")
    assert refused(spec(d, split="test", column="text")) == "good: test.parquet cannot be read as parquet (ArrowInvalid)"


# ------------------------------------------------------------------ T13: limits
def test_t13_limits(tmp_path, monkeypatch):
    p = w(tmp_path, "many.txt", "".join(f"line {i}\n" for i in range(1025)))
    assert refused(spec(p)) == (
        "good must list 1..1024 prompts; the prompt set gives 1,025 (give a smaller split, such as train[:400])")
    assert len(users(spec(p, split="[:1024]"))) == 1024
    assert refused([spec(p, split="[:1000]"), spec(p, split="[:30]")]) == (
        "good must list 1..1024 prompts; the prompt set gives 1,030 (give a smaller split, such as train[:400])")
    assert refused(spec(p, split="[5:5]")) == "good must list 1..1024 prompts; the prompt set gives 0"
    assert shape_refused([spec(p)] * 17) == "good: at most 16 dataset specifications per prompt set"
    assert len(users([spec(p, split="[:1]")] * 16)) == 16
    assert shape_refused(["x"] * 1024 + [{"user": "y"}]) == "good must list 1..1024 prompts; the list has 1,025 items"
    d = tmp_path / "long"
    w(d, "train.jsonl", J(["short", "y" * 19_998]))
    assert users(spec(d, split="train", column="text", prefix="P")) == ["P short", "P " + "y" * 19_998]
    assert refused(["a", spec(d, split="train", column="text", prefix="PP")]) == (
        f"good prompt 3 ({d}, split train row 2) must be text of 1..20000 characters")
    assert refused(spec(p, split="[:2]", suffix="s" * 20_000)) == (
        f"good prompt 1 ({p}, row 1) must be text of 1..20000 characters")
    d = tmp_path / "bytes"
    w(d, "train-a.jsonl", J(["a"]))
    w(d, "train-b.jsonl", J(["b"]))
    one = os.path.getsize(d / "train-a.jsonl")
    monkeypatch.setattr(DD, "MAX_TEXT_BYTES", one + 1)
    assert refused(spec(d, split="train", column="text")) == (
        f"good: the json, csv and txt files of this specification take {2 * one:,} bytes; the IDE parses at most "
        f"{one + 1:,} bytes of them per specification (parquet has no such limit)")
    monkeypatch.setattr(DD, "MAX_TEXT_BYTES", 256 << 20)
    monkeypatch.setattr(DD, "MAX_FILE_BYTES", 5)
    assert refused(spec(d, split="train", column="text")) == (
        f"good: train-a.jsonl is {one:,} bytes; the IDE reads data files of at most 5 bytes")
    monkeypatch.setattr(DD, "MAX_FILE_BYTES", 2 << 30)
    monkeypatch.setattr(DD, "MAX_SPLIT_FILES", 1)
    assert refused(spec(d, split="train", column="text")) == (
        f"good: split train of {d} has 2 files; at most 1 per specification")


# ------------------------------------------------------------------ T14, T15: prefix, suffix, systems, canonical form
def test_t14_prefix_suffix_system(tmp_path):
    p = w(tmp_path, "x.txt", "x\n")
    assert users(spec(p, prefix="P")) == ["P x"]
    assert users(spec(p, suffix="S")) == ["x S"]
    assert users(spec(p, prefix="P", suffix="S")) == ["P x S"]
    assert users(spec(p, prefix="", suffix="")) == ["x"]
    canon, rec = resolve(spec(p), system="R")
    assert canon == ["x"] and rec["parts"][0]["system"] == "R"
    canon, rec = resolve(spec(p, system_prompt="own"), system="R")
    assert canon == [{"system": "own", "user": "x"}] and rec["parts"][0]["system"] == "own"
    canon, rec = resolve(spec(p, system_prompt=""), system=None)
    assert canon == [{"system": "", "user": "x"}]


def test_t15_canonical_form_and_hash(tmp_path):
    p = w(tmp_path, "x.txt", "first\nsecond\n")
    canon, rec = resolve(["a", {"user": "b"}, {"user": "c", "system": SYS}], system=SYS)
    assert canon == ["a", "b", "c"]
    canon, rec = resolve(spec(p), system=SYS)
    assert canon == ["first", "second"]
    assert rec["prompts_sha256"] == typed_sha(["first", "second"])
    assert rec["prompts_sha256"] == DR._sha(["first", "second"])
    canon, _ = resolve(spec(p, system_prompt=SYS), system=SYS)
    assert canon == ["first", "second"]
    canon, rec = resolve([spec(p, system_prompt="other"), {"user": "x", "system": "mine"}, "t"], system=SYS)
    assert canon == [{"system": "other", "user": "first"}, {"system": "other", "user": "second"},
                     {"system": "mine", "user": "x"}, "t"]
    assert rec["prompts_sha256"] == typed_sha(canon)
    assert [list(c) for c in canon if isinstance(c, dict)] == [["system", "user"]] * 3
    assert DD.canonical([{"user": "x"}], None) == ["x"]
    assert DD.system_of("x", "S") == "S" and DD.system_of({"system": "o", "user": "x"}, "S") == "o"
    assert DD.user_of("x") == "x" and DD.user_of({"system": "o", "user": "y"}) == "y"
    assert DD.prompts_sha256(["first", "second"]) == typed_sha(["first", "second"])


def test_t15_own_system_with_chat_off(tmp_path):
    p = w(tmp_path, "x.txt", "first\n")
    assert refused(["a", spec(p, system_prompt="s")], allow_system=False) == (
        "good prompt 2 has its own system message (from good[1].system_prompt); system messages apply to prompts read "
        "as a chat message; set chat to true or leave system_prompt out")
    assert refused([{"user": "x", "system": ""}], allow_system=False) == (
        "good prompt 1 has its own system message (from good[0].system); system messages apply to prompts read as a "
        "chat message; set chat to true or leave system_prompt out")
    assert users([{"user": "x"}, spec(p)], allow_system=False) == ["x", "first"]


# ------------------------------------------------------------------ T16: the set record
def test_t16_the_record_of_a_mixed_set(tmp_path):
    d = tmp_path / "d"
    w(d, "README.md", "# no front matter\n")
    w(d, "data/train.jsonl", J(["r0", "r1", "r2", "r3"]))
    w(d, "data/test.jsonl", J(["t0"]))
    p = w(tmp_path, "p.txt", "line a\n\nline b\nline c\n")
    value = ["one", "two", spec(d, split="train[1:3]", column="text"),
             spec(p, split="[1:]", system_prompt="own", column="text")]
    canon, rec = resolve(value, system=SYS)
    assert canon == ["one", "two", "r1", "r2", {"system": "own", "user": "line b"},
                     {"system": "own", "user": "line c"}]
    assert rec == {
        "version": 1, "count": 6, "prompts_sha256": typed_sha(canon),
        "parts": [
            {"kind": "inline", "first": 1, "count": 2},
            {"kind": "directory", "first": 3, "count": 2,
             "specification": {"dataset": str(d), "commit": None, "config": None, "split": "train[1:3]",
                               "column": "text", "prefix": "", "suffix": "", "system_prompt": None},
             "repository": None, "commit": None, "commit_from": None, "config": "default", "format": "json",
             "split": [{"split": "train", "rows": 4, "from": 1, "to": 3}],
             "files": [{"path": "data/train.jsonl", "bytes": os.path.getsize(d / "data/train.jsonl"),
                        "sha256": sha_hex(d / "data/train.jsonl")}],
             "listing": "directory", "rows_checked": None, "system": SYS, "prompts_sha256": typed_sha(["r1", "r2"]),
             "local": True},
            {"kind": "text_file", "first": 5, "count": 2,
             "specification": {"dataset": p, "commit": None, "config": None, "split": "[1:]", "column": "text",
                               "prefix": "", "suffix": "", "system_prompt": "own"},
             "repository": None, "commit": None, "commit_from": None, "config": None, "format": "text_file",
             "split": [{"split": None, "rows": 3, "from": 1, "to": 3}],
             "files": [{"path": "p.txt", "bytes": os.path.getsize(p), "sha256": sha_hex(p)}],
             "listing": None, "rows_checked": None, "system": "own",
             "prompts_sha256": typed_sha([{"system": "own", "user": "line b"}, {"system": "own", "user": "line c"}]),
             "local": True}]}
    assert list(rec["parts"][1]) == ["kind", "first", "count", "specification", "repository", "commit", "commit_from",
                                     "config", "format", "split", "files", "listing", "rows_checked", "system",
                                     "prompts_sha256", "local"]


# ------------------------------------------------------------------ the shape of a value (5.1, 5.2)
def test_items_keep_the_old_path_and_check_shapes(tmp_path):
    assert DD.items(["a", "b"], "good", "good") is None
    assert DD.items([], "good", "good") is None
    assert DD.items("a", "good", "good") is None
    assert DD.items(None, "good", "good") is None
    got = DD.items(["a", {"user": "b"}, {"user": "c", "system": ""}, {"dataset": "/x", "split": "train"}],
                   "good", "good")
    assert got == [("good[0]", "a"), ("good[1]", {"user": "b"}), ("good[2]", {"user": "c", "system": ""}),
                   ("good[3]", {"dataset": "/x", "commit": None, "config": None, "split": "train", "column": None,
                                "prefix": "", "suffix": "", "system_prompt": None})]
    assert DD.items({"dataset": "org/name"}, "scores[0]", "scores[0].prompts")[0][0] == "scores[0].prompts"
    F = "scores[1].prompts"
    assert shape_refused([{"user": "x", "dataset": "y"}], "scores[1]", F) == (
        "scores[1].prompts[0]: an object must be a dataset specification (with dataset) or a prompt (with user), not "
        "both")
    assert shape_refused([{"text": "x"}], "scores[1]", F) == (
        "scores[1].prompts[0]: an object must be a dataset specification (with dataset) or a prompt (with user)")
    assert shape_refused([{"user": "x"}, 3], "scores[1]", F) == (
        "scores[1].prompts[1] must be a text, a prompt object (user, system) or a dataset specification (dataset, ...)")
    assert shape_refused({"dataset": "/x", "splits": "train"}) == (
        "good.splits: unknown field; the fields of a dataset specification are dataset, commit, config, split, "
        "column, prefix, suffix, system_prompt")
    assert shape_refused([{"user": "x", "role": "user"}]) == "good[0].role: unknown field; a prompt has user and system"
    assert shape_refused([{"user": ""}]) == "good[0].user must be text of 1..20000 characters"
    assert shape_refused([{"user": "x", "system": None}]) == "good[0].system must be text of at most 20000 characters"
    assert shape_refused({"dataset": ""}) == "good.dataset must be text of 1..4096 characters"
    assert shape_refused({"dataset": "/x", "split": ""}) == "good.split must be text of 1..1024 characters or null"
    assert shape_refused({"dataset": "/x", "commit": 3}) == "good.commit must be text of 1..256 characters or null"
    assert shape_refused({"dataset": "/x", "prefix": None}) == "good.prefix must be text of at most 20000 characters"
    assert shape_refused({"dataset": "/x", "system_prompt": 1}) == (
        "good.system_prompt must be text of at most 20000 characters or null")


def test_inline_texts_in_a_new_path_list(tmp_path):
    assert refused(["", {"user": "b"}]) == "good prompt 1 must be text of 1..20000 characters"
    assert users(["a", {"user": "b"}]) == ["a", "b"]


# ------------------------------------------------------------------ T20: kinds
def test_t20_kinds_refused(tmp_path):
    assert refused({"dataset": "a b", "split": "train", "column": "text"}).startswith(
        "good.dataset: 'a b' is not a Hugging Face repository id (")
    assert refused({"dataset": "a b", "split": "train", "column": "text"}).endswith(
        "), and local paths must be absolute")
    missing = tmp_path / "missing"
    assert refused(spec(missing, split="train", column="text")) == f"good.dataset: no file or directory at {missing}"
    d = tmp_path / "d"
    w(d, "train.jsonl", J(["a"]))
    assert refused(spec(d, column="text")) == (
        "good.split: required for datasets (it is optional only for a plain text file)")
    assert refused(spec(d, split="train")) == "good.column: required for datasets (it is ignored for a plain text file)"
    assert refused(spec(d, split="train", column="text", commit="abc")) == (
        "good.commit applies to Hugging Face datasets only")


def test_t20_home_relative_paths_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    w(tmp_path, "p.txt", "home\n")
    canon, rec = resolve({"dataset": "~/p.txt"})
    assert canon == ["home"]
    assert rec["parts"][0]["specification"]["dataset"] == "~/p.txt"


# ------------------------------------------------------------------ what a local source may read (addendum C)
def test_c1_regular_files_only(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert refused(spec(fifo)) == f"good.dataset: no file or directory at {fifo}"
    d = tmp_path / "d"
    w(d, "train.jsonl", J(["a"]))
    os.mkfifo(d / "test.jsonl")
    assert DD.walk(str(d), "good", str(d)) == ["train.jsonl"]
    assert users(spec(d, split="train", column="text")) == ["a"]
    assert refused(spec(d, split="test", column="text")) == (
        f"good.split: {d} (config default) has no split 'test'; its splits are train")


def test_c2_a_walk_does_not_loop(tmp_path):
    d = tmp_path / "d"
    w(d, "a/train.jsonl", J(["once"]))
    os.symlink(d, d / "a" / "loop")
    os.symlink(d / "a", d / "again")
    assert DD.walk(str(d), "good", str(d)) == ["a/train.jsonl"]
    assert users(spec(d, split="train", column="text")) == ["once"]


def test_c2_a_walk_is_bounded_while_walking(tmp_path, monkeypatch):
    d = tmp_path / "d"
    for i in range(10):
        w(d, f"s{i}/train-{i}.jsonl", J([str(i)]))
    monkeypatch.setattr(DD, "MAX_LISTED_FILES", 5)
    visited = []
    real = os.walk

    def counting(*a, **k):
        for entry in real(*a, **k):
            visited.append(entry[0])
            yield entry
    monkeypatch.setattr(DD.os, "walk", counting)
    assert refused(spec(d, split="train", column="text")) == (
        f"good: {d} holds more than 5 files; give the directory that holds only the dataset")
    assert len(visited) < 11
    assert DD.LIMITS["listed_files"] == 100_000


def test_c3_sizes_before_opening_and_bounded_reads(tmp_path, monkeypatch):
    p = w(tmp_path, "big.txt", "x" * 100 + "\n")
    monkeypatch.setattr(DD, "MAX_TEXT_BYTES", 10)
    assert refused(spec(p)) == (
        "good: this text file takes 101 bytes; the IDE parses at most 10 bytes of them per specification (parquet "
        "has no such limit)")
    opened = []
    real_open = builtins.open

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    refused(spec(p))
    assert p not in opened
    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(DD, "_size", lambda path: 0)      # the file grew after its size was read
    assert refused(spec(p)) == (
        "good: this text file takes 11 bytes; the IDE parses at most 10 bytes of them per specification (parquet "
        "has no such limit)")


def test_c4_paths_stay_inside(tmp_path):
    d = tmp_path / "s"
    w(d, "state.json", json.dumps({"_data_files": [{"filename": "/etc/hosts"}]}))
    assert refused(spec(d, split="train", column="text")) == (
        f"good: {d} lists '/etc/hosts', which is not a path inside the dataset")
    w(d, "state.json", json.dumps({"_data_files": [{"filename": "../x.arrow"}]}))
    assert refused(spec(d, split="train", column="text")) == (
        f"good: {d} lists '../x.arrow', which is not a path inside the dataset")
    assert not DD._inside("") and not DD._inside("a//b") and not DD._inside("data/../../x")
    assert DD._inside("data/train-00000-of-00001.parquet")


def test_c5_no_file_content_in_a_refusal(tmp_path):
    secret = "SECRET-CONTENT"
    messages = []
    d = tmp_path / "badjson"
    w(d, "train.jsonl", f"{secret}\n")
    messages.append(refused(spec(d, split="train", column="text")))
    d = tmp_path / "type"
    w(d, "train.jsonl", json.dumps({"text": [secret]}) + "\n")
    messages.append(refused(spec(d, split="train", column="text")))
    d = tmp_path / "dict"
    w(d, "train.json", json.dumps({secret: [secret]}, indent=1))
    messages.append(refused(spec(d, split="train", column="text")))
    p = w(tmp_path, "long.txt", secret + "x" * 20_000 + "\n")
    messages.append(refused(spec(p)))
    d = tmp_path / "empty"
    w(d, "train.txt", f"{secret}\n\n")
    messages.append(refused(spec(d, split="train", column="text")))
    d = tmp_path / "utf8"
    wb(d, "train.txt", secret.encode() + b"\xff\n")
    messages.append(refused(spec(d, split="train", column="text")))
    d = tmp_path / "csvtype"
    w(d, "train.csv", f"id,text\n{secret}\n")
    messages.append(refused(spec(d, split="train", column="text")))
    assert len(messages) == 7
    for m in messages:
        assert secret not in m, m


def test_c6_agent_tools_do_not_reach_datasets():
    for name in ("tools.py", "mcp.py", "harness.py"):
        with open(os.path.join(APP, "wsp", name), encoding="utf-8") as f:
            assert "directions_datasets" not in f.read(), name


# ------------------------------------------------------------------ T34 (constants)
def test_t34_constants_and_heretic_defaults():
    assert DD.MAX_PROMPTS == DR.MAX_PROMPTS
    assert DD.MAX_CHARS == DR.MAX_CHARS
    assert DD.DISK_RESERVE == DR.DISK_RESERVE
    want = {
        "system_prompt": "You are a helpful assistant.",
        "good_prompts": {"dataset": "mlabonne/harmless_alpaca", "split": "train[:400]", "column": "text"},
        "bad_prompts": {"dataset": "mlabonne/harmful_behaviors", "split": "train[:400]", "column": "text"},
        "scorers": {"KeywordRate": {"dataset": "mlabonne/harmful_behaviors", "split": "test[:100]", "column": "text"},
                    "KLDivergence": {"dataset": "mlabonne/harmless_alpaca", "split": "test[:100]", "column": "text"},
                    "ResponseLength": {"dataset": "mlabonne/harmless_alpaca", "split": "test[:100]", "column": "text"}},
        "response_prefix_test_prompts": [
            {"dataset": "mlabonne/harmless_alpaca", "split": "train[:100]", "column": "text"},
            {"dataset": "mlabonne/harmful_behaviors", "split": "train[:100]", "column": "text"}],
    }
    assert DD.HERETIC_DEFAULTS == want
    assert DD.LIMITS["heretic_defaults"] == want
    assert DD.LIMITS["fields"] == ["dataset", "commit", "config", "split", "column", "prefix", "suffix",
                                   "system_prompt"]
    assert DD.LIMITS["specifications_per_set"] == 16


def test_the_module_imports_only_the_standard_library_at_module_level():
    with open(os.path.join(APP, "wsp", "directions_datasets.py"), encoding="utf-8") as f:
        src = f.read()
    top = [line for line in src.splitlines() if line.startswith(("import ", "from "))]
    assert top == ["from __future__ import annotations", "import csv", "import glob", "import hashlib", "import io",
                   "import json", "import math", "import os", "import re", "import time"]


# ------------------------------------------------------------------ the Hub (5.7, 5.9): fixtures
REPO = "org/name"
C1 = "0123456789abcdef0123456789abcdef01234567"
C2 = "fedcba9876543210fedcba9876543210fedcba98"
C3 = "1111111111111111111111111111111111111111"
TRAIN = J(["h0", "h1", "h2"]).encode()
TEST = J(["x0", "x1"]).encode()
FETCH_HINT = 'fetch them with POST /api/directions/datasets ("download": true)'


def hub_card(train=3, test=2, numbers=True):
    info = (f"dataset_info:\n  features:\n  - name: text\n    dtype: string\n  splits:\n  - name: train\n"
            f"    num_examples: {train}\n  - name: test\n    num_examples: {test}\n") if numbers else ""
    return ("---\n" + info + "configs:\n- config_name: default\n  data_files:\n  - split: train\n"
            "    path: data/train-*\n  - split: test\n    path: data/test-*\n---\n# card\n").encode()


def repo_files(**kw):
    return {"README.md": hub_card(**kw), "data/train-00000-of-00001.jsonl": TRAIN,
            "data/test-00000-of-00001.jsonl": TEST}


def snapshot(cache, repo, commit, files, ref=None):
    """A fake hub-cache snapshot with plain files (datasets--org--name/snapshots/<sha>/...), optional refs/<ref>."""
    base = os.path.join(str(cache), "datasets--" + repo.replace("/", "--"))
    snap = os.path.join(base, "snapshots", commit)
    for rel, data in files.items():
        wb(snap, rel, data)
    if ref:
        w(os.path.join(base, "refs"), ref, commit)
    return snap


def response(status):
    return SimpleNamespace(status_code=status, headers={}, request=None)


class FakeHub:
    """Hub answers typed by the test; a fetch writes the file into the cache's snapshot and reports its bytes."""

    def __init__(self, cache):
        self.cache, self.repos, self.info_calls, self.fetch_calls, self.fail = cache, {}, [], [], {}

    def add(self, repo, revision, commit, files):
        self.repos[(repo, revision)] = (commit, files)

    def info(self, repo, revision):
        self.info_calls.append((repo, revision))
        if ("info", repo) in self.fail:
            raise self.fail[("info", repo)]
        commit, files = self.repos[(repo, revision)]
        return {"commit": commit, "files": [{"path": p, "bytes": len(b)} for p, b in files.items()]}

    def fetch(self, repo, commit, path, *, tqdm_class=None):
        self.fetch_calls.append(path)
        if path in self.fail:
            raise self.fail[path]
        data = next(f for (r, _), (c, f) in self.repos.items() if r == repo and c == commit)[path]
        bar = tqdm_class(total=len(data), unit="B")
        bar.update(len(data))
        bar.close()
        return os.path.join(snapshot(self.cache, repo, commit, {path: data}), path)


@pytest.fixture
def hub(tmp_path, monkeypatch, no_network):
    """The hub cache is tmp_path/hub; DD.hub_info and DD.hub_fetch refuse unless the test installs the fake."""
    cache = tmp_path / "hub"
    monkeypatch.setattr(DD, "hub_cache_dir", lambda: str(cache))

    def refuse(*a, **k):
        raise AssertionError("network: DD.hub_info / DD.hub_fetch")
    monkeypatch.setattr(DD, "hub_info", refuse)
    monkeypatch.setattr(DD, "hub_fetch", refuse)
    fake = FakeHub(cache)

    def install():
        monkeypatch.setattr(DD, "hub_info", fake.info)
        monkeypatch.setattr(DD, "hub_fetch", fake.fetch)
        return fake
    ws = SimpleNamespace(dir=str(tmp_path / "ws"))
    return SimpleNamespace(cache=cache, fake=fake, install=install, ws=ws,
                           state=os.path.join(str(tmp_path), "ws", "directions", "datasets"))


def hs(**kw):
    return {"dataset": REPO, "split": "train", "column": "text", **kw}


def fetch(hub, value, **kw):
    its = DD.items(value, "prompts", "prompts")
    return DD.fetch(its, "prompts", system=None, state=hub.state, **kw)


def fetch_refused(hub, value, **kw):
    with pytest.raises(DD.DatasetError) as e:
        fetch(hub, value, **kw)
    return str(e.value)


# ------------------------------------------------------------------ T17-T19: the cache, offline
def test_t17_pinned_and_cached(hub):
    assert DD.state_dir(hub.ws) == hub.state
    snapshot(hub.cache, REPO, C1, repo_files())
    canon, rec = resolve(hs(commit=C1, split="train[1:]"), state=hub.state)
    assert canon == ["h1", "h2"]
    assert rec["parts"][0] == {
        "kind": "hub", "first": 1, "count": 2,
        "specification": {"dataset": REPO, "commit": C1, "config": None, "split": "train[1:]", "column": "text",
                          "prefix": "", "suffix": "", "system_prompt": None},
        "repository": REPO, "commit": C1, "commit_from": "specification", "config": "default", "format": "json",
        "split": [{"split": "train", "rows": 3, "from": 1, "to": 3}],
        "files": [{"path": "data/train-00000-of-00001.jsonl", "bytes": len(TRAIN),
                   "sha256": hashlib.sha256(TRAIN).hexdigest()}],
        "listing": "cache", "rows_checked": True, "system": None, "prompts_sha256": typed_sha(["h1", "h2"]),
        "local": False}
    snapshot(hub.cache, REPO, C2, repo_files(train=4))
    assert refused(hs(commit=C2), state=hub.state) == (
        f"good: the cached files of split train of {REPO} at {C2[:12]} hold 3 rows and its dataset card says 4: files "
        f"are missing from the cache (it was filled by another tool); {FETCH_HINT}")
    snapshot(hub.cache, REPO, C3, repo_files(numbers=False))
    _, rec = resolve(hs(commit=C3), state=hub.state)
    assert rec["parts"][0]["rows_checked"] is None
    assert hub.fake.info_calls == [] and hub.fake.fetch_calls == []


def test_t17_parquet_from_the_cache(hub):
    d = snapshot(hub.cache, REPO, C1, {"README.md": hub_card(train=4, test=1)})
    pq_write(d, "data/train-00000-of-00001.parquet", ["p0", "p1", "p2", "p3"])
    pq_write(d, "data/test-00000-of-00001.parquet", ["q0"])
    canon, rec = resolve(hs(commit=C1, split="train[:2]+test"), state=hub.state)
    assert canon == ["p0", "p1", "q0"]
    part = rec["parts"][0]
    assert (part["format"], part["rows_checked"], part["listing"]) == ("parquet", True, "cache")
    assert [f["sha256"] for f in part["files"]] == [sha_hex(os.path.join(d, "data/train-00000-of-00001.parquet")),
                                                    sha_hex(os.path.join(d, "data/test-00000-of-00001.parquet"))]


def test_t18_unpinned_commits(hub):
    snapshot(hub.cache, REPO, C1, repo_files(), ref="main")
    canon, rec = resolve(hs(), state=hub.state)
    assert canon == ["h0", "h1", "h2"]
    assert (rec["parts"][0]["commit"], rec["parts"][0]["commit_from"]) == (C1, "cache_ref")
    snapshot(hub.cache, REPO, C2, {**repo_files(numbers=False), "data/train-00000-of-00001.jsonl": J(["other"]).encode()})
    w(hub.state, "org--name/refs.json", json.dumps({"main": {"commit": C2, "listed": 1}}))
    canon, rec = resolve(hs(), state=hub.state)
    assert canon == ["other"]
    assert (rec["parts"][0]["commit"], rec["parts"][0]["commit_from"]) == (C2, "cache_ref")
    snapshot(hub.cache, "org/two", C2, repo_files())
    snapshot(hub.cache, "org/two", C1, repo_files())
    with pytest.raises(DD.NotCached) as e:
        resolve(hs(dataset="org/two"), state=hub.state)
    assert str(e.value) == (
        "good: org/two has no commit given and the cache has no latest commit for it; give commit (cached: "
        "0123456789ab, fedcba987654) or fetch with POST /api/directions/datasets (\"download\": true), which asks the "
        "Hub for its latest commit")
    assert refused(hs(dataset="org/none"), state=hub.state) == (
        "good: org/none has no commit given and the cache has no latest commit for it; give commit (cached: none) or "
        "fetch with POST /api/directions/datasets (\"download\": true), which asks the Hub for its latest commit")
    with pytest.raises(DD.NotCached) as e:
        resolve(hs(commit="v1"), state=hub.state)
    assert str(e.value) == (
        "good.commit: 'v1' is not a commit hash, and the cache does not say which commit it names; give the "
        "40-character commit or fetch with POST /api/directions/datasets (\"download\": true)")
    assert e.value.would_fetch["revision"] == "v1" and e.value.would_fetch["commit"] is None
    snapshot(hub.cache, REPO, C1, {}, ref="v1")
    _, rec = resolve(hs(commit="v1"), state=hub.state)
    assert (rec["parts"][0]["commit"], rec["parts"][0]["commit_from"]) == (C1, "cache_ref")
    assert refused(hs(commit="../../x"), state=hub.state).startswith("good.commit: '../../x' is not a commit hash")


def test_t19_not_cached_is_refused_without_the_network(hub):
    with pytest.raises(DD.NotCached) as e:
        resolve(hs(commit=C1), state=hub.state)
    assert str(e.value) == (
        f"good: {REPO} at {C1[:12]} is not in the local Hugging Face cache ({hub.cache}), and nothing is downloaded "
        f"without a request; fetch it with POST /api/directions/datasets (this prompt set and \"download\": true), "
        f"which fetches the repository's file list, README.md and the files of split train")
    assert e.value.would_fetch == {
        "field": "good", "repository": REPO, "revision": C1, "commit": C1, "files": None,
        "requests": [f"the repository's file list at {C1} (one Hub API request)", "README.md",
                     "the data files of split train (names and sizes known after the file list)"]}
    shards = {"README.md": hub_card(train=4), "data/train-00000-of-00002.jsonl": TRAIN,
              "data/train-00001-of-00002.jsonl": J(["h3"]).encode(), "data/test-00000-of-00001.jsonl": TEST}
    w(hub.state, f"org--name/{C2}.json", json.dumps({"version": 1, "repository": REPO, "commit": C2, "listed": 1,
                                                    "files": [{"path": p, "bytes": len(b)} for p, b in shards.items()]}))
    snapshot(hub.cache, REPO, C2, {p: b for p, b in shards.items() if p != "data/train-00001-of-00002.jsonl"})
    n = len(shards["data/train-00001-of-00002.jsonl"])
    with pytest.raises(DD.NotCached) as e:
        resolve(["a", hs(commit=C2)], state=hub.state)
    assert str(e.value) == (
        f"good[1]: {REPO} at {C2[:12]}: 1 file(s) of split train are not cached (data/train-00001-of-00002.jsonl "
        f"{n:,} bytes); nothing is downloaded without a request; {FETCH_HINT}")
    assert e.value.would_fetch == {
        "field": "good[1]", "repository": REPO, "revision": C2, "commit": C2,
        "files": [{"path": "data/train-00001-of-00002.jsonl", "bytes": n}],
        "requests": ["data/train-00001-of-00002.jsonl"]}
    canon, rec = resolve(hs(commit=C2, split="test"), state=hub.state)
    assert canon == ["x0", "x1"] and rec["parts"][0]["listing"] == "manifest"
    its = DD.items([hs(commit=C1), hs(commit=C2), hs(commit=C2, split="test")], "good", "good")
    assert [x["field"] for x in DD.plan(its, "good", system=None, state=hub.state)] == ["good[0]", "good[1]"]
    os.remove(os.path.join(str(hub.cache), "datasets--org--name", "snapshots", C2, "README.md"))
    with pytest.raises(DD.NotCached) as e:
        resolve(hs(commit=C2, split="test"), state=hub.state)
    assert str(e.value).startswith(f"good: {REPO} at {C2[:12]}: 1 file(s) of split test are not cached (README.md ")
    assert e.value.would_fetch["requests"] == [
        "README.md", "the data files of split test that are not cached (known after README.md)"]
    assert hub.fake.info_calls == [] and hub.fake.fetch_calls == []


# ------------------------------------------------------------------ T21-T24: the fetch, with fakes
def test_t21_fetch_lists_once_and_fetches_the_split_only(hub):
    fake = hub.install()
    files = repo_files()
    fake.add(REPO, "main", C1, files)
    events = []
    got = fetch(hub, hs(split="train[:2]"), progress=events.append)
    assert fake.info_calls == [(REPO, "main")]
    assert fake.fetch_calls == ["README.md", "data/train-00000-of-00001.jsonl"]
    assert got["stopped"] is False
    assert got["fetched"] == [
        {"repository": REPO, "commit": C1, "path": "README.md", "bytes": len(files["README.md"])},
        {"repository": REPO, "commit": C1, "path": "data/train-00000-of-00001.jsonl", "bytes": len(TRAIN)}]
    assert got["fetched_bytes"] == len(files["README.md"]) + len(TRAIN)
    assert got["prompts"] == ["h0", "h1"]
    part = got["datasets"]["parts"][0]
    assert (part["listing"], part["commit"], part["commit_from"], part["rows_checked"]) == ("manifest", C1, "hub", True)
    with open(os.path.join(hub.state, "org--name", C1 + ".json")) as f:
        manifest = json.load(f)
    assert isinstance(manifest.pop("listed"), int)
    assert manifest == {"version": 1, "repository": REPO, "commit": C1,
                        "files": [{"path": p, "bytes": len(b)} for p, b in files.items()]}
    with open(os.path.join(hub.state, "org--name", "refs.json")) as f:
        refs = json.load(f)
    assert list(refs) == ["main"] and refs["main"]["commit"] == C1 and isinstance(refs["main"]["listed"], int)
    assert sorted(os.listdir(hub.state)) == ["org--name"]
    assert events[0] == {"phase": "list", "file": None, "done_bytes": 0, "total_bytes": 0}
    assert events[-1]["file"] == "data/train-00000-of-00001.jsonl"
    assert events[-1]["done_bytes"] == events[-1]["total_bytes"] == len(files["README.md"]) + len(TRAIN)
    again = fetch(hub, hs(split="train[:2]"))
    assert fake.info_calls == [(REPO, "main")] * 2
    assert fake.fetch_calls == ["README.md", "data/train-00000-of-00001.jsonl"]
    assert again["fetched"] == [] and again["fetched_bytes"] == 0 and again["prompts"] == ["h0", "h1"]
    canon, rec = resolve(hs(split="train[:2]"), state=hub.state)
    assert canon == ["h0", "h1"]
    assert (rec["parts"][0]["commit_from"], rec["parts"][0]["listing"]) == ("cache_ref", "manifest")


def test_t21_a_pinned_cached_spec_lists_nothing(hub):
    fake = hub.install()
    snapshot(hub.cache, REPO, C1, repo_files())
    got = fetch(hub, ["inline", hs(commit=C1)])
    assert fake.info_calls == [] and fake.fetch_calls == [] and got["fetched"] == []
    assert got["prompts"] == ["inline", "h0", "h1", "h2"]
    assert got["datasets"]["parts"][1]["commit_from"] == "specification"


def test_t21_progress_class():
    seen = []
    cls = DD.progress_class(seen.append)
    bar = cls(total=10, unit="B", desc="x")
    bar.update(4)
    bar.update(6)
    bar.close()
    assert seen == [4, 6]


def test_t22_refusals_before_any_file_is_fetched(hub, monkeypatch):
    from huggingface_hub import errors as HE
    fake = hub.install()
    fake.add(REPO, C1, C1, repo_files())
    snapshot(hub.cache, REPO, C1, {"README.md": hub_card()})
    monkeypatch.setattr(DD, "MAX_DOWNLOAD_BYTES", 10)
    assert fetch_refused(hub, hs(commit=C1)) == (
        f"prompts: the fetch needs {len(TRAIN):,} bytes; one request fetches at most 10")
    assert fake.fetch_calls == []
    monkeypatch.setattr(DD, "MAX_DOWNLOAD_BYTES", 4 << 30)
    monkeypatch.setattr(DR, "_free_bytes", lambda path: DD.DISK_RESERVE + 10)
    assert fetch_refused(hub, hs(commit=C1)) == (
        f"prompts: the fetch needs {len(TRAIN):,} bytes; the cache's file system ({hub.cache}) has "
        f"{DD.DISK_RESERVE + 10:,} bytes free, of which 2,147,483,648 stay free")
    assert fake.fetch_calls == []
    for exc, message in [
            (HE.OfflineModeIsEnabled("offline"),
             "prompts: download was asked for, but HF_HUB_OFFLINE is set in the IDE's environment, so nothing can be "
             "downloaded"),
            (HE.RepositoryNotFoundError("no", response=response(404)),
             "prompts: the Hub refused org/name (404): it does not exist or is private or gated; a private or gated "
             "dataset needs HF_TOKEN in the IDE's environment"),
            (HE.GatedRepoError("gated", response=response(403)),
             "prompts: the Hub refused org/name (403): it does not exist or is private or gated; a private or gated "
             "dataset needs HF_TOKEN in the IDE's environment"),
            (HE.HfHubHTTPError("auth", response=response(401)),
             "prompts: the Hub refused org/name (401): it does not exist or is private or gated; a private or gated "
             "dataset needs HF_TOKEN in the IDE's environment"),
            (HE.RevisionNotFoundError("rev", response=response(404)),
             "prompts: org/name has no revision 'main' on the Hub"),
            (TimeoutError("slow"),
             "prompts: fetching the file list of org/name failed (TimeoutError: slow); the files fetched before it "
             "stay in the cache")]:
        fake.fail[("info", REPO)] = exc
        assert fetch_refused(hub, hs()) == message
    assert fake.fetch_calls == []
    assert not os.path.exists(os.path.join(hub.state, "org--name", "refs.json"))


def test_t22_a_failed_file_keeps_the_files_before_it(hub):
    fake = hub.install()
    files = {"README.md": hub_card(numbers=False), "data/train-00000-of-00002.jsonl": J(["s0"]).encode(),
             "data/train-00001-of-00002.jsonl": J(["s1"]).encode()}
    fake.add("org/shards", "main", C1, files)
    fake.fail["data/train-00001-of-00002.jsonl"] = RuntimeError("boom")
    assert fetch_refused(hub, hs(dataset="org/shards")) == (
        "prompts: fetching data/train-00001-of-00002.jsonl of org/shards failed (RuntimeError: boom); the files fetched "
        "before it stay in the cache")
    snap = os.path.join(str(hub.cache), "datasets--org--shards", "snapshots", C1)
    assert os.path.isfile(os.path.join(snap, "data/train-00000-of-00002.jsonl"))
    assert not os.path.exists(os.path.join(snap, "data/train-00001-of-00002.jsonl"))


def test_t23_stop_between_files(hub):
    fake = hub.install()
    shards = {f"data/train-0000{i}-of-00003.jsonl": J([f"s{i}"]).encode() for i in range(3)}
    fake.add(REPO, "main", C1, {"README.md": hub_card(numbers=False), **shards})
    snapshot(hub.cache, REPO, C1, {"README.md": hub_card(numbers=False)})
    names = list(shards)
    got = fetch(hub, hs(), stopped=lambda: len(fake.fetch_calls) >= 1)
    assert got == {"stopped": True, "done_bytes": len(shards[names[0]]),
                   "total_bytes": sum(len(b) for b in shards.values()),
                   "fetched": [{"repository": REPO, "commit": C1, "path": names[0], "bytes": len(shards[names[0]])}]}
    assert fake.fetch_calls == names[:1]
    got = fetch(hub, hs())
    assert fake.fetch_calls == names
    assert [f["path"] for f in got["fetched"]] == names[1:]
    assert got["prompts"] == ["s0", "s1", "s2"]


def test_t24_token_policy(hub, monkeypatch):
    import huggingface_hub
    calls = []

    def download(*a, **k):
        calls.append(("download", a, k))
        return "README.md"

    def info(self, *a, **k):
        calls.append(("info", a, k))
        return SimpleNamespace(sha=C1, siblings=[SimpleNamespace(rfilename="README.md", size=5),
                                                 SimpleNamespace(rfilename="data/x.parquet", size=None)])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", info)
    for env, token in ((None, False), ("", False), ("t", "t")):
        if env is None:
            monkeypatch.delenv("HF_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HF_TOKEN", env)
        calls.clear()
        assert REAL_HUB_INFO(REPO, "main") == {"commit": C1, "files": [{"path": "README.md", "bytes": 5},
                                                                      {"path": "data/x.parquet", "bytes": None}]}
        assert REAL_HUB_FETCH(REPO, C1, "README.md", tqdm_class=None) == "README.md"
        assert calls[0] == ("info", (REPO,), {"revision": "main", "files_metadata": True, "token": token})
        assert calls[1] == ("download", (REPO, "README.md"), {"repo_type": "dataset", "revision": C1,
                                                              "cache_dir": str(hub.cache), "token": token,
                                                              "tqdm_class": None})
        assert calls[0][2]["token"] is token


# ------------------------------------------------------------------ T16 (with the Hub), addendum C (Hub paths)
def test_t16_a_hub_part_in_a_mixed_set(hub, tmp_path):
    snapshot(hub.cache, REPO, C1, repo_files())
    p = w(tmp_path, "p.txt", "local\n")
    canon, rec = resolve(["one", "two", hs(commit=C1, split="test"), spec(p)], system=SYS, state=hub.state)
    assert canon == ["one", "two", "x0", "x1", "local"]
    assert [(x["kind"], x["first"], x["count"], x.get("local")) for x in rec["parts"]] == [
        ("inline", 1, 2, None), ("hub", 3, 2, False), ("text_file", 5, 1, True)]
    hubpart = rec["parts"][1]
    assert hubpart["files"] == [{"path": "data/test-00000-of-00001.jsonl", "bytes": len(TEST),
                                 "sha256": hashlib.sha256(TEST).hexdigest()}]
    assert hubpart["split"] == [{"split": "test", "rows": 2, "from": 0, "to": 2}]
    assert (hubpart["repository"], hubpart["commit"], hubpart["listing"], hubpart["rows_checked"]) == (
        REPO, C1, "cache", True)
    assert rec["prompts_sha256"] == typed_sha(["one", "two", "x0", "x1", "local"])


def test_c4_hub_paths_stay_inside(hub):
    w(hub.state, f"org--name/{C1}.json", json.dumps({"version": 1, "repository": REPO, "commit": C1, "listed": 1,
                                                    "files": [{"path": "../x.parquet", "bytes": 1}]}))
    assert refused(hs(commit=C1), state=hub.state) == (
        f"good: {REPO} lists '../x.parquet', which is not a path inside the dataset")
    fake = hub.install()
    fake.add("org/evil", "main", C2, {"README.md": b"---\n---\n", "data/../../x": b"x"})
    assert fetch_refused(hub, hs(dataset="org/evil")) == (
        "prompts: org/evil lists 'data/../../x', which is not a path inside the dataset")
    assert fake.fetch_calls == []
    assert not os.path.exists(os.path.join(hub.state, "org--evil"))


# ------------------------------------------------------------------ part 2, addendum J
def test_j1_an_oversized_card_is_refused_before_it_is_fetched(hub, monkeypatch):
    fake = hub.install()
    files = repo_files()
    fake.add(REPO, "main", C1, files)
    monkeypatch.setattr(DD, "MAX_TEXT_BYTES", 10)
    assert fetch_refused(hub, hs()) == (
        f"prompts: the dataset card (README.md) of {REPO} cannot be read (it takes {len(files['README.md']):,} bytes)")
    assert fake.info_calls == [(REPO, "main")] and fake.fetch_calls == []


def test_j2_a_dataset_name_with_a_control_character(tmp_path):
    message = "must be text of 1..4096 characters without control characters"
    assert shape_refused(spec(str(tmp_path) + "/a\x00b.txt", split="train", column="text")) == f"good.dataset {message}"
    assert shape_refused(["x", {"dataset": "org/na\nme", "split": "train", "column": "text"}]) == (
        f"good[1].dataset {message}")
    for c in ("\x1f", "\x7f", "\t"):
        assert shape_refused({"dataset": "org/name" + c}) == f"good.dataset {message}"
    assert DD.items({"dataset": "org/name é", "split": "train", "column": "text"}, "good", "good")[0][0] == "good"


# ------------------------------------------------------------------ part 2: the requests (tiny models, scripted runners)
# The workspaces are this project's tiny random models (tests/test_steering.py's make_ws, and a wider vocabulary for
# chat renderings as tests/test_directions_scores.py builds it); no model is trained, downloaded or loaded.
CHAT_CHARS = list("abcdefghijklmnopqrstuvwxyz</>\n ")


def tiny_ws(directory):
    from test_steering import make_ws
    return make_ws(directory)


def chat_ws(directory):
    from test_steering import CFG
    from wsp.model import Config, init_params
    from wsp.tokenizer import VocabTokenizer
    from wsp.workspace import Workspace
    cfg = Config(**{**CFG, "d_vocab": len(CHAT_CHARS), "n_ctx": 64})
    return Workspace.create(str(directory), cfg, VocabTokenizer(CHAT_CHARS, "char"), init_params(cfg, seed=2, std=0.3))


def writes_system(self, messages, **options):
    """A chat template that writes the system message (when there is one) before the user turn."""
    system = "".join("xx" + m["content"] for m in messages if m["role"] == "system")
    return system + "zz" + messages[-1]["content"] + "yy"


def use_template(monkeypatch, template=writes_system):
    import wsp.runner
    from wsp import probes as P
    monkeypatch.setattr(P, "chat_template", lambda w: (True, "the model's own chat template"))
    monkeypatch.setattr(wsp.runner.Runner, "chat_prompt", template)


def spy_residuals(monkeypatch):
    got = []
    real = DR.residuals

    def keep(w, prompts_ids, variant="base", module_io=False):
        got.extend([list(x) for x in prompts_ids])
        return real(w, prompts_ids, variant, module_io)
    monkeypatch.setattr(DR, "residuals", keep)
    return got


def session_on(tmp_path, make=None):
    from wsp.session import Session
    (make or tiny_ws)(tmp_path / "rnd")
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def app_client(sess):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def drain(q, kind):
    out = []
    while not q.empty():
        e = q.get_nowait()
        if e.get("type") == kind:
            out.append(e)
    return out


def text_file_part(path, lines, n_rows=None, system=None, **spec_kw):
    """The record of one text-file specification without a split, typed from 6.1 (hashes over the bytes and texts
    written in the test)."""
    with open(path, "rb") as f:
        data = f.read()
    n = len(lines) if n_rows is None else n_rows
    return {"kind": "text_file", "specification": {"dataset": str(path), "commit": None, "config": None, "split": None,
                                                   "column": None, "prefix": "", "suffix": "", "system_prompt": None,
                                                   **spec_kw},
            "repository": None, "commit": None, "commit_from": None, "config": None, "format": "text_file",
            "split": [{"split": None, "rows": n, "from": 0, "to": n}],
            "files": [{"path": os.path.basename(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
            "listing": None, "rows_checked": None, "system": system, "local": True}


# ------------------------------------------------------------------ T25-T27, T31, T33, T19 (measure)
def test_t25_a_measurement_from_a_text_file_equals_the_same_texts_inline(tmp_path):
    import torch
    ws = tiny_ws(tmp_path / "rnd")
    lines = ["abc", "hello", "fox"]
    p = w(tmp_path, "data/good.txt", "  abc\n\nhello \n fox\n")
    from_file = DR.measure(ws, "spec", spec(p), ["xyz", "qq"])
    inline = DR.measure(ws, "inline", lines, ["xyz", "qq"])
    assert DR.prompts(ws, "spec") == DR.prompts(ws, "inline") == {"good": lines, "bad": ["xyz", "qq"]}
    part = {**text_file_part(p, lines), "first": 1, "count": 3, "prompts_sha256": typed_sha(lines)}
    assert from_file["datasets"] == {"version": 1, "good": {"version": 1, "count": 3, "prompts_sha256": typed_sha(lines),
                                                            "parts": [part]}, "bad": None}
    assert inline["datasets"] == {"version": 1, "good": None, "bad": None}
    assert from_file["prompts_sha256"] == inline["prompts_sha256"] == {"good": typed_sha(lines),
                                                                       "bad": typed_sha(["xyz", "qq"])}
    assert (from_file["counts"], from_file["tokens"], from_file["options"]) == (
        inline["counts"], inline["tokens"], inline["options"])
    _, g1, b1 = DR.load(ws, "spec")
    _, g2, b2 = DR.load(ws, "inline")
    assert torch.equal(g1, g2) and torch.equal(b1, b2)
    with open(os.path.join(ws.dir, "directions", "spec.json"), encoding="utf-8") as f:
        assert json.load(f)["datasets"]["good"]["parts"][0]["files"][0]["path"] == "good.txt"


def test_t26_a_system_message_per_prompt(tmp_path, monkeypatch):
    ws = chat_ws(tmp_path / "chat")
    use_template(monkeypatch)
    got = spy_residuals(monkeypatch)
    f1, f2, f3 = (w(tmp_path, n, t) for n, t in (("one.txt", "ab\ncd\n"), ("two.txt", "ef\n"), ("three.txt", "gh\n")))
    good = [spec(f1, system_prompt="sa"), spec(f2, system_prompt="sb")]
    bad = [spec(f3), "zz", {"user": "yy", "system": ""}, {"user": "ww", "system": "q"}]
    meta = DR.measure(ws, "m", good, bad, chat=True, system="q")
    want = [("sa", "ab"), ("sa", "cd"), ("sb", "ef"), ("q", "gh"), ("q", "zz"), ("", "yy"), ("q", "ww")]
    rendered = [ws.runner.chat_prompt([{"role": "system", "content": s}, {"role": "user", "content": u}])
                for s, u in want]
    assert rendered[0] == "xxsazzabyy" and rendered[5] == "xxzzyyyy"
    assert got == [ws.tok.encode(t) for t in rendered]
    assert DR.prompts(ws, "m") == {
        "good": [{"system": "sa", "user": "ab"}, {"system": "sa", "user": "cd"}, {"system": "sb", "user": "ef"}],
        "bad": ["gh", "zz", {"system": "", "user": "yy"}, "ww"]}
    assert meta["options"]["system"] == "q" and meta["counts"] == {"good": 3, "bad": 4}
    assert meta["prompts_sha256"]["bad"] == typed_sha(["gh", "zz", {"system": "", "user": "yy"}, "ww"])
    assert [x["system"] for x in meta["datasets"]["good"]["parts"]] == ["sa", "sb"]
    assert [x["kind"] for x in meta["datasets"]["bad"]["parts"]] == ["text_file", "inline"]
    assert meta["datasets"]["bad"]["parts"][1] == {"kind": "inline", "first": 2, "count": 3}


def test_t27_own_system_messages_need_chat(tmp_path):
    ws = tiny_ws(tmp_path / "rnd")
    p = w(tmp_path, "p.txt", "abc\n")
    tail = "system messages apply to prompts read as a chat message; set chat to true or leave system_prompt out"
    with pytest.raises(DR.DirectionError) as e:
        DR.check_measure(ws, "m", spec(p, system_prompt="s"), ["xyz"])
    assert str(e.value) == f"good prompt 1 has its own system message (from good.system_prompt); {tail}"
    with pytest.raises(DR.DirectionError) as e:
        DR.check_measure(ws, "m", ["abc"], ["xyz", {"user": "ab", "system": "s"}])
    assert str(e.value) == f"bad prompt 2 has its own system message (from bad[1].system); {tail}"
    with pytest.raises(DR.DirectionError) as e:
        DR.check_measure(ws, "m", ["abc"], ["xyz", {"user": "ab", "system": ""}])
    assert str(e.value) == f"bad prompt 2 has its own system message (from bad[1].system); {tail}"
    checked = DR.check_measure(ws, "m", [spec(p), {"user": "ab"}], ["xyz"])
    assert checked["good"] == ["abc", "ab"] and checked["ids"]["good"] == [ws.tok.encode("abc"), ws.tok.encode("ab")]
    assert not os.path.exists(os.path.join(ws.dir, "directions"))


def test_t31_the_old_path_of_a_measurement_is_untouched(tmp_path):
    ws = tiny_ws(tmp_path / "rnd")

    def message(good, bad=("xyz",)):
        with pytest.raises(DR.DirectionError) as e:
            DR.check_measure(ws, "m", good, list(bad))
        return str(e.value)
    assert message(["abc", 3]) == "good prompt 2 must be text of 1..20000 characters"
    assert message(["abc", ""]) == "good prompt 2 must be text of 1..20000 characters"
    assert message(["a"] * 1025) == "good must list 1..1024 prompts"
    assert message([]) == "good must list 1..1024 prompts"
    assert message("abc") == "good must list 1..1024 prompts"
    assert message(["abc"], [["x"]]) == "bad prompt 1 must be text of 1..20000 characters"
    assert message(["abc", {"x": 1}]) == (
        "good[1]: an object must be a dataset specification (with dataset) or a prompt (with user)")
    assert message({"user": "ab", "dataset": "/x"}) == (
        "good: an object must be a dataset specification (with dataset) or a prompt (with user), not both")
    assert message(["abc", 3, {"user": "ab"}]) == (
        "good[1] must be a text, a prompt object (user, system) or a dataset specification (dataset, ...)")
    assert message(["abc", "", {"user": "ab"}]) == "good prompt 2 must be text of 1..20000 characters"
    checked = DR.check_measure(ws, "m", ["abc", "abd"], ["xyz"])
    assert checked["good"] == ["abc", "abd"] and checked["datasets"] == {"good": None, "bad": None}


def test_t31_the_old_path_through_the_route(tmp_path):
    c = app_client(session_on(tmp_path))
    r = c.post("/api/directions/measure", json={"name": "m", "good": ["abc", 3], "bad": ["xyz"]})
    assert r.status_code == 400 and r.json() == {"error": "DirectionError: good prompt 2 must be text of 1..20000 characters"}
    r = c.post("/api/directions/measure", json={"name": "m", "good": ["abc", {"x": 1}], "bad": ["xyz"]})
    assert r.status_code == 400 and r.json() == {
        "error": "DirectionError: good[1]: an object must be a dataset specification (with dataset) or a prompt (with user)"}


def test_t33_the_measure_events_count_resolved_prompts(tmp_path):
    sess = session_on(tmp_path)
    c = app_client(sess)
    q = sess.subscribe()
    p = w(tmp_path, "good.txt", "abc\nabd\nhello\nfox\nzzz\n")
    r = c.post("/api/directions/measure", json={"name": "m", "good": {"dataset": p}, "bad": ["xyz"], "stream": "ms-1"})
    assert r.status_code == 200, r.text
    assert r.json()["counts"] == {"good": 5, "bad": 1}
    ev = drain(q, "directions")
    measuring = [e for e in ev if e["state"] == "measuring"]
    assert measuring and all(e["total"] == 6 for e in measuring) and measuring[0]["done"] == 0
    assert ev[-1]["state"] == "measured" and ev[-1]["total"] == ev[-1]["done"] == 6 and ev[-1]["stream"] == "ms-1"


def test_t19_measure_never_reaches_the_network(hub, tmp_path):
    sess = session_on(tmp_path)
    c = app_client(sess)
    r = c.post("/api/directions/measure", json={"name": "m", "good": hs(commit=C1), "bad": ["xyz"]})
    assert r.status_code == 400
    assert r.json()["error"] == (
        f"DirectionError: good: {REPO} at {C1[:12]} is not in the local Hugging Face cache ({hub.cache}), and nothing "
        f"is downloaded without a request; fetch it with POST /api/directions/datasets (this prompt set and "
        f"\"download\": true), which fetches the repository's file list, README.md and the files of split train")
    r = c.post("/api/directions/measure", json={"name": "m", "good": ["abc"], "bad": [hs()]})
    assert r.status_code == 400 and r.json()["error"].startswith(
        f"DirectionError: bad[0]: {REPO} has no commit given and the cache has no latest commit for it")
    snapshot(hub.cache, REPO, C1, {"README.md": hub_card(numbers=False),
                                   "data/train-00000-of-00001.jsonl": J(["abc", "hello"]).encode()}, ref="main")
    r = c.post("/api/directions/measure", json={"name": "m", "good": hs(), "bad": ["xyz"]})
    assert r.status_code == 200, r.text
    part = r.json()["datasets"]["good"]["parts"][0]
    assert (part["commit"], part["commit_from"], part["listing"]) == (C1, "cache_ref", "cache")
    assert DR.prompts(sess.ws, "m")["good"] == ["abc", "hello"]
    assert hub.fake.info_calls == [] and hub.fake.fetch_calls == []


# ------------------------------------------------------------------ T28-T30, T19, T31 (scores, prefix, search)
def params_of(ws, variant="base"):
    return {n: ws.read(n, (), variant, None) for n in ws.names()}


def kl64(pb, ps, cfg, ids):
    """KL(base || state) of the next-token distribution in float64, from wsp.model.forward (as the scores tests)."""
    import torch
    from wsp.model import forward
    lb = torch.log_softmax(forward(pb, cfg, torch.tensor(ids))[0, -1].double(), -1)
    ls = torch.log_softmax(forward(ps, cfg, torch.tensor(ids))[0, -1].double(), -1)
    return float((lb.exp() * (lb - ls)).sum())


def jsonl_dir(root, name, rows):
    d = os.path.join(str(root), name)
    w(d, "train.jsonl", J(rows))
    return d


def test_t28_kl_from_a_specification_equals_the_same_prompts_inline(tmp_path, monkeypatch):
    from wsp import directions_scores as DS
    sess = session_on(tmp_path, chat_ws)
    ws = sess.ws
    use_template(monkeypatch)
    assert sess.apply("mlp(0).W_out *= 3")["committed"]
    p = w(tmp_path, "kl.txt", "ab\ncd\n")
    as_spec = [spec(p, system_prompt="sa"), "ef"]
    inline = [{"user": "ab", "system": "sa"}, {"user": "cd", "system": "sa"}, "ef"]

    def kl(prompts):
        return DS.score(ws, {"state": "current", "chat": True, "system": "q",
                             "scores": [{"plugin": "KLDivergence", "prompts": prompts}]})
    a, b = kl(as_spec), kl(inline)
    ra, rb = a["scores"][0], b["scores"][0]
    assert ra["reference"]["key"] == rb["reference"]["key"] and rb["reference"]["cached"] is True
    assert ra["per_prompt"] == rb["per_prompt"] and ra["value"] == rb["value"]
    pb, pc = params_of(ws, "base"), params_of(ws, "current")
    rendered = [writes_system(None, [{"role": "system", "content": s}, {"role": "user", "content": u}])
                for s, u in (("sa", "ab"), ("sa", "cd"), ("q", "ef"))]
    want = [kl64(pb, pc, ws.cfg, ws.tok.encode(t)) for t in rendered]
    assert min(want) > 1e-4
    for got, x in zip(ra["per_prompt"], want):
        assert got == pytest.approx(x, rel=1e-6)
    canon = [{"system": "sa", "user": "ab"}, {"system": "sa", "user": "cd"}, "ef"]
    rec_a, rec_b = a["datasets"][0], b["datasets"][0]
    assert rec_a["prompts_sha256"] == rec_b["prompts_sha256"] == typed_sha(canon)
    assert [x["kind"] for x in rec_a["parts"]] == ["text_file", "inline"]
    assert rec_b == {"version": 1, "count": 3, "prompts_sha256": typed_sha(canon),
                     "parts": [{"kind": "inline", "first": 1, "count": 3}]}
    assert kl(["ef", "ab"])["datasets"] == [None]
    out = DS.score(ws, {"state": "base", "chat": True, "system": "q", "max_response_length": 2,
                        "scores": [{"plugin": "ResponseLength", "prompts": as_spec, "print_responses": True},
                                   {"plugin": "KeywordRate", "prompts": ["ab"]}]})
    assert [r["prompt"] for r in out["scores"][0]["responses"]] == canon
    assert out["datasets"][0]["parts"][0]["kind"] == "text_file" and out["datasets"][1] is None
    same = DS.score(ws, {"state": "base", "chat": True, "system": "q", "max_response_length": 2,
                         "scores": [{"plugin": "ResponseLength", "prompts": spec(p, system_prompt="q"),
                                     "print_responses": True}]})
    assert [r["prompt"] for r in same["scores"][0]["responses"]] == ["ab", "cd"]       # the request's: bare texts
    assert same["datasets"][0]["prompts_sha256"] == typed_sha(["ab", "cd"])


def test_t29_prefix_detection_from_heretics_default_list_shape(tmp_path, monkeypatch):
    from wsp import directions_prefix as DP
    from wsp import directions_scores as DS
    ws = chat_ws(tmp_path / "chat")
    use_template(monkeypatch)
    harmless = jsonl_dir(tmp_path, "harmless", ["ab", "cd", "ef"])
    harmful = jsonl_dir(tmp_path, "harmful", ["gh", "ij"])
    seen = []
    real = DS.generate_response

    def spy(w, ids, *a, **kw):
        seen.append(list(ids))
        return real(w, ids, *a, **kw)
    monkeypatch.setattr(DS, "generate_response", spy)
    body = {"prompts": [{"dataset": harmless, "split": "train[:2]", "column": "text"},
                        {"dataset": harmful, "split": "train[:1]", "column": "text"}],
            "system": "q", "max_response_length": 2}
    out = DP.detect(ws, body)
    rendered = [writes_system(None, [{"role": "system", "content": "q"}, {"role": "user", "content": u}])
                for u in ("ab", "cd", "gh")]
    assert seen[:3] == [ws.tok.encode(t) for t in rendered] and out["steps"]["common_prefix"]["responses"] == 3
    rec = out["datasets"]
    assert rec["count"] == 3 and rec["prompts_sha256"] == typed_sha(["ab", "cd", "gh"])
    assert [(x["kind"], x["first"], x["count"], x["listing"], x["split"]) for x in rec["parts"]] == [
        ("directory", 1, 2, "directory", [{"split": "train", "rows": 3, "from": 0, "to": 2}]),
        ("directory", 3, 1, "directory", [{"split": "train", "rows": 2, "from": 0, "to": 1}])]
    assert DP.check_prefix(ws, {"prompts": ["ab"]})["datasets"] is None


def test_t31_the_old_path_of_scores_and_prefix_is_untouched(tmp_path):
    from wsp import directions_prefix as DP
    from wsp import directions_scores as DS
    ws = tiny_ws(tmp_path / "rnd")

    def scores_message(prompts):
        with pytest.raises(DR.DirectionError) as e:
            DS.check_scores(ws, {"state": "base", "scores": [{"plugin": "KeywordRate", "prompts": prompts}]})
        return str(e.value)
    assert scores_message(["a", 3]) == "scores[0] prompt 2 must be text of 1..20000 characters"
    assert scores_message(["a", ""]) == "scores[0] prompt 2 must be text of 1..20000 characters"
    assert scores_message(["a"] * 1025) == "scores[0] must list 1..1024 prompts"
    assert scores_message(["a", {"x": 1}]) == (
        "scores[0].prompts[1]: an object must be a dataset specification (with dataset) or a prompt (with user)")
    assert scores_message({"dataset": "relative/../x y"}).startswith(
        "scores[0].prompts.dataset: 'relative/../x y' is not a Hugging Face repository id (")
    with pytest.raises(DR.DirectionError) as e:
        DP.check_prefix(ws, {"prompts": ["a", 3]})
    assert str(e.value) == "prompts prompt 2 must be text of 1..20000 characters"
    with pytest.raises(DR.DirectionError) as e:
        DP.check_prefix(ws, {"prompts": ["a", {"user": "b", "x": 1}]})
    assert str(e.value) == "prompts[1].x: unknown field; a prompt has user and system"


def test_t19_scores_prefix_and_search_never_reach_the_network(hub, tmp_path, monkeypatch):
    from wsp import directions_prefix as DP
    from wsp import directions_scores as DS
    from wsp import directions_search as DSR
    ws = tiny_ws(tmp_path / "rnd")
    DR.measure(ws, "m", ["abc", "hello"], ["xyz", "qqq"])
    gone = (f"{REPO} at {C1[:12]} is not in the local Hugging Face cache ({hub.cache}), and nothing is downloaded "
            f"without a request")
    with pytest.raises(DR.DirectionError) as e:
        DS.check_scores(ws, {"state": "base", "scores": [{"plugin": "KLDivergence", "prompts": ["ab", hs(commit=C1)]}]})
    assert str(e.value).startswith(f"scores[0].prompts[1]: {gone}")
    with pytest.raises(DSR.SearchError) as e:
        DSR.check_search(ws, {"name": "s1", "modifiers": [{"plugin": "Abliteration", "direction": "m"}],
                              "scorers": [{"plugin": "KLDivergence", "optimization": "minimize", "prompts": hs(commit=C1)}]})
    assert str(e.value).startswith(f"scorers[0].prompts: {gone}")
    chat = chat_ws(tmp_path / "chat")
    use_template(monkeypatch)
    with pytest.raises(DR.DirectionError) as e:
        DP.check_prefix(chat, {"prompts": hs(commit=C1)})
    assert str(e.value).startswith(f"prompts: {gone}")
    sess = session_on(tmp_path / "s")
    c = app_client(sess)
    r = c.post("/api/directions/scores", json={"state": "base", "scores": [{"plugin": "KeywordRate", "prompts": hs()}]})
    assert r.status_code == 400 and r.json()["error"].startswith(
        f"DirectionError: scores[0].prompts: {REPO} has no commit given and the cache has no latest commit")
    assert hub.fake.info_calls == [] and hub.fake.fetch_calls == []
    assert not os.path.exists(hub.state)


def test_t30_a_search_keeps_the_resolved_prompts(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from wsp import directions_search as DSR
    ws = tiny_ws(tmp_path / "rnd")
    DR.measure(ws, "m", ["abc", "hello", "fox"], ["xyz", "qqq", "zzz"])
    kr_file = w(tmp_path, "data/refusal.txt", "ab\ncd\n")
    kl_dir = jsonl_dir(tmp_path / "data", "harmless", ["ef", "gh", "ij"])
    scorers = [{"plugin": "KeywordRate", "optimization": "minimize", "prompts": {"dataset": kr_file},
                "keyword_markers": ["e", "q"]},
               {"plugin": "KLDivergence", "optimization": "minimize",
                "prompts": ["ab", {"dataset": kl_dir, "split": "train[1:]", "column": "text"}]},
               {"plugin": "ResponseLength", "optimization": "none", "prompts": ["zz"]}]
    body = {"name": "s1", "modifiers": [{"plugin": "Abliteration", "direction": "m"}], "scorers": scorers,
            "n_trials": 1, "n_startup_trials": 1, "seed": 7, "max_response_length": 3}
    s = DSR.create(ws, DSR.check_search(ws, body))
    req = s["scoring"]["request"]["scores"]
    assert req == [{**scorers[0], "prompts": ["ab", "cd"]}, {**scorers[1], "prompts": ["ab", "gh", "ij"]},
                   scorers[2]]
    assert [list(x) for x in req] == [list(x) for x in scorers]          # the request's own key order
    recs = s["scoring"]["datasets"]
    assert recs[0] == {"version": 1, "count": 2, "prompts_sha256": typed_sha(["ab", "cd"]), "parts": [
        {**text_file_part(kr_file, ["ab", "cd"]), "first": 1, "count": 2, "prompts_sha256": typed_sha(["ab", "cd"])}]}
    assert [x["kind"] for x in recs[1]["parts"]] == ["inline", "directory"] and recs[2] is None
    with open(os.path.join(ws.dir, "directions", "searches", "s1", "search.json"), encoding="utf-8") as f:
        assert json.load(f)["scoring"]["datasets"] == recs
    import shutil
    shutil.rmtree(os.path.join(str(tmp_path), "data"))
    calls = []
    real = DD.resolve

    def spy(items, *a, **kw):
        calls.append([item for _, item in items])
        return real(items, *a, **kw)
    monkeypatch.setattr(DD, "resolve", spy)
    out = DSR.run(ws, "s1")
    assert out["state"] == "finished" and out["trials"] == {"complete": 1}
    assert not any(isinstance(item, dict) and "dataset" in item for c in calls for item in c)


# ------------------------------------------------------------------ T32, T34, T19 (POST /api/directions/datasets)
ANSWER_KEYS = ["cached", "count", "prompts_sha256", "datasets", "preview", "fetched", "fetched_bytes", "seconds"]


def test_t32_the_route_refuses_bad_requests(tmp_path):
    c = app_client(session_on(tmp_path))
    for body, message in [
            ({"prompts": ["a"], "x": 1}, "x: unknown field; the fields are prompts, system, download, texts and stream"),
            ({}, "prompts: required"),
            ({"prompts": ["a"], "download": "yes"}, "download must be true or false"),
            ({"prompts": ["a"], "texts": 1}, "texts must be true or false"),
            ({"prompts": ["a"], "system": 3}, "system must be text of at most 20000 characters"),
            ({"prompts": ["a"], "system": "s" * 20001}, "system must be text of at most 20000 characters"),
            ({"prompts": []}, "prompts must list 1..1024 prompts"),
            ({"prompts": ["a", 3]}, "prompts prompt 2 must be text of 1..20000 characters"),
            ({"prompts": ["a", {"x": 1}]},
             "prompts[1]: an object must be a dataset specification (with dataset) or a prompt (with user)"),
            ({"prompts": {"dataset": str(tmp_path / "none.txt")}},
             f"prompts.dataset: no file or directory at {tmp_path / 'none.txt'}")]:
        r = c.post("/api/directions/datasets", json=body)
        assert r.status_code == 400 and r.json() == {"error": f"DirectionError: {message}"}, body
    from wsp.session import Session
    closed = app_client(Session(str(tmp_path)))
    r = closed.post("/api/directions/datasets", json={"prompts": ["a"]})
    assert r.status_code == 409


def test_t32_local_sets_through_the_route(tmp_path):
    c = app_client(session_on(tmp_path))
    lines = ["alpha", "beta", "gamma", "delta", "epsilon"]
    p = w(tmp_path, "data/prompts.txt", "\n".join(lines) + "\n")
    r = c.post("/api/directions/datasets", json={"prompts": {"dataset": p}})
    assert r.status_code == 200, r.text
    out = r.json()
    assert list(out) == ANSWER_KEYS and "prompts" not in out
    assert (out["cached"], out["count"], out["preview"], out["fetched"], out["fetched_bytes"]) == (
        True, 5, lines[:3], [], 0)
    assert out["prompts_sha256"] == typed_sha(lines) == out["datasets"]["prompts_sha256"]
    assert out["datasets"]["parts"] == [{**text_file_part(p, lines), "first": 1, "count": 5,
                                         "prompts_sha256": typed_sha(lines)}]
    out = c.post("/api/directions/datasets", json={"prompts": {"dataset": p}, "texts": True}).json()
    assert out["prompts"] == lines and list(out) == ANSWER_KEYS + ["prompts"]
    out = c.post("/api/directions/datasets", json={"prompts": [{"dataset": p, "split": "[3:]", "system_prompt": "s"},
                                                              "x"], "system": "q", "texts": True}).json()
    assert out["prompts"] == [{"system": "s", "user": "delta"}, {"system": "s", "user": "epsilon"}, "x"]
    out = c.post("/api/directions/datasets", json={"prompts": ["a", "b"], "texts": True}).json()
    assert out["datasets"] is None and out["prompts"] == ["a", "b"] and out["prompts_sha256"] == typed_sha(["a", "b"])
    r = c.post("/api/directions/datasets", json={"prompts": {"dataset": p, "split": "train[1]"}})
    assert r.status_code == 400 and r.json()["error"].startswith("DirectionError: prompts.split: 'train[1]' is not")


def test_t32_not_cached_answers_what_a_fetch_would_get(hub, tmp_path):
    sess = session_on(tmp_path)
    c = app_client(sess)
    q = sess.subscribe()
    r = c.post("/api/directions/datasets", json={"prompts": [hs(), "x", hs(commit=C1, split="test")]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert isinstance(out.pop("seconds"), float)
    assert out == {"cached": False, "fetched": [], "fetched_bytes": 0, "would_fetch": [
        {"field": "prompts[0]", "repository": REPO, "revision": "main", "commit": None, "files": None,
         "requests": ["the repository's file list at main (one Hub API request)", "README.md",
                      "the data files of split train (names and sizes known after the file list)"]},
        {"field": "prompts[2]", "repository": REPO, "revision": C1, "commit": C1, "files": None,
         "requests": [f"the repository's file list at {C1} (one Hub API request)", "README.md",
                      "the data files of split test (names and sizes known after the file list)"]}]}
    assert drain(q, "direction_datasets") == []
    snapshot(hub.cache, REPO, C1, repo_files(train=4))
    r = c.post("/api/directions/datasets", json={"prompts": hs(commit=C1)})
    assert r.status_code == 400 and r.json()["error"].startswith(
        f"DirectionError: prompts: the cached files of split train of {REPO} at {C1[:12]} hold 3 rows")
    out = c.post("/api/directions/datasets", json={"prompts": hs(commit=C1, split="test")}).json()
    assert out["cached"] is True and out["preview"] == ["x0", "x1"]
    assert hub.fake.info_calls == [] and hub.fake.fetch_calls == []
    assert not os.path.exists(DD.state_dir(sess.ws))


def test_t32_a_fetch_through_the_route(hub, tmp_path):
    sess = session_on(tmp_path)
    c = app_client(sess)
    q = sess.subscribe()
    fake = hub.install()
    files = repo_files()
    fake.add(REPO, "main", C1, files)
    r = c.post("/api/directions/datasets", json={"prompts": hs(split="train[:2]"), "download": True, "stream": "ds-1"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert list(out) == ANSWER_KEYS and out["cached"] is True and out["preview"] == ["h0", "h1"]
    assert out["fetched"] == [
        {"repository": REPO, "commit": C1, "path": "README.md", "bytes": len(files["README.md"])},
        {"repository": REPO, "commit": C1, "path": "data/train-00000-of-00001.jsonl", "bytes": len(TRAIN)}]
    assert out["fetched_bytes"] == len(files["README.md"]) + len(TRAIN)
    assert (out["datasets"]["parts"][0]["listing"], out["datasets"]["parts"][0]["commit_from"]) == ("manifest", "hub")
    assert os.path.isfile(os.path.join(DD.state_dir(sess.ws), "org--name", C1 + ".json"))
    ev = drain(q, "direction_datasets")
    assert [e["state"] for e in ev][0] == "fetching" and ev[-1]["state"] == "fetched"
    assert {k: ev[0][k] for k in ("workspace", "phase", "file", "done_bytes", "total_bytes", "stream")} == {
        "workspace": "rnd", "phase": "list", "file": None, "done_bytes": 0, "total_bytes": 0, "stream": "ds-1"}
    assert ev[-1]["fetched"] == out["fetched"] and ev[-1]["fetched_bytes"] == out["fetched_bytes"]
    assert all(e["stream"] == "ds-1" and isinstance(e["seconds"], float) for e in ev)
    again = c.post("/api/directions/datasets", json={"prompts": hs(split="train[:2]")}).json()
    assert again["cached"] is True and again["prompts_sha256"] == out["prompts_sha256"] == typed_sha(["h0", "h1"])
    assert fake.info_calls == [(REPO, "main")] and drain(q, "direction_datasets") == []
    fake.add("org/bad", "main", C2, files)
    fake.fail["data/train-00000-of-00001.jsonl"] = RuntimeError("boom")
    r = c.post("/api/directions/datasets", json={"prompts": hs(dataset="org/bad"), "download": True})
    assert r.status_code == 400 and r.json()["error"] == (
        "DirectionError: prompts: fetching data/train-00000-of-00001.jsonl of org/bad failed (RuntimeError: boom); the "
        "files fetched before it stay in the cache")
    last = drain(q, "direction_datasets")[-1]
    assert last["state"] == "failed" and last["error"] == r.json()["error"] and "stream" not in last


def test_t32_a_stop_through_the_route(hub, tmp_path):
    sess = session_on(tmp_path)
    app = __import__("wsp.server", fromlist=["create_app"]).create_app(sess, "http://127.0.0.1:0")
    from fastapi.testclient import TestClient
    c = TestClient(app, raise_server_exceptions=False)
    q = sess.subscribe()
    fake = hub.install()
    shards = {f"data/train-0000{i}-of-00003.jsonl": J([f"s{i}"]).encode() for i in range(3)}
    card = hub_card(numbers=False)
    fake.add(REPO, "main", C1, {"README.md": card, **shards})
    assert c.post("/api/generation/stop", json={"stream": "early"}).json()["found"] is False
    out = c.post("/api/directions/datasets", json={"prompts": hs(), "download": True, "stream": "early"}).json()
    assert out == {"stopped": True, "done_bytes": 0, "total_bytes": 0, "fetched": []} and fake.fetch_calls == []
    real = fake.fetch

    def first_then_stop(*a, **kw):
        got = real(*a, **kw)
        if len(fake.fetch_calls) == 2:                 # README.md, then the first data file
            app.state.generation_stops.stop("mid")
        return got
    fake.fetch = first_then_stop
    hub.install()
    names = list(shards)
    out = c.post("/api/directions/datasets", json={"prompts": hs(), "download": True, "stream": "mid"}).json()
    assert out["stopped"] is True and [f["path"] for f in out["fetched"]] == ["README.md", names[0]]
    assert out["done_bytes"] == len(card) + len(shards[names[0]])
    assert out["total_bytes"] == len(card) + sum(len(b) for b in shards.values())
    assert fake.fetch_calls == ["README.md", names[0]]
    ev = drain(q, "direction_datasets")
    assert ev[-1]["state"] == "stopped" and ev[-1]["stream"] == "mid" and ev[-1]["fetched"] == out["fetched"]
    out = c.post("/api/directions/datasets", json={"prompts": hs(), "download": True}).json()
    assert [f["path"] for f in out["fetched"]] == names[1:] and out["count"] == 3


def test_t34_the_limits_route_publishes_heretics_defaults(tmp_path):
    c = app_client(session_on(tmp_path))
    limits = c.get("/api/directions").json()["limits"]
    assert limits["datasets"]["heretic_defaults"] == {
        "system_prompt": "You are a helpful assistant.",
        "good_prompts": {"dataset": "mlabonne/harmless_alpaca", "split": "train[:400]", "column": "text"},
        "bad_prompts": {"dataset": "mlabonne/harmful_behaviors", "split": "train[:400]", "column": "text"},
        "scorers": {"KeywordRate": {"dataset": "mlabonne/harmful_behaviors", "split": "test[:100]", "column": "text"},
                    "KLDivergence": {"dataset": "mlabonne/harmless_alpaca", "split": "test[:100]", "column": "text"},
                    "ResponseLength": {"dataset": "mlabonne/harmless_alpaca", "split": "test[:100]", "column": "text"}},
        "response_prefix_test_prompts": [
            {"dataset": "mlabonne/harmless_alpaca", "split": "train[:100]", "column": "text"},
            {"dataset": "mlabonne/harmful_behaviors", "split": "train[:100]", "column": "text"}]}
    assert limits["datasets"] == json.loads(json.dumps(DD.LIMITS))
    assert (limits["datasets"]["specifications_per_set"], limits["datasets"]["download_bytes"],
            limits["datasets"]["listed_files"]) == (16, 4 << 30, 100_000)
    assert limits["prompts_per_set"] == 1024 and "scores" in limits and "search" in limits
    assert DD.DATASET_FIELDS == ("prompts", "system", "download", "texts")
