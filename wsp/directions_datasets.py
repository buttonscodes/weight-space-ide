"""Prompt sets from datasets: Heretic's dataset specifications, resolved by the IDE's own code.

A prompt field of a behaviour-direction request (``good``/``bad`` of a measurement, a scorer's ``prompts``, the
prefix detection's ``prompts``) takes a list of texts as before, one dataset specification (Heretic's eight fields:
``dataset``, ``commit``, ``config``, ``split``, ``column``, ``prefix``, ``suffix``, ``system_prompt``), or a list
whose items are texts, prompt objects ``{"user", "system"}`` and specifications, expanded in order.

The sources are Heretic's (heretic/src/heretic/utils.py ``_load_prompts_single`` at 64e9991), with its branching:
an existing local file is plain text, one prompt per stripped non-empty line; a local directory holding
``state.json`` is a ``save_to_disk`` dataset; any other local directory is read the way ``datasets.load_dataset``
reads it (the dataset card's ``configs``, else datasets' default file patterns, the format by datasets' extension
vote).  The split grammar is datasets' (``arrow_reader.py``).  Parquet and arrow files need ``pyarrow``
(requirements-datasets.txt), imported lazily; json, csv, tsv and txt files are read with the standard library.

Resolution only reads files.  A refusal is a ``DatasetError`` carrying its final message; the callers re-raise it
as ``DirectionError``.  No message quotes the content of a file: messages name the file, the row or line, and the
type.

Module-level imports are the standard library only, so the module imports where ``wsp`` is not installed;
``fsspec``, ``yaml``, ``pyarrow`` and ``huggingface_hub`` are imported inside the functions that need them.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import math
import os
import re
import time

VERSION = 1
FIELDS = ("dataset", "commit", "config", "split", "column", "prefix", "suffix", "system_prompt")
PROMPT_FIELDS = ("user", "system")
FIELD_CHARS = {"dataset": 4096, "commit": 256, "config": 256, "split": 1024, "column": 256}
MAX_PROMPTS = 1024             # per set (equal to wsp.directions.MAX_PROMPTS)
MAX_CHARS = 20_000             # per text (equal to wsp.directions.MAX_CHARS)
MAX_SPECS = 16                 # dataset specifications per prompt set
MAX_FILE_BYTES = 2 << 30       # one data file
MAX_SPLIT_FILES = 1000         # data files per specification
MAX_TEXT_BYTES = 256 << 20     # json, csv and txt bytes parsed per specification
MAX_LISTED_FILES = 100_000     # files a directory walk lists before it refuses
MAX_DOWNLOAD_BYTES = 4 << 30   # bytes one fetch request downloads
DISK_RESERVE = 2 << 30         # bytes that stay free on the cache's file system (equal to wsp.directions.DISK_RESERVE)
PREVIEW = 3
REQUIREMENTS = "requirements-datasets.txt"
HASH_BLOCK = 1 << 20

# datasets' extension table (packaged_modules/__init__.py): the readers the IDE has, and what it refuses
FORMATS = {".parquet": "parquet", ".geoparquet": "parquet", ".gpq": "parquet", ".arrow": "arrow",
           ".json": "json", ".jsonl": "json", ".ndjson": "json", ".csv": "csv", ".tsv": "csv", ".txt": "text"}
PYARROW_FORMATS = ("parquet", "arrow")
REFUSED_EXTENSIONS = (".conll", ".conllu", ".tar", ".xml", ".hdf5", ".h5", ".eval", ".lance", ".tsfile", ".zip",
                      ".gz", ".bz2", ".xz", ".zst", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif",
                      ".tiff", ".wav", ".mp3", ".flac", ".ogg", ".mp4", ".avi", ".mov", ".mkv", ".pdf")
COMPRESSION = (".gz", ".bz2", ".xz", ".zst", ".lz4", ".zip")
METADATA_FILES = ("metadata.csv", "metadata.jsonl", "metadata.parquet")
IGNORED_FILES = ("README.md", "config.json", "dataset_info.json", "dataset_infos.json", "dummy_data.zip",
                 "dataset_dict.json")
CARD_KEYS = ("config_name", "data_files", "data_dir", "default", "features")
READS = "parquet, arrow, json, jsonl, ndjson, csv, tsv and txt"

# datasets' default file patterns (data_files.py, the fsspec >= 2023.12 branch), tried in order
KEYWORDS = {"train": ["train", "training"], "validation": ["validation", "valid", "dev", "val"],
            "test": ["test", "testing", "eval", "evaluation"]}
SEP = "-._ 0-9"
SHARDED = "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"
_DIRNAME = ("**/{kw}/**", "**/{kw}[{sep}]*/**", "**/*[{sep}]{kw}/**", "**/*[{sep}]{kw}[{sep}]*/**")
_FILENAME = ("**/{kw}[{sep}]*", "**/*[{sep}]{kw}[{sep}]*")
INFER = [SHARDED,
         {"logs": ["**/*.eval"]},
         {s: [p.format(kw=kw, sep=SEP) for kw in kws for p in _DIRNAME] for s, kws in KEYWORDS.items()},
         {s: [p.format(kw=kw, sep=SEP) for kw in kws for p in _FILENAME] for s, kws in KEYWORDS.items()},
         {"train": ["**"]}]

HERETIC_DEFAULTS = {
    "system_prompt": "You are a helpful assistant.",
    "good_prompts": {"dataset": "mlabonne/harmless_alpaca", "split": "train[:400]", "column": "text"},
    "bad_prompts": {"dataset": "mlabonne/harmful_behaviors", "split": "train[:400]", "column": "text"},
    "scorers": {"KeywordRate": {"dataset": "mlabonne/harmful_behaviors", "split": "test[:100]", "column": "text"},
                "KLDivergence": {"dataset": "mlabonne/harmless_alpaca", "split": "test[:100]", "column": "text"},
                "ResponseLength": {"dataset": "mlabonne/harmless_alpaca", "split": "test[:100]", "column": "text"}},
    "response_prefix_test_prompts": [{"dataset": "mlabonne/harmless_alpaca", "split": "train[:100]", "column": "text"},
                                     {"dataset": "mlabonne/harmful_behaviors", "split": "train[:100]", "column": "text"}],
}

LIMITS = {"fields": list(FIELDS), "formats": sorted(FORMATS), "specifications_per_set": MAX_SPECS,
          "file_bytes": MAX_FILE_BYTES, "files_per_specification": MAX_SPLIT_FILES, "text_bytes": MAX_TEXT_BYTES,
          "listed_files": MAX_LISTED_FILES, "download_bytes": MAX_DOWNLOAD_BYTES, "disk_reserve": DISK_RESERVE,
          "heretic_defaults": HERETIC_DEFAULTS}


class DatasetError(ValueError):
    """A refusal with its final message (callers re-raise it as DirectionError)."""


class NotCached(DatasetError):
    """A Hub dataset that is not (completely) in the local cache; ``would_fetch`` says what a fetch would get."""

    def __init__(self, message, would_fetch):
        super().__init__(message)
        self.would_fetch = would_fetch


class _BadPart(DatasetError):
    def __init__(self, part, message):
        super().__init__(message)
        self.part = part


class _UnknownSplit(DatasetError):
    def __init__(self, name, splits):
        super().__init__(f"no split {name!r}; the splits are {', '.join(splits)}")
        self.name, self.splits = name, list(splits)


class _SmallSplit(DatasetError):
    def __init__(self, name, n):
        super().__init__(f"(pct1_dropremainder) needs a split of at least 100 rows; {name} has {n}")
        self.name, self.n = name, n


# ------------------------------------------------------------------ the split grammar (datasets.arrow_reader)
_SUB = re.compile(r"^(?P<split>\w+(\.\w+)*)(\[((?P<from>-?[\d_]+)(?P<from_pct>%)?)?:((?P<to>-?[\d_]+)(?P<to_pct>%)?)?\])?"
                  r"(\((?P<rounding>[^\)]*)\))?$")
_PLUS = re.compile(r"\s*\+\s*")
_ROUNDINGS = ("closest", "pct1_dropremainder")


def _part_text(part):
    return (f"{part!r} is not a split instruction: a split name, optionally with a slice such as [:400], [10:20], "
            f"[-50:] or [:10%], parts joined by +")


def parse_split(text) -> list[tuple[str, int | None, int | None, str, str | None]]:
    """A split string -> its parts ``(name, from, to, unit, rounding)``, unit ``"%"`` or ``"abs"``; rounding
    ``closest`` (the default with ``%``) or ``pct1_dropremainder``, None for absolute parts.  Refusals are
    DatasetError with a message that names no field."""
    out = []
    for part in _PLUS.split(text):
        m = _SUB.fullmatch(part)
        if not m:
            raise _BadPart(part, _part_text(part))
        unit = "%" if m.group("from_pct") or m.group("to_pct") else "abs"
        try:
            a = int(m.group("from")) if m.group("from") else None
            b = int(m.group("to")) if m.group("to") else None
        except ValueError:
            raise _BadPart(part, _part_text(part)) from None
        rounding = m.group("rounding")
        if rounding is not None and (rounding not in _ROUNDINGS or unit != "%"):
            raise DatasetError("rounding is (closest) or (pct1_dropremainder), and only with percent slices")
        if unit == "%" and any(x is not None and abs(x) > 100 for x in (a, b)):
            raise DatasetError("percent bounds must be from -100% to 100%")
        if unit == "%" and rounding is None:
            rounding = "closest"
        out.append((m.group("split"), a, b, unit, rounding))
    first = out[0]
    if first[3] == "%" and any(p[3] == "%" and p[4] != first[4] for p in out[1:]):
        raise DatasetError("every percent part must use the same rounding")
    return out


def absolute(parts, lengths: dict[str, int]) -> list[tuple[str, int, int]]:
    """Parts and the row count of each split -> ``(name, from, to)`` row ranges (``rows[from:to]``)."""
    out = []
    for name, a, b, unit, rounding in parts:
        if name not in lengths:
            raise _UnknownSplit(name, lengths)
        n = lengths[name]
        if unit == "%":
            if rounding == "pct1_dropremainder":
                if n < 100:
                    raise _SmallSplit(name, n)
                step = math.trunc(n / 100.0)

                def conv(x, step=step):
                    return x * step
            else:
                def conv(x, n=n):
                    return int(round(x * n / 100.0))
            a = 0 if a is None else conv(a)
            b = n if b is None else conv(b)
        else:
            a = 0 if a is None else a
            b = n if b is None else b
        if a < 0:
            a = max(n + a, 0)
        if b < 0:
            b = max(n + b, 0)
        out.append((name, min(a, n), min(b, n)))
    return out


# ------------------------------------------------------------------ prompts, the canonical form, hashes
def system_of(p, system):
    """A prompt's system message: the request's for a text, its own for an object."""
    return system if isinstance(p, str) else p.get("system", system)


def user_of(p):
    return p if isinstance(p, str) else p["user"]


def canonical(prompts, system) -> list:
    """The canonical form of resolved prompts under the request's system message ``system``: a prompt whose system
    message is the request's is the bare text, any other is ``{"system", "user"}`` (keys in that order)."""
    out = []
    for p in prompts:
        own = system_of(p, system)
        out.append(user_of(p) if own == system else {"system": own, "user": user_of(p)})
    return out


def prompts_sha256(canonical) -> str:
    from . import directions as DR
    return DR._sha(canonical)


def state_dir(ws) -> str:
    """The workspace's dataset state (manifests and refs of Hub datasets)."""
    from . import directions as DR
    return os.path.join(ws.dir, DR.DIRECTORY, "datasets")


# ------------------------------------------------------------------ the shape of a value (5.1, 5.2)
def _text(value, lo, hi):
    return isinstance(value, str) and lo <= len(value) <= hi


def _check_prompt(item, fk):
    for key in item:
        if key not in PROMPT_FIELDS:
            raise DatasetError(f"{fk}.{key}: unknown field; a prompt has user and system")
    if not _text(item["user"], 1, MAX_CHARS):
        raise DatasetError(f"{fk}.user must be text of 1..{MAX_CHARS} characters")
    if "system" in item and not _text(item["system"], 0, MAX_CHARS):
        raise DatasetError(f"{fk}.system must be text of at most {MAX_CHARS} characters")
    return {k: item[k] for k in PROMPT_FIELDS if k in item}


def _check_spec(item, fk):
    for key in item:
        if key not in FIELDS:
            raise DatasetError(f"{fk}.{key}: unknown field; the fields of a dataset specification are "
                               f"{', '.join(FIELDS)}")
    if not _text(item["dataset"], 1, FIELD_CHARS["dataset"]):
        raise DatasetError(f"{fk}.dataset must be text of 1..{FIELD_CHARS['dataset']} characters")
    if any(c < " " or c == "\x7f" for c in item["dataset"]):
        raise DatasetError(f"{fk}.dataset must be text of 1..{FIELD_CHARS['dataset']} characters without control "
                           f"characters")
    for key in ("commit", "config", "split", "column"):
        v = item.get(key)
        if v is not None and not _text(v, 1, FIELD_CHARS[key]):
            raise DatasetError(f"{fk}.{key} must be text of 1..{FIELD_CHARS[key]} characters or null")
    for key in ("prefix", "suffix"):
        if key in item and not _text(item[key], 0, MAX_CHARS):
            raise DatasetError(f"{fk}.{key} must be text of at most {MAX_CHARS} characters")
    v = item.get("system_prompt")
    if v is not None and not _text(v, 0, MAX_CHARS):
        raise DatasetError(f"{fk}.system_prompt must be text of at most {MAX_CHARS} characters or null")
    return {"dataset": item["dataset"], "commit": item.get("commit"), "config": item.get("config"),
            "split": item.get("split"), "column": item.get("column"), "prefix": item.get("prefix", ""),
            "suffix": item.get("suffix", ""), "system_prompt": item.get("system_prompt")}


def items(value, label, field) -> list | None:
    """5.1: None for the old path (a list without objects, or anything that is neither a list nor an object: the
    caller then uses wsp.directions._texts, with today's checks and messages); else the checked items as
    ``(path, item)`` pairs: texts as given, prompt objects as {"user": u} or {"user": u, "system": s} (an absent
    system stays absent and means the request's), specifications normalized to the eight fields.  Shape only:
    nothing on disk is looked at."""
    if isinstance(value, dict):
        raw = [(field, value)]
    elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
        if len(value) > MAX_PROMPTS:
            raise DatasetError(f"{label} must list 1..{MAX_PROMPTS} prompts; the list has {len(value):,} items")
        raw = [(f"{field}[{k}]", v) for k, v in enumerate(value)]
    else:
        return None
    out, specs = [], 0
    for fk, v in raw:
        if isinstance(v, str):
            out.append((fk, v))
        elif isinstance(v, dict):
            if "user" in v and "dataset" in v:
                raise DatasetError(f"{fk}: an object must be a dataset specification (with dataset) or a prompt "
                                   f"(with user), not both")
            if "user" in v:
                out.append((fk, _check_prompt(v, fk)))
            elif "dataset" in v:
                specs += 1
                if specs > MAX_SPECS:
                    raise DatasetError(f"{field}: at most {MAX_SPECS} dataset specifications per prompt set")
                out.append((fk, _check_spec(v, fk)))
            else:
                raise DatasetError(f"{fk}: an object must be a dataset specification (with dataset) or a prompt "
                                   f"(with user)")
        else:
            raise DatasetError(f"{fk} must be a text, a prompt object (user, system) or a dataset specification "
                               f"(dataset, ...)")
    return out


# ------------------------------------------------------------------ files
def _inside(path) -> bool:
    """A relative POSIX path that stays inside its root (no absolute path, no empty or '..' part)."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path:
        return False
    parts = path.split("/")
    return all(p and p != ".." for p in parts)


def _check_inside(fk, source, path):
    if not _inside(path):
        raise DatasetError(f"{fk}: {source} lists {path!r}, which is not a path inside the dataset")


def _size(path) -> int:
    return os.stat(path).st_size


def walk(root, fk, source) -> list[str]:
    """Every regular file under ``root`` (links followed, as datasets follows them), as sorted relative POSIX
    paths.  A directory already visited (same device and inode) is not entered again; more than
    MAX_LISTED_FILES files is a refusal, checked while walking."""
    out, seen = [], set()
    try:
        st = os.stat(root)
        seen.add((st.st_dev, st.st_ino))
    except OSError:
        pass
    for here, dirs, files in os.walk(root, followlinks=True):
        keep = []
        for d in sorted(dirs):
            try:
                st = os.stat(os.path.join(here, d))
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            keep.append(d)
        dirs[:] = keep
        for f in files:
            full = os.path.join(here, f)
            if not os.path.isfile(full):
                continue
            out.append(os.path.relpath(full, root).replace(os.sep, "/"))
            if len(out) > MAX_LISTED_FILES:
                raise DatasetError(f"{fk}: {source} holds more than {MAX_LISTED_FILES:,} files; give the directory "
                                   f"that holds only the dataset")
    return sorted(out)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(HASH_BLOCK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _basename(path):
    return path.rsplit("/", 1)[-1]


def _suffixes(path):
    return ["." + s.lower() for s in _basename(path).split(".")[1:]]


def _magic_re(pattern):
    from fsspec.utils import glob_translate
    return re.compile(glob_translate(pattern))


def match(pattern, files) -> list[str]:
    """datasets' resolve_pattern over a file list: the glob match, then the ignored names (minus the pattern's own
    basename), unrequested ``__special__`` directories and unrequested hidden parts dropped; sorted."""
    rx = _magic_re(pattern)
    ignore = set(IGNORED_FILES) - {_basename(pattern)}
    pat_parts = pattern.split("/")
    pat_special = sum(1 for p in pat_parts[:-1] if p.startswith("__"))
    pat_hidden = sum(1 for p in pat_parts if p.startswith(".") and set(p) != {"."})
    out = []
    for f in files:
        if not rx.fullmatch(f) or _basename(f) in ignore:
            continue
        parts = f.split("/")
        if sum(1 for p in parts[:-1] if p.startswith("__")) != pat_special:
            continue
        if sum(1 for p in parts if p.startswith(".") and set(p) != {"."}) != pat_hidden:
            continue
        out.append(f)
    return sorted(out)


def infer(files) -> dict[str, list[str]]:
    """datasets' default patterns for a file list: {split: [pattern, ...]}, or {} when nothing matches."""
    sharded = SHARDED.replace("{split}", "*")
    hits = match(sharded, files)
    if hits:
        names = {_basename(p).split("-")[0] for p in hits}
        order = [s for s in ("train", "validation", "test") if s in names]
        order += sorted(names - {"train", "validation", "test"})
        return {s: [SHARDED.format(split=s)] for s in order}
    for table in INFER[1:]:
        found = [s for s, patterns in table.items() if any(match(p, files) for p in patterns)]
        if found:
            return {s: list(table[s]) for s in found}
    return {}


def vote(paths):
    """datasets' format vote over one split's files -> the deciding extension, or None (no known extension)."""
    from collections import Counter
    counts = Counter((ext, _basename(p) in METADATA_FILES) for p in paths for ext in _suffixes(p))

    def key(item):
        (ext, is_metadata), n = item
        return (not is_metadata, n, ext == ".parquet", ext == ".lance", ext == ".arrow", ext == ".jsonl",
                ext == ".json", ext == ".csv", ext)
    for (ext, _), _ in sorted(counts.items(), key=key, reverse=True):
        if ext in FORMATS or ext in REFUSED_EXTENSIONS:
            return ext
    return None


# ------------------------------------------------------------------ the dataset card
_CARD = re.compile(r"^(\s*---[\r\n]+)([\S\s]*?)([\r\n]+---(\r\n|\n|$))")


def _yaml_error(e):
    problem = getattr(e, "problem", None)
    mark = getattr(e, "problem_mark", None)
    if problem and mark is not None:
        return f"{problem} at line {mark.line + 1}, column {mark.column + 1}"
    return type(e).__name__


def _card(fk, source, path):
    """The YAML front matter of a dataset card as a dict ({} when there is none)."""
    try:
        size = _size(path)
        if size > MAX_TEXT_BYTES:
            raise DatasetError(f"{fk}: the dataset card (README.md) of {source} cannot be read (it takes "
                               f"{size:,} bytes)")
        with open(path, "rb") as f:
            data = f.read(MAX_TEXT_BYTES + 1)
        if len(data) > MAX_TEXT_BYTES:
            raise DatasetError(f"{fk}: the dataset card (README.md) of {source} cannot be read (it takes more than "
                               f"{MAX_TEXT_BYTES:,} bytes)")
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DatasetError(f"{fk}: the dataset card (README.md) of {source} cannot be read (it is not valid "
                           f"UTF-8)") from None
    except OSError as e:
        raise DatasetError(f"{fk}: the dataset card (README.md) of {source} cannot be read "
                           f"({e.strerror or type(e).__name__})") from None
    m = _CARD.match(text)
    if not m:
        return {}
    import yaml
    try:
        meta = yaml.safe_load(m.group(2))
    except yaml.YAMLError as e:
        raise DatasetError(f"{fk}: the dataset card (README.md) of {source} cannot be read ({_yaml_error(e)})") \
            from None
    return meta if isinstance(meta, dict) else {}


def _patterns_of(fk, source, name, entry):
    """One card config's data_files -> {split: [pattern]} (None: no data_files given)."""
    df = entry.get("data_files")
    bad = DatasetError(f"{fk}: the dataset card of {source} gives data_files for config {name!r} in a form datasets "
                       f"does not accept either")
    if df is None:
        return None
    if isinstance(df, str):
        return {"train": [df]}
    if not isinstance(df, list) or not df:
        raise bad
    if all(isinstance(x, str) for x in df):
        return {"train": list(df)}
    out = {}
    for x in df:
        if not isinstance(x, dict) or set(x) != {"split", "path"} or not isinstance(x["split"], str):
            raise bad
        p = x["path"]
        if isinstance(p, str):
            p = [p]
        if not isinstance(p, list) or not p or not all(isinstance(y, str) for y in p) or x["split"] in out:
            raise bad
        out[x["split"]] = list(p)
    return out


def configs(fk, source, files, meta) -> tuple[dict, list[str]]:
    """The configs of a source -> ({name: {split: [pattern]}}, the names of the default candidates)."""
    cards = meta.get("configs")
    if not cards:
        return {"default": infer(files)}, ["default"]
    if not isinstance(cards, list):
        raise DatasetError(f"{fk}: the dataset card of {source} lists a config without a config_name")
    out, defaults = {}, []
    for entry in cards:
        if not isinstance(entry, dict) or not isinstance(entry.get("config_name"), str):
            raise DatasetError(f"{fk}: the dataset card of {source} lists a config without a config_name")
        name = entry["config_name"]
        for key in entry:
            if key not in CARD_KEYS:
                raise DatasetError(f"{fk}: the dataset card of {source} sets {key!r} for config {name!r}, which "
                                   f"changes how the files are read; the IDE reads config_name, data_files, data_dir, "
                                   f"default and features")
        data_dir = entry.get("data_dir")
        if data_dir is not None and not isinstance(data_dir, str):
            raise DatasetError(f"{fk}: the dataset card of {source} gives data_files for config {name!r} in a form "
                               f"datasets does not accept either")
        patterns = _patterns_of(fk, source, name, entry)
        if data_dir:
            base = data_dir.rstrip("/") + "/"
            if patterns is None:
                patterns = infer([f[len(base):] for f in files if f.startswith(base)])
            patterns = {s: [base + p for p in ps] for s, ps in patterns.items()}
        elif patterns is None:
            patterns = infer(files)
        out[name] = patterns
    for entry in cards:
        name = entry["config_name"]
        if len(cards) == 1 or name == "default" or entry.get("default"):
            defaults.append(name)
    return out, defaults


# ------------------------------------------------------------------ reading one source
_MISSING = object()


class _Source:
    """What one specification reads: its path in messages, its name, the bytes parsed as text, the files read."""

    def __init__(self, fk, spec, name):
        self.fk, self.spec, self.name = fk, spec, name
        self.column = spec["column"]
        self.text_bytes = 0
        self.files = {}           # relative path -> {"path", "bytes", "sha256"}

    def err(self, message):
        return DatasetError(f"{self.fk}{message}")

    def stat(self, full, rel):
        try:
            size = _size(full)
        except OSError as e:
            raise self.err(f": {rel} cannot be read ({e.strerror or type(e).__name__})") from None
        if size > MAX_FILE_BYTES:
            raise self.err(f": {rel} is {size:,} bytes; the IDE reads data files of at most {MAX_FILE_BYTES:,} bytes")
        return size

    def _too_much_text(self, total, whole_file):
        if whole_file:
            return self.err(f": this text file takes {total:,} bytes; the IDE parses at most {MAX_TEXT_BYTES:,} "
                            f"bytes of them per specification (parquet has no such limit)")
        return self.err(f": the json, csv and txt files of this specification take {total:,} bytes; the IDE parses "
                        f"at most {MAX_TEXT_BYTES:,} bytes of them per specification (parquet has no such limit)")

    def read_text_bytes(self, full, rel, whole_file=False):
        """A text file's bytes, counted under MAX_TEXT_BYTES: the size is checked before the file is opened, and the
        read never takes more than the remaining allowance plus one byte."""
        size = self.stat(full, rel)
        remaining = MAX_TEXT_BYTES - self.text_bytes
        if size > remaining:
            raise self._too_much_text(self.text_bytes + size, whole_file)
        try:
            with open(full, "rb") as f:
                data = f.read(remaining + 1)
        except OSError as e:
            raise self.err(f": {rel} cannot be read ({e.strerror or type(e).__name__})") from None
        if len(data) > remaining:
            raise self._too_much_text(self.text_bytes + len(data), whole_file)
        self.text_bytes += len(data)
        self.files[rel] = {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def hash_file(self, full, rel):
        if rel not in self.files:
            try:
                self.files[rel] = {"path": rel, "bytes": _size(full), "sha256": _sha256_file(full)}
            except OSError as e:
                raise self.err(f": {rel} cannot be read ({e.strerror or type(e).__name__})") from None

    def decode(self, data, rel, encoding="utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            raise self.err(f": {rel} is not valid UTF-8 text") from None


def _pyarrow(src, fmt):
    try:
        import pyarrow  # noqa: F401
        import pyarrow.ipc  # noqa: F401
        if fmt == "parquet":
            import pyarrow.parquet  # noqa: F401
    except ImportError:
        raise src.err(f": reading {fmt} files needs pyarrow, which is not installed here: python -m pip install -r "
                      f"{REQUIREMENTS}") from None
    import pyarrow as pa
    return pa


class _Rows:
    """One file's rows of the column: ``length``, then ``cells(a, b)`` -> [(value, row_in_file)]."""
    length = 0

    def cells(self, a, b):
        raise NotImplementedError


class _ListRows(_Rows):
    def __init__(self, values):
        self.values = values
        self.length = len(values)

    def cells(self, a, b):
        return [(self.values[i], i + 1) for i in range(a, b)]


def _json_rows(src, full, rel, column):
    text = src.decode(src.read_text_bytes(full, rel), rel, "utf-8-sig")
    shape = DatasetError(f"{src.fk}: {rel} must hold a JSON list of objects or one JSON object per line")

    def bad(e, line):
        return DatasetError(f"{src.fk}: {rel} line {line} is not valid JSON ({e})")

    def load(s, line):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise bad(e, line - 1 + e.lineno) from None
        except RecursionError:
            raise bad("nested too deeply", line) from None
        except ValueError as e:
            raise bad(e, line) from None
    if text.lstrip().startswith("["):
        rows = load(text, 1)
        if not isinstance(rows, list):
            raise shape
    else:
        rows = []
        for n, line in enumerate(text.split("\n"), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, RecursionError) as e:
                try:                  # one JSON value over several lines (datasets reads the whole file then)
                    whole = json.loads(text)
                except (ValueError, RecursionError):
                    whole = _MISSING
                if whole is not _MISSING:
                    raise shape from None
                if isinstance(e, json.JSONDecodeError):
                    raise bad(e, n) from None
                raise bad("nested too deeply" if isinstance(e, RecursionError) else e, n) from None
    values = []
    for r in rows:
        if not isinstance(r, dict):
            raise shape
        values.append(r.get(column, _MISSING))
    return _ListRows(values)


def _csv_rows(src, full, rel, column, delimiter, split):
    text = src.decode(src.read_text_bytes(full, rel), rel)
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        names = reader.fieldnames or []
        if column not in names:
            raise DatasetError(f"{src.fk}.column: split {split} of {src.name} has no column {column!r}; its columns "
                               f"are {', '.join(map(str, names))}")
        values = [row.get(column) for row in reader]
    except csv.Error as e:
        raise DatasetError(f"{src.fk}: {rel} line {reader.line_num} is not valid csv ({e})") from None
    return _ListRows(values)


def _text_rows(src, full, rel):
    text = src.decode(src.read_text_bytes(full, rel), rel)
    text = text.replace("\r\n", "\n").replace("\r", "\n")      # open()'s universal newlines
    return _ListRows([line.rstrip("\n") for line in io.StringIO(text).readlines()])


class _ParquetRows(_Rows):
    def __init__(self, src, full, rel, column, split):
        pa = _pyarrow(src, "parquet")
        import pyarrow.parquet as pq
        src.stat(full, rel)
        self.full, self.rel, self.column, self.src = full, rel, column, src
        try:
            pf = pq.ParquetFile(full)
            try:
                names = pf.schema_arrow.names
                self.groups = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
            finally:
                pf.close()
        except (pa.ArrowException, OSError) as e:
            raise _unreadable(src, rel, "parquet", e) from None
        if column not in names:
            raise DatasetError(f"{src.fk}.column: split {split} of {src.name} has no column {column!r}; its columns "
                               f"are {', '.join(names)}")
        self.length = sum(self.groups)

    def cells(self, a, b):
        import pyarrow as pa
        import pyarrow.parquet as pq
        out = []
        try:
            pf = pq.ParquetFile(self.full)
            try:
                start = 0
                for i, n in enumerate(self.groups):
                    lo, hi = max(a, start), min(b, start + n)
                    if lo < hi:
                        values = pf.read_row_group(i, columns=[self.column]).column(0).to_pylist()
                        out += [(values[j - start], j + 1) for j in range(lo, hi)]
                    start += n
            finally:
                pf.close()
        except (pa.ArrowException, OSError) as e:
            raise _unreadable(self.src, self.rel, "parquet", e) from None
        return out


def _open_arrow(pa, source):
    try:
        reader = pa.ipc.open_stream(source)
        return reader.schema, list(reader)
    except pa.ArrowInvalid:
        source.seek(0)
        reader = pa.ipc.open_file(source)
        return reader.schema, [reader.get_batch(i) for i in range(reader.num_record_batches)]


class _ArrowRows(_Rows):
    def __init__(self, src, full, rel, column, split):
        pa = _pyarrow(src, "arrow")
        src.stat(full, rel)
        self.full, self.rel, self.column, self.src = full, rel, column, src
        try:
            with pa.memory_map(full) as source:
                schema, batches = _open_arrow(pa, source)
                self.batches = [b.num_rows for b in batches]
                del batches
        except (pa.ArrowException, OSError) as e:
            raise _unreadable(src, rel, "arrow", e) from None
        if column not in schema.names:
            raise DatasetError(f"{src.fk}.column: split {split} of {src.name} has no column {column!r}; its columns "
                               f"are {', '.join(schema.names)}")
        self.length = sum(self.batches)

    def cells(self, a, b):
        import pyarrow as pa
        out = []
        try:
            with pa.memory_map(self.full) as source:
                schema, batches = _open_arrow(pa, source)
                index = schema.names.index(self.column)
                start = 0
                for batch in batches:
                    n = batch.num_rows
                    lo, hi = max(a, start), min(b, start + n)
                    if lo < hi:
                        values = batch.column(index).slice(lo - start, hi - lo).to_pylist()
                        out += [(v, j + 1) for v, j in zip(values, range(lo, hi))]
                    start += n
                del batches
        except (pa.ArrowException, OSError) as e:
            raise _unreadable(self.src, self.rel, "arrow", e) from None
        return out


def _unreadable(src, rel, fmt, e):
    return DatasetError(f"{src.fk}: {rel} cannot be read as {fmt} ({type(e).__name__})")


def _select(readers, ranges, budget):
    """The selected cells of one specification, in order -> (count, [(value, split, file, row_in_file)] or None
    when the count is over ``budget``: then no cell is read)."""
    count = sum(max(0, b - a) for _, a, b in ranges)
    if count > budget:
        return count, None
    out = []
    for name, a, b in ranges:
        start = 0
        for rel, rows in readers[name]:
            lo, hi = max(a, start), min(b, start + rows.length)
            if lo < hi:
                out += [(v, name, rel, r) for v, r in rows.cells(lo - start, hi - start)]
            start += rows.length
    return count, out


def _check_cells(src, cells):
    """Selected cells -> their texts (E42 for a json row without the column, E43 for a cell that is not text)."""
    out = []
    for i, (v, split, rel, r) in enumerate(cells):
        if v is _MISSING:
            raise DatasetError(f"{src.fk}.column: row {r} of {rel} has no {src.column!r}")
        if not isinstance(v, str):
            raise DatasetError(f"{src.fk}: row {i + 1} of split {split} of {src.name} is not text in column "
                               f"{src.column!r} (it is {type(v).__name__})")
        out.append((v, split))
    return out


def _split_part(src, text):
    try:
        return parse_split(text)
    except _BadPart as e:
        raise DatasetError(f"{src.fk}.split: {e}") from None
    except DatasetError as e:
        raise DatasetError(f"{src.fk}.split: {e}") from None


def _absolute(src, parts, lengths, config=None, display=None):
    try:
        return absolute(parts, lengths)
    except _UnknownSplit as e:
        where = f"{src.name} (config {config})" if config is not None else src.name
        raise DatasetError(f"{src.fk}.split: {where} has no split {e.name!r}; its splits are "
                           f"{', '.join(e.splits)}") from None
    except _SmallSplit as e:
        what = display(e.name) if display else f"{e.name} of {src.name}"
        raise DatasetError(f"{src.fk}.split: (pct1_dropremainder) needs a split of at least 100 rows; {what} has "
                           f"{e.n}") from None


def _one_part(src, text, what):
    if len(_PLUS.split(text)) != 1:
        raise DatasetError(f"{src.fk}.split: {what} takes one slice, not parts joined by + (Heretic would read only "
                           f"the first part)")


def _text_file(src, path, budget):
    """Heretic's own branch: one prompt per stripped non-empty line; the split optional, one part."""
    rel = os.path.basename(path)
    split = src.spec["split"]
    if split is not None:
        _one_part(src, split, "a plain text file")
        try:
            parts = parse_split("_" + split)
        except _BadPart:
            raise DatasetError(f"{src.fk}.split: {_part_text(split)}") from None
        except DatasetError as e:
            raise DatasetError(f"{src.fk}.split: {e}") from None
    text = src.decode(src.read_text_bytes(path, rel, whole_file=True), rel)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    n = len(lines)
    if split is None:
        a, b, name = 0, n, None
    else:
        ((_, a, b),) = _absolute(src, parts, {parts[0][0]: n}, display=lambda _: src.name)
        name = parts[0][0][1:] or None
    count = max(0, b - a)
    record = {"format": "text_file", "split": [{"split": name, "rows": n, "from": a, "to": b}], "listing": None,
              "config": None}
    if count > budget:
        return count, None, record
    return count, [(v, None) for v in lines[a:b]], record


def _saved_to_disk(src, root, budget):
    """A save_to_disk directory: state.json's arrow files in its order; one split part, its name not checked."""
    split = src.spec["split"]
    _one_part(src, split, "a dataset saved with save_to_disk")
    parts = _split_part(src, split)
    state = os.path.join(root, "state.json")
    try:
        with open(state, "rb") as f:
            data = f.read(MAX_TEXT_BYTES + 1)
    except OSError as e:
        raise src.err(f": state.json cannot be read ({e.strerror or type(e).__name__})") from None
    try:
        meta = json.loads(src.decode(data, "state.json"))
    except json.JSONDecodeError as e:
        raise DatasetError(f"{src.fk}: state.json line {e.lineno} is not valid JSON ({e})") from None
    except (ValueError, RecursionError):
        meta = None
    listed = meta.get("_data_files") if isinstance(meta, dict) else None
    if not isinstance(listed, list) or not listed or not all(
            isinstance(d, dict) and isinstance(d.get("filename"), str) for d in listed):
        raise DatasetError(f"{src.fk}: {src.name}/state.json does not list its data files (_data_files)")
    for d in listed:
        _check_inside(src.fk, src.name, d["filename"])
    name = parts[0][0]
    readers = {name: [(d["filename"], _ArrowRows(src, os.path.join(root, d["filename"]), d["filename"], src.column,
                                                 name)) for d in listed]}
    n = sum(r.length for _, r in readers[name])
    ranges = _absolute(src, parts, {name: n})
    count, cells = _select(readers, ranges, budget)
    record = {"format": "arrow", "split": [{"split": s, "rows": n, "from": a, "to": b} for s, a, b in ranges],
              "listing": None, "config": None}
    if cells is None:
        return count, None, record
    for rel, _ in readers[name]:
        src.hash_file(os.path.join(root, rel), rel)
    return count, _check_cells(src, cells), record


def _from_files(src, root, files, listing, budget, before_read=None):
    """A source read as datasets reads a directory or a Hub repository: card, config, split files, format, rows.
    ``before_read(names, paths)`` (Hub with a manifest) refuses before any data file is opened when one is not on
    disk."""
    spec = src.spec
    parts = _split_part(src, spec["split"])
    meta = {}
    if "README.md" in files:
        meta = _card(src.fk, src.name, os.path.join(root, "README.md"))
    table, defaults = configs(src.fk, src.name, files, meta)
    config = spec["config"]
    if config is not None:
        if config not in table:
            raise DatasetError(f"{src.fk}.config: {src.name} has no config {config!r}; its configs are "
                               f"{', '.join(table)}")
    elif len(defaults) > 1:
        raise DatasetError(f"{src.fk}.config: {src.name} has several default configs ({defaults[0]!r} and "
                           f"{defaults[1]!r}); give config")
    elif not defaults:
        raise DatasetError(f"{src.fk}.config: required: {src.name} has several configs ({', '.join(table)}) and none "
                           f"is the default")
    else:
        config = defaults[0]
    patterns = table[config]

    def no_data_files():
        note = ""
        if any("/" not in f and f.endswith(".py") for f in files):
            note = " (a loading script, which datasets 4 and later do not run either)"
        return DatasetError(f"{src.fk}: {src.name} has no data files{note}")
    if not patterns:
        raise no_data_files()
    names = []
    for p in parts:
        if p[0] not in names:
            names.append(p[0])
    for name in names:
        if name not in patterns:
            raise DatasetError(f"{src.fk}.split: {src.name} (config {config}) has no split {name!r}; its splits are "
                               f"{', '.join(patterns)}")
    split_files = {}
    for name in names:
        got = []
        for pattern in patterns[name]:
            hits = match(pattern, files)
            if not hits and not glob.has_magic(pattern):
                raise DatasetError(f"{src.fk}: the dataset card of {src.name} names {pattern!r}, which matches no file")
            got += hits                      # datasets concatenates without removing duplicates
        split_files[name] = got
    readers_of, voted = {}, {}
    for name in names:
        ext = vote(split_files[name])
        if ext is None:
            raise no_data_files()
        if ext not in FORMATS:
            raise DatasetError(f"{src.fk}: split {name} of {src.name} is {ext} files; the IDE reads {READS}")
        voted[name] = ext
    kinds = {name: ("tsv" if ext == ".tsv" else FORMATS[ext]) for name, ext in voted.items()}
    a = names[0]
    for b in names[1:]:
        if kinds[b] != kinds[a]:
            raise DatasetError(f"{src.fk}: the splits of {src.name} are in different formats ({a}: {kinds[a]}, "
                               f"{b}: {kinds[b]})")
    ext0 = voted[names[0]]
    fmt = FORMATS[ext0]
    exts = [e for e, r in FORMATS.items() if r == fmt]
    total_files = 0
    for name in names:
        kept = [f for f in split_files[name] if any(s in exts for s in _suffixes(f))]
        for f in kept:
            compressed = [s for s in _suffixes(f) if s in COMPRESSION]
            if compressed:
                raise DatasetError(f"{src.fk}: {f} is compressed ({compressed[-1]}); the IDE reads uncompressed files")
        split_files[name] = kept
        if len(kept) > MAX_SPLIT_FILES:
            raise DatasetError(f"{src.fk}: split {name} of {src.name} has {len(kept):,} files; at most "
                               f"{MAX_SPLIT_FILES} per specification")
        total_files += len(kept)
        if not kept:
            raise no_data_files()
    if total_files > MAX_SPLIT_FILES:
        raise DatasetError(f"{src.fk}: split {'+'.join(names)} of {src.name} has {total_files:,} files; at most "
                           f"{MAX_SPLIT_FILES} per specification")
    if fmt in PYARROW_FORMATS:
        _pyarrow(src, fmt)
    if fmt == "text" and src.column != "text":
        raise DatasetError(f"{src.fk}.column: a txt dataset has one column, 'text'")
    if before_read is not None:
        before_read(names, list(dict.fromkeys(f for name in names for f in split_files[name])))
    delimiter = "\t" if ext0 == ".tsv" else ","
    parsed, readers = {}, {}
    for name in names:
        readers[name] = []
        for rel in split_files[name]:
            if rel not in parsed:
                full = os.path.join(root, rel)
                if fmt == "json":
                    parsed[rel] = _json_rows(src, full, rel, src.column)
                elif fmt == "csv":
                    parsed[rel] = _csv_rows(src, full, rel, src.column, delimiter, name)
                elif fmt == "text":
                    parsed[rel] = _text_rows(src, full, rel)
                elif fmt == "parquet":
                    parsed[rel] = _ParquetRows(src, full, rel, src.column, name)
                else:
                    parsed[rel] = _ArrowRows(src, full, rel, src.column, name)
            readers[name].append((rel, parsed[rel]))
    lengths = {name: sum(r.length for _, r in readers[name]) for name in names}
    ranges = _absolute(src, parts, lengths, config=config)
    count, cells = _select(readers, ranges, budget)
    record = {"format": fmt, "split": [{"split": s, "rows": lengths[s], "from": a, "to": b} for s, a, b in ranges],
              "listing": listing, "config": config, "_lengths": lengths, "_meta": meta}
    if cells is None:
        return count, None, record
    for rel in parsed:
        src.hash_file(os.path.join(root, rel), rel)
    return count, _check_cells(src, cells), record


def _kind(spec, fk):
    """5.3: the kind of source and the local path (None for the Hub)."""
    d = spec["dataset"]
    if d.startswith(("/", "~")):
        p = os.path.expanduser(d) if d.startswith("~") else d
        if os.path.isfile(p):
            return "text_file", p
        if os.path.isdir(p):
            return ("saved_to_disk" if os.path.isfile(os.path.join(p, "state.json")) else "directory"), p
        raise DatasetError(f"{fk}.dataset: no file or directory at {d}")
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id
    try:
        validate_repo_id(d)
    except HFValidationError as e:
        raise DatasetError(f"{fk}.dataset: {d!r} is not a Hugging Face repository id ({e}), and local paths must be "
                           f"absolute") from None
    return "hub", None


def _spec_rows(spec, fk, state, budget, commit=None):
    """One specification -> (count, [(text, split)] or None when over ``budget``, the part record without
    first/count/system/prompts_sha256)."""
    kind, path = _kind(spec, fk)
    if spec["commit"] is not None and kind != "hub":
        raise DatasetError(f"{fk}.commit applies to Hugging Face datasets only")
    if spec["config"] is not None and kind in ("text_file", "saved_to_disk"):
        raise DatasetError(f"{fk}.config applies to datasets with configs (not to a plain text file or a save_to_disk "
                           f"directory)")
    if kind != "text_file":
        if spec["split"] is None:
            raise DatasetError(f"{fk}.split: required for datasets (it is optional only for a plain text file)")
        if spec["column"] is None:
            raise DatasetError(f"{fk}.column: required for datasets (it is ignored for a plain text file)")
    src = _Source(fk, spec, spec["dataset"])
    base = {"repository": None, "commit": None, "commit_from": None, "rows_checked": None}
    if kind == "text_file":
        count, cells, rec = _text_file(src, path, budget)
    elif kind == "saved_to_disk":
        count, cells, rec = _saved_to_disk(src, path, budget)
    elif kind == "directory":
        files = walk(path, fk, src.name)
        count, cells, rec = _from_files(src, path, files, "directory", budget)
    else:
        count, cells, rec, base = _hub(src, state, budget, commit)
    part = {"kind": kind, "specification": dict(spec), "repository": base["repository"], "commit": base["commit"],
            "commit_from": base["commit_from"], "config": rec["config"], "format": rec["format"],
            "split": rec["split"], "files": list(src.files.values()), "listing": rec["listing"],
            "rows_checked": base["rows_checked"]}
    return count, cells, part


# ------------------------------------------------------------------ the Hugging Face cache (5.7)
_HEX40 = re.compile(r"[0-9a-fA-F]{40}")


class _Incomplete(DatasetError):
    """E54: the cached files hold fewer (or more) rows than the dataset card says."""


def hub_cache_dir() -> str:
    """The Hugging Face hub cache as huggingface_hub resolves it (HF_HUB_CACHE, else HF_HOME/hub, else
    ~/.cache/huggingface/hub)."""
    from huggingface_hub import constants
    return constants.HF_HUB_CACHE


def _token():
    """Exactly the checkpoint policy: an HF_TOKEN in the IDE's environment is sent; a stored login never is."""
    return os.environ.get("HF_TOKEN") or False


def hub_info(repo, revision) -> dict:
    """The network: {"commit": sha, "files": [{"path", "bytes"}]} of a Hub dataset at a revision."""
    import huggingface_hub
    info = huggingface_hub.HfApi().dataset_info(repo, revision=revision, files_metadata=True, token=_token())
    return {"commit": info.sha, "files": [{"path": s.rfilename, "bytes": s.size} for s in (info.siblings or [])]}


def hub_fetch(repo, commit, path, *, tqdm_class=None) -> str:
    """The network: one file of a Hub dataset into the hub cache -> its local path."""
    import huggingface_hub
    return huggingface_hub.hf_hub_download(repo, path, repo_type="dataset", revision=commit, cache_dir=hub_cache_dir(),
                                           token=_token(), tqdm_class=tqdm_class)


def progress_class(callback):
    """A tqdm class for huggingface_hub that draws nothing and calls ``callback(n_bytes_added)`` on every update."""
    from tqdm.std import tqdm

    class _Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if n:
                callback(n)
            return super().update(n)
    return _Progress


def _repo_dir(repo):
    return os.path.join(hub_cache_dir(), "datasets--" + repo.replace("/", "--"))


def _state_repo(state, repo):
    return os.path.join(state, repo.replace("/", "--"))


def _read_json(path, limit=MAX_TEXT_BYTES):
    try:
        with open(path, "rb") as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            return None
        return json.loads(data.decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        return None


def _refs(state, repo) -> dict:
    if state is None:
        return {}
    got = _read_json(os.path.join(_state_repo(state, repo), "refs.json"))
    return got if isinstance(got, dict) else {}


def _manifest(state, repo, commit, fk):
    """The workspace's file list of (repository, commit) -> [{"path", "bytes"}] or None."""
    if state is None:
        return None
    path = os.path.join(_state_repo(state, repo), commit + ".json")
    if not os.path.isfile(path):
        return None
    got = _read_json(path)
    files = got.get("files") if isinstance(got, dict) else None
    if not isinstance(files, list) or not all(isinstance(f, dict) for f in files):
        raise DatasetError(f"{fk}: the workspace's file list of {repo} at {commit[:12]} cannot be read ({path})")
    out = []
    for f in files:
        _check_inside(fk, repo, f.get("path"))
        b = f.get("bytes")
        out.append({"path": f["path"], "bytes": b if isinstance(b, int) and not isinstance(b, bool) else None})
    return out


def _would(fk, repo, rev, commit, names, files=None, listed=True, known_after="the file list"):
    requests = [f"the repository's file list at {rev} (one Hub API request)"] if listed else []
    if files is None:
        requests += ["README.md", f"the data files of split {names} (names and sizes known after {known_after})"]
    else:
        requests += [f["path"] for f in files]
    return {"field": fk, "repository": repo, "revision": rev, "commit": commit, "files": files, "requests": requests}


def _cached_ref(repo, rev):
    if not _inside(rev):
        return None
    try:
        with open(os.path.join(_repo_dir(repo), "refs", rev), encoding="utf-8") as f:
            text = f.read(256).strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text if _HEX40.fullmatch(text) else None


def _commit_of(src, state, names):
    """5.7: which commit, without the network -> (commit, commit_from)."""
    repo, given = src.name, src.spec["commit"]
    if given is not None and _HEX40.fullmatch(given):
        return given, "specification"
    rev = given or "main"
    entry = _refs(state, repo).get(rev)
    commit = entry.get("commit") if isinstance(entry, dict) else None
    if not (isinstance(commit, str) and _HEX40.fullmatch(commit)):
        commit = _cached_ref(repo, rev)
    if commit:
        return commit, "cache_ref"
    would = _would(src.fk, repo, rev, None, names)
    if given is not None:
        raise NotCached(f"{src.fk}.commit: {given!r} is not a commit hash, and the cache does not say which commit it "
                        f"names; give the 40-character commit or fetch with POST /api/directions/datasets "
                        f"(\"download\": true)", would)
    snaps = os.path.join(_repo_dir(repo), "snapshots")
    try:
        cached = sorted(d[:12] for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d)))
    except OSError:
        cached = []
    raise NotCached(f"{src.fk}: {repo} has no commit given and the cache has no latest commit for it; give commit "
                    f"(cached: {', '.join(cached) or 'none'}) or fetch with POST /api/directions/datasets "
                    f"(\"download\": true), which asks the Hub for its latest commit", would)


def _missing_text(missing):
    shown = [f"{f['path']} {f['bytes']:,} bytes" if f["bytes"] is not None else f"{f['path']} of unknown size"
             for f in missing[:10]]
    return ", ".join(shown) + (", ..." if len(missing) > 10 else "")


def _card_rows(meta, config) -> dict:
    """The dataset card's num_examples per split of one config (dataset_info: a dict, or a list with config_name)."""
    infos = meta.get("dataset_info")
    entries = infos if isinstance(infos, list) else [infos] if isinstance(infos, dict) else []
    for e in entries:
        if isinstance(e, dict) and e.get("config_name", "default") == config:
            out = {}
            for s in e.get("splits") or []:
                n = s.get("num_examples") if isinstance(s, dict) else None
                if isinstance(s, dict) and isinstance(s.get("name"), str) and isinstance(n, int) \
                        and not isinstance(n, bool):
                    out[s["name"]] = n
            return out
    return {}


def _hub(src, state, budget, override):
    """A Hub dataset from the cache (no network) -> (count, cells, record, the hub fields of the part record)."""
    spec, fk, repo = src.spec, src.fk, src.name
    parts = _split_part(src, spec["split"])
    names = "+".join(dict.fromkeys(p[0] for p in parts))
    rev = spec["commit"] or "main"
    commit, commit_from = override if override is not None else _commit_of(src, state, names)
    listed = commit_from != "specification"
    snap = os.path.join(_repo_dir(repo), "snapshots", commit)
    manifest = _manifest(state, repo, commit, fk)
    if manifest is None:
        if not os.path.isdir(snap):
            raise NotCached(
                f"{fk}: {repo} at {commit[:12]} is not in the local Hugging Face cache ({hub_cache_dir()}), and nothing "
                f"is downloaded without a request; fetch it with POST /api/directions/datasets (this prompt set and "
                f"\"download\": true), which fetches the repository's file list, README.md and the files of split "
                f"{names}", _would(fk, repo, rev, commit, names))
        files, listing, sizes = walk(snap, fk, repo), "cache", {}
    else:
        files, listing = sorted(f["path"] for f in manifest), "manifest"
        sizes = {f["path"]: f["bytes"] for f in manifest}

    def missing(paths):
        return [{"path": p, "bytes": sizes.get(p)} for p in paths if not os.path.isfile(os.path.join(snap, p))]

    def not_cached(split, gone, files_known=True):
        would = _would(fk, repo, rev, commit, names, gone, listed)
        if not files_known:
            would["requests"].append(f"the data files of split {names} that are not cached (known after README.md)")
        return NotCached(f"{fk}: {repo} at {commit[:12]}: {len(gone)} file(s) of split {split} are not cached "
                         f"({_missing_text(gone)}); nothing is downloaded without a request; fetch them with "
                         f"POST /api/directions/datasets (\"download\": true)", would)
    if listing == "manifest" and "README.md" in files:
        gone = missing(["README.md"])
        if gone:
            raise not_cached(names, gone, files_known=False)

    def before_read(split_names, paths):
        if listing == "manifest":
            gone = missing(paths)
            if gone:
                raise not_cached("+".join(split_names), gone)
    count, cells, rec = _from_files(src, snap, files, listing, budget, before_read)
    want = _card_rows(rec["_meta"], rec["config"])
    checked = []
    for name, n in rec["_lengths"].items():
        if name in want:
            if want[name] != n:
                raise _Incomplete(f"{fk}: the cached files of split {name} of {repo} at {commit[:12]} hold {n:,} rows "
                                  f"and its dataset card says {want[name]:,}: files are missing from the cache (it was "
                                  f"filled by another tool); fetch them with POST /api/directions/datasets "
                                  f"(\"download\": true)")
            checked.append(name)
    rows_checked = True if checked and len(checked) == len(rec["_lengths"]) else None
    base = {"repository": repo, "commit": commit, "commit_from": commit_from, "rows_checked": rows_checked}
    return count, cells, rec, base


def _is_hub_spec(item):
    return isinstance(item, dict) and "dataset" in item and not item["dataset"].startswith(("/", "~"))


def plan(items, field, *, system, state) -> list[dict]:
    """What a fetch would get (no network): the would_fetch entries of every Hub specification that is not cached.
    Any other refusal of a specification is raised."""
    out = []
    for fk, item in items:
        if not _is_hub_spec(item):
            continue
        try:
            _spec_rows(item, fk, state, MAX_PROMPTS)
        except NotCached as e:
            out.append(e.would_fetch)
    return out


# ------------------------------------------------------------------ fetching (5.9; the only network path)
def _hub_call(field, fk, repo, rev, what, fn):
    from huggingface_hub import errors as HE
    try:
        return fn()
    except HE.OfflineModeIsEnabled:
        raise DatasetError(f"{field}: download was asked for, but HF_HUB_OFFLINE is set in the IDE's environment, so "
                           f"nothing can be downloaded") from None
    except HE.RevisionNotFoundError:
        raise DatasetError(f"{fk}: {repo} has no revision {rev!r} on the Hub") from None
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(e, HE.RepositoryNotFoundError) or (isinstance(e, HE.HfHubHTTPError) and status in (401, 403)):
            raise DatasetError(f"{fk}: the Hub refused {repo} ({status or type(e).__name__}): it does not exist or is "
                               f"private or gated; a private or gated dataset needs HF_TOKEN in the IDE's "
                               f"environment") from None
        raise DatasetError(f"{fk}: fetching {what} of {repo} failed ({type(e).__name__}: {e}); the files fetched "
                           f"before it stay in the cache") from None


def fetch(items, field, *, system, state, progress=None, stopped=None) -> dict:
    """The fetch of 5.9 (network), then the offline resolution -> {"stopped": False, "fetched": [{"repository",
    "commit", "path", "bytes"}], "fetched_bytes", "prompts": the canonical set, "datasets": the set record}, or
    {"stopped": True, "done_bytes", "total_bytes", "fetched"} when ``stopped()`` said so between two files.
    ``progress(event)`` gets {"phase": "list" | "file", "file", "done_bytes", "total_bytes"}."""
    from . import directions as DR
    if state is None:
        raise DatasetError(f"{field}: a fetch needs an open workspace (it keeps the file lists there)")
    fetched, done, total = [], [0], [0]
    commits, todo = {}, []

    def emit(phase, file, current=0):
        if progress is not None:
            progress({"phase": phase, "file": file, "done_bytes": done[0] + current, "total_bytes": total[0]})

    def halt():
        if stopped is not None and stopped():
            return {"stopped": True, "done_bytes": done[0], "total_bytes": total[0], "fetched": list(fetched)}
        return None

    def get(fk, repo, commit, path, size):
        current = [0]

        def added(n):
            current[0] += n
            emit("file", path, current[0])
        emit("file", path)
        local = _hub_call(field, fk, repo, commit, path,
                          lambda: hub_fetch(repo, commit, path, tqdm_class=progress_class(added)))
        if size is None:
            try:
                size = os.path.getsize(local)
            except (OSError, TypeError):
                size = current[0]
        done[0] += size
        fetched.append({"repository": repo, "commit": commit, "path": path, "bytes": size})
    for index, (fk, item) in enumerate(items):
        if not _is_hub_spec(item):
            continue
        _kind(item, fk)
        repo, given = item["dataset"], item["commit"]
        pinned = given is not None and bool(_HEX40.fullmatch(given))
        if pinned:
            has_manifest = _manifest(state, repo, given, fk) is not None
            try:
                _spec_rows(item, fk, state, MAX_PROMPTS)
                continue
            except (NotCached, _Incomplete):
                pass
            except DatasetError:
                if has_manifest:          # the manifest is the whole file list: the refusal stands
                    raise
        rev = given or "main"
        emit("list", None)
        info = _hub_call(field, fk, repo, rev, "the file list", lambda: hub_info(repo, rev))
        commit = info.get("commit") if isinstance(info, dict) else None
        if not (isinstance(commit, str) and _HEX40.fullmatch(commit)):
            raise DatasetError(f"{fk}: fetching the file list of {repo} failed (the Hub named no commit); the files "
                               f"fetched before it stay in the cache")
        listing = []
        for f in info.get("files") or []:
            path = f.get("path") if isinstance(f, dict) else None
            _check_inside(fk, repo, path)
            b = f.get("bytes")
            listing.append({"path": path, "bytes": b if isinstance(b, int) and not isinstance(b, bool) else None})
        now = int(time.time())
        DR._write_json(os.path.join(_state_repo(state, repo), commit + ".json"),
                       {"version": VERSION, "repository": repo, "commit": commit, "listed": now, "files": listing})
        if not pinned:
            refs = _refs(state, repo)
            refs[rev] = {"commit": commit, "listed": now}
            DR._write_json(os.path.join(_state_repo(state, repo), "refs.json"), refs)
            commits[index] = (commit, "hub")
        snap = os.path.join(_repo_dir(repo), "snapshots", commit)
        readme = [f for f in listing if f["path"] == "README.md"]
        if readme and not os.path.isfile(os.path.join(snap, "README.md")):
            size = readme[0]["bytes"]
            if size is not None and size > MAX_TEXT_BYTES:      # it would be refused after the fetch anyway
                raise DatasetError(f"{fk}: the dataset card (README.md) of {repo} cannot be read (it takes {size:,} "
                                   f"bytes)")
            stop = halt()
            if stop:
                return stop
            get(fk, repo, commit, "README.md", readme[0]["bytes"])
        todo.append((index, fk, item, repo, commit))
    needed, seen = [], set()
    for index, fk, item, repo, commit in todo:
        try:
            _spec_rows(item, fk, state, MAX_PROMPTS, commits.get(index))
        except NotCached as e:
            for f in e.would_fetch.get("files") or []:
                if (repo, commit, f["path"]) not in seen:
                    seen.add((repo, commit, f["path"]))
                    needed.append((fk, repo, commit, f["path"], f["bytes"]))
    need = sum(b or 0 for *_, b in needed)
    total[0] = done[0] + need
    if need > MAX_DOWNLOAD_BYTES:
        raise DatasetError(f"{field}: the fetch needs {need:,} bytes; one request fetches at most "
                           f"{MAX_DOWNLOAD_BYTES:,}")
    if needed:
        cache = hub_cache_dir()
        os.makedirs(cache, exist_ok=True)
        free = DR._free_bytes(cache)
        if need > free - DISK_RESERVE:
            raise DatasetError(f"{field}: the fetch needs {need:,} bytes; the cache's file system ({cache}) has "
                               f"{free:,} bytes free, of which {DISK_RESERVE:,} stay free")
    for fk, repo, commit, path, size in needed:
        stop = halt()
        if stop:
            return stop
        get(fk, repo, commit, path, size)
    prompts, record = _resolve(items, field, field, system=system, state=state, commits=commits)
    return {"stopped": False, "fetched": fetched, "fetched_bytes": done[0], "prompts": prompts, "datasets": record}


# ------------------------------------------------------------------ resolving a set
def resolve(items, label, field, *, system, state, allow_system=True) -> tuple[list, dict]:
    """Expand and read (offline only) -> (the canonical set, the set record).  ``state`` is state_dir(ws) (manifests
    and refs) or None; ``allow_system`` False (chat off) refuses a prompt with a system message of its own."""
    return _resolve(items, label, field, system=system, state=state, allow_system=allow_system)


def _resolve(items, label, field, *, system, state, allow_system=True, commits=None):
    commits = commits or {}
    groups, total = [], 0
    for index, (fk, item) in enumerate(items):
        if isinstance(item, str) or "dataset" not in item:
            if groups and groups[-1][0] == "inline":
                groups[-1][1].append((fk, item))
            else:
                groups.append(("inline", [(fk, item)]))
            total += 1
            continue
        count, cells, part = _spec_rows(item, fk, state, MAX_PROMPTS - total, commits.get(index))
        groups.append(("spec", (fk, item, count, cells, part)))
        total += count
    if not 1 <= total <= MAX_PROMPTS:
        hint = " (give a smaller split, such as train[:400])" if total > MAX_PROMPTS else ""
        raise DatasetError(f"{label} must list 1..{MAX_PROMPTS} prompts; the prompt set gives {total:,}{hint}")
    prompts, parts = [], []
    for kind, group in groups:
        first = len(prompts) + 1
        if kind == "inline":
            for fk, item in group:
                k = len(prompts) + 1
                if isinstance(item, str):
                    if not 1 <= len(item) <= MAX_CHARS:
                        raise DatasetError(f"{label} prompt {k} must be text of 1..{MAX_CHARS} characters")
                    prompts.append(item)
                    continue
                if "system" in item and not allow_system:
                    raise DatasetError(_own_system(label, k, f"{fk}.system"))
                prompts.append(dict(item))
            parts.append({"kind": "inline", "first": first, "count": len(group)})
            continue
        fk, spec, count, cells, part = group
        own = spec["system_prompt"]
        sys_msg = own if own is not None else system
        mine = []
        for r, (text, split) in enumerate(cells, 1):
            k = len(prompts) + 1
            if own is not None and not allow_system:
                raise DatasetError(_own_system(label, k, f"{fk}.system_prompt"))
            user = text
            if spec["prefix"]:
                user = f"{spec['prefix']} {user}"
            if spec["suffix"]:
                user = f"{user} {spec['suffix']}"
            if not 1 <= len(user) <= MAX_CHARS:
                where = f"row {r}" if part["kind"] == "text_file" else f"split {split} row {r}"
                raise DatasetError(f"{label} prompt {k} ({spec['dataset']}, {where}) must be text of 1..{MAX_CHARS} "
                                   f"characters")
            p = {"system": sys_msg, "user": user}
            prompts.append(p)
            mine.append(p)
        parts.append({"kind": part["kind"], "first": first, "count": count,
                      **{key: v for key, v in part.items() if key != "kind"},
                      "system": sys_msg, "prompts_sha256": prompts_sha256(canonical(mine, system)),
                      "local": part["kind"] != "hub"})
    out = canonical(prompts, system)
    record = {"version": VERSION, "count": len(out), "prompts_sha256": prompts_sha256(out), "parts": parts}
    return out, record


def _own_system(label, k, where):
    return (f"{label} prompt {k} has its own system message (from {where}); system messages apply to prompts read "
            f"as a chat message; set chat to true or leave system_prompt out")


# ------------------------------------------------------------------ POST /api/directions/datasets (7.2)
DATASET_FIELDS = ("prompts", "system", "download", "texts")


def request_body(body) -> dict:
    """The arguments of a datasets request (``stream`` is the route's); an unknown field is refused."""
    if not isinstance(body, dict):
        raise DatasetError("the request must be an object")
    unknown = sorted(set(body) - set(DATASET_FIELDS) - {"stream"}, key=str)
    if unknown:
        raise DatasetError(f"{unknown[0]}: unknown field; the fields are {', '.join(DATASET_FIELDS)} and stream")
    if "prompts" not in body:
        raise DatasetError("prompts: required")
    download, texts, system = body.get("download", False), body.get("texts", False), body.get("system")
    if not isinstance(download, bool):
        raise DatasetError("download must be true or false")
    if not isinstance(texts, bool):
        raise DatasetError("texts must be true or false")
    if system is not None and (not isinstance(system, str) or len(system) > MAX_CHARS):
        raise DatasetError(f"system must be text of at most {MAX_CHARS} characters")
    return {"prompts": body["prompts"], "system": system, "download": download, "texts": texts}


def run_request(ws, body, *, progress=None, stopped=None) -> dict:
    """The datasets route's work: resolve a prompt set offline and answer what it holds, or, when something is not
    cached, what a fetch would get; with ``download`` true, fetch it first (the only network path, 5.9).
    ``progress`` and ``stopped`` are fetch's. Refusals are DatasetError."""
    started = time.monotonic()
    args = request_body(body)
    system, state = args["system"], state_dir(ws)
    found = items(args["prompts"], "prompts", "prompts")
    fetched, fetched_bytes = [], 0
    if found is None:                   # a list of texts (or a value that is no set at all): the old path
        from . import directions as DR
        try:
            prompts, record = list(DR._texts(args["prompts"], "prompts")), None
        except DR.DirectionError as e:
            raise DatasetError(str(e)) from None
    elif args["download"]:
        got = fetch(found, "prompts", system=system, state=state, progress=progress, stopped=stopped)
        if got["stopped"]:
            return got
        prompts, record, fetched, fetched_bytes = got["prompts"], got["datasets"], got["fetched"], got["fetched_bytes"]
    else:
        try:
            prompts, record = resolve(found, "prompts", "prompts", system=system, state=state)
        except NotCached:
            return {"cached": False, "would_fetch": plan(found, "prompts", system=system, state=state), "fetched": [],
                    "fetched_bytes": 0, "seconds": round(time.monotonic() - started, 3)}
    out = {"cached": True, "count": len(prompts), "prompts_sha256": prompts_sha256(prompts), "datasets": record,
           "preview": prompts[:PREVIEW], "fetched": fetched, "fetched_bytes": fetched_bytes,
           "seconds": round(time.monotonic() - started, 3)}
    if args["texts"]:
        out["prompts"] = prompts
    return out
