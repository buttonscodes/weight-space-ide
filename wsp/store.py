"""Lazy access to checkpoint tensors: read the slice you need, not the model.

A *store* is a set of safetensors files (one file, a sharded Hugging Face checkpoint, or
URLs on the Hub).  Opening one reads only the JSON headers: a few kilobytes per file, however
large the model is.  After that

    store.read(name, region)        -> float32 torch tensor of just that region

touches only the bytes of that region: local files are memory-mapped, remote files are read
with HTTP range requests (and cached per tensor on disk).  Nothing here ever materialises a
whole checkpoint, so the cost of an operation is the size of what it looks at.

A region is one entry per axis: an int (axis dropped) or a range (axis kept).  `norm_index`
turns ordinary Python indexing (ints, slices, missing trailing axes) into that form.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import urllib.request
from typing import Iterable, Iterator

import numpy as np
import torch

NP_DTYPE = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64,
            "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_, "U16": np.uint16,
            "U32": np.uint32, "U64": np.uint64}
TORCH_DTYPE = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
FLOATS = set(TORCH_DTYPE)
Region = tuple          # of int | range, one per axis


def norm_index(idx, shape: tuple[int, ...]) -> Region:
    """Python basic indexing -> one int or range per axis."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    if len(idx) > len(shape):
        raise IndexError(f"{len(idx)} indices for a tensor of shape {list(shape)}")
    out = []
    for d, n in enumerate(shape):
        i = idx[d] if d < len(idx) else slice(None)
        if isinstance(i, range):
            out.append(i)
        elif isinstance(i, slice):
            out.append(range(*i.indices(n)))
        else:
            i = int(i)
            if not -n <= i < n:
                raise IndexError(f"index {i} out of range for axis {d} of size {n}")
            out.append(i % n)
    return tuple(out)


def region_shape(region: Region) -> tuple[int, ...]:
    return tuple(len(r) for r in region if isinstance(r, range))


def region_size(region: Region) -> int:
    n = 1
    for r in region:
        if isinstance(r, range):
            n *= len(r)
    return n


def region_index(region: Region) -> tuple:
    """A region as something numpy / torch can index with."""
    out = []
    for r in region:
        if isinstance(r, range):
            if len(r) == 0:
                out.append(slice(0, 0))
            else:
                stop = r[-1] + (1 if r.step > 0 else -1)
                out.append(slice(r.start, stop if stop >= 0 else None, r.step))
        else:
            out.append(r)
    return tuple(out)


def to_float(arr: np.ndarray, dtype: str) -> torch.Tensor:
    """A numpy array as stored -> float32 torch tensor (a copy)."""
    shape = arr.shape
    if dtype == "BF16":
        t = torch.from_numpy(np.array(arr, copy=True).reshape(-1)).view(torch.bfloat16).to(torch.float32)
        return t.reshape(shape)
    return torch.from_numpy(np.asarray(arr, dtype=np.float32).reshape(-1).copy()).reshape(shape)


def from_float(t: torch.Tensor, dtype: str) -> bytes:
    """float32 tensor -> bytes in a checkpoint dtype."""
    if dtype in TORCH_DTYPE:
        t = t.to(TORCH_DTYPE[dtype]).contiguous()
        if dtype == "BF16":
            return t.view(torch.int16).numpy().tobytes()
        return t.numpy().tobytes()
    return t.to(torch.float32).numpy().astype(NP_DTYPE[dtype]).tobytes()


class _File:
    """One safetensors file: header parsed eagerly, data read lazily."""
    remote = False

    def __init__(self, location: str):
        self.location = location
        n = struct.unpack("<Q", self._bytes(0, 8))[0]
        if n > 200_000_000:
            raise ValueError(f"{location} does not look like a safetensors file")
        header = json.loads(self._bytes(8, 8 + n))
        self.user_meta = header.pop("__metadata__", None) or {}
        self.data_start = 8 + n
        self.tensors = header                      # name -> {dtype, shape, data_offsets}

    def _bytes(self, a: int, b: int) -> bytes:
        raise NotImplementedError

    def array(self, name: str) -> np.ndarray:
        raise NotImplementedError


class LocalFile(_File):
    def __init__(self, path: str):
        self.path = path
        self._mm: np.memmap | None = None
        super().__init__(path)

    def _bytes(self, a: int, b: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(a)
            return f.read(b - a)

    def array(self, name: str) -> np.ndarray:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        m = self.tensors[name]
        a, b = m["data_offsets"]
        return np.ndarray(tuple(m["shape"]), dtype=NP_DTYPE[m["dtype"]], buffer=self._mm,
                          offset=self.data_start + a) if b > a else np.zeros(tuple(m["shape"]), NP_DTYPE[m["dtype"]])


class RemoteFile(_File):
    """A safetensors file behind HTTP: range requests, with whole tensors cached on first full read."""
    remote = True
    WHOLE = 48 * 1024 * 1024        # tensors up to this size are always fetched whole

    def __init__(self, url: str, cache_dir: str):
        self.url = url
        self.cache = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest()[:16])
        os.makedirs(self.cache, exist_ok=True)
        self.lock = threading.Lock()
        super().__init__(url)

    def _bytes(self, a: int, b: int) -> bytes:
        hp = os.path.join(self.cache, "header.bin")
        if os.path.exists(hp):
            with open(hp, "rb") as f:
                blob = f.read()
            if len(blob) >= b:
                return blob[a:b]
        data = self._range(a, b)
        if a == 8:                               # header: keep it, so reopening is offline and instant
            with open(hp, "wb") as f:
                f.write(struct.pack("<Q", b - 8) + data)
        return data

    def _range(self, a: int, b: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}"})
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in self.url:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if len(data) != b - a:
            raise IOError(f"range request returned {len(data)} bytes, expected {b - a}")
        return data

    def _blob(self, name: str) -> str:
        return os.path.join(self.cache, hashlib.sha1(name.encode()).hexdigest()[:20] + ".bin")

    def cached(self, name: str) -> bool:
        return os.path.exists(self._blob(name))

    def array(self, name: str) -> np.ndarray:
        m = self.tensors[name]
        a, b = m["data_offsets"]
        path = self._blob(name)
        with self.lock:
            if not os.path.exists(path) and b > a:
                tmp = path + ".part"
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.data_start + a}-{self.data_start + b - 1}"})
                token = os.environ.get("HF_TOKEN")
                if token and "huggingface.co" in self.url:
                    req.add_header("Authorization", f"Bearer {token}")
                got = 0
                with urllib.request.urlopen(req, timeout=3600) as r, open(tmp, "wb") as f:      # one streamed request
                    while True:
                        chunk = r.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                if got != b - a:
                    os.remove(tmp)
                    raise IOError(f"{name}: received {got} bytes, expected {b - a}")
                os.replace(tmp, path)
        if b == a:
            return np.zeros(tuple(m["shape"]), NP_DTYPE[m["dtype"]])
        return np.memmap(path, dtype=NP_DTYPE[m["dtype"]], mode="r", shape=tuple(m["shape"]))

    def rows(self, name: str, r0: int, r1: int) -> np.ndarray:
        """Rows r0:r1 of a tensor without fetching the rest (rows are contiguous on disk)."""
        m = self.tensors[name]
        shape = tuple(m["shape"])
        dt = np.dtype(NP_DTYPE[m["dtype"]])
        per = int(np.prod(shape[1:])) * dt.itemsize
        a = self.data_start + m["data_offsets"][0]
        return np.frombuffer(self._range(a + r0 * per, a + r1 * per), dtype=dt).reshape((r1 - r0,) + shape[1:])


class Store:
    def __init__(self, locations: Iterable[str], cache_dir: str = ""):
        self.files: list[_File] = []
        self.where: dict[str, _File] = {}
        self.cache_dir = cache_dir
        for loc in locations:
            f = RemoteFile(loc, cache_dir) if loc.startswith(("http://", "https://")) else LocalFile(loc)
            self.files.append(f)
            for name in f.tensors:
                self.where.setdefault(name, f)
        self._names = sorted(self.where, key=_natural)

    # ---------------------------------------------------------------- description
    @property
    def remote(self) -> bool:
        return any(f.remote for f in self.files)

    def names(self) -> list[str]:
        return self._names

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.where[name].tensors[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.where[name].tensors[name]["dtype"]

    def numel(self, name: str) -> int:
        n = 1
        for s in self.shape(name):
            n *= s
        return n

    def fetch(self, name: str) -> None:
        """Make sure a tensor is on disk (downloads and caches a remote one whole). Call before a pass over all of it."""
        f = self.where[name]
        if f.remote and not f.cached(name):
            f.array(name)

    def local(self, name: str) -> bool:
        """Is this tensor on disk already (a local file, or a cached copy of a remote one)?"""
        f = self.where[name]
        return not f.remote or f.cached(name)

    def size_bytes(self, name: str) -> int:
        a, b = self.where[name].tensors[name]["data_offsets"]
        return b - a

    def is_float(self, name: str) -> bool:
        return self.dtype(name) in FLOATS and not getattr(self, 'quantized_config', None)

    def total(self) -> int:
        return sum(self.numel(n) for n in self._names)

    def nbytes(self) -> int:
        return sum(m["data_offsets"][1] - m["data_offsets"][0] for f in self.files for m in f.tensors.values())

    # ---------------------------------------------------------------- reading
    def read(self, name: str, region: Region | None = None) -> torch.Tensor:
        """float32 copy of a region (default: the whole tensor)."""
        f = self.where[name]
        shape = self.shape(name)
        region = norm_index((), shape) if region is None else region
        dt = self.dtype(name)
        if getattr(self, 'quantized_config', None) or dt not in FLOATS:
            raise ValueError(f'{dt} storage is not a supported logical floating-point weight. '
                             'Packed MLX/scaled FP8 decoding needs an explicit format backend; raw integers are not weights.')
        if f.remote and not f.cached(name) and len(shape) >= 1 and \
                (f.tensors[name]["data_offsets"][1] - f.tensors[name]["data_offsets"][0]) > f.WHOLE:
            first = region[0]                       # big remote tensor: fetch only the rows asked for
            rows = range(first, first + 1) if isinstance(first, int) else first
            if len(rows) and rows.step == 1 and len(rows) * 4 < shape[0]:
                block = f.rows(name, rows.start, rows.stop)
                sub = (0 if isinstance(first, int) else slice(None),) + region_index(region)[1:]
                return to_float(block[sub], dt)
        return to_float(f.array(name)[region_index(region)], dt)

    def row_blocks(self, name: str, region: Region | None = None, max_elems: int = 16_000_000) -> Iterator[tuple[range, torch.Tensor]]:
        """Iterate a region in slabs along its first kept axis: (positions within the region, float32 slab).

        This is how anything that must look at a whole large tensor does it in bounded memory."""
        shape = self.shape(name)
        region = norm_index((), shape) if region is None else region
        kept = [d for d, r in enumerate(region) if isinstance(r, range)]
        if not kept:
            yield range(0, 1), self.read(name, region)
            return
        ax = kept[0]
        rows = region[ax]
        per = max(1, region_size(region) // max(1, len(rows)))
        step = max(1, max_elems // per)
        for a in range(0, len(rows), step):
            part = rows[a:a + step]
            yield range(a, a + len(part)), self.read(name, region[:ax] + (part,) + region[ax + 1:])

    def native(self, name: str) -> np.ndarray:
        """The tensor as stored (memory-mapped; no copy)."""
        return self.where[name].array(name)


def _natural(name: str):
    import re
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


# -------------------------------------------------------------------------------------- writing
def write_safetensors(path: str, items: list[tuple[str, str, tuple[int, ...]]], data, metadata: dict | None = None) -> None:
    """Write a safetensors file tensor by tensor. `items` = (name, dtype, shape); `data(name)` returns its bytes.

    The header needs every offset up front, which shapes and dtypes give; the tensors are then produced
    and written one at a time, so memory use is one tensor."""
    header: dict = {}
    off = 0
    for name, dtype, shape in items:
        n = int(np.prod(shape)) * np.dtype(NP_DTYPE[dtype]).itemsize if shape else np.dtype(NP_DTYPE[dtype]).itemsize
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name, dtype, shape in items:
            b = data(name)
            a0, a1 = header[name]["data_offsets"]
            if len(b) != a1 - a0:
                raise ValueError(f"{name}: produced {len(b)} bytes, expected {a1 - a0}")
            f.write(b)
    os.replace(tmp, path)


def checkpoint_files(directory: str) -> list[str]:
    """The safetensors files that make up the model in a Hugging Face style directory."""
    idx = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(directory, x) for x in files]
    one = os.path.join(directory, "model.safetensors")
    if os.path.exists(one):
        return [one]
    return sorted(os.path.join(directory, x) for x in os.listdir(directory) if x.endswith(".safetensors"))
