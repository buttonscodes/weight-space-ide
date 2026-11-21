"""Weight scripts: a small, safe language for writing weights.

Dense numbers are the most expensive way to tell a model what a weight should be.  A weight
script states an edit in terms of *directions with meaning* (a token's embedding, a token's
output direction, an activation, a singular vector) and structure (loops, outer products),
and the IDE turns it into compact deltas:

    # make neuron 12 of layer 0 fire on " cat" and promote " dog"   (any architecture)
    n = neuron(0, 12)
    n.w_in  = 4 * unit(E(" cat"))
    n.w_out = 3 * unit(U(" dog"))

    head(1, 3).O *= 0                                   # switch a head off
    orthogonalize(unit(U(" the")))                      # remove a direction from everything that writes to the stream
    rank1(mlp(2).W_out, k, v)                           # W_out += outer(k, v), stored as two vectors

    v = act("I love this", "blocks.2.hook_resid_post") - act("I hate this", "blocks.2.hook_resid_post")
    mlp(2).b_out += 4 * unit(v)                         # a steering vector, baked into a bias

Syntax is a subset of Python (assignments, augmented assignments, for / if, list
comprehensions, calls to the functions below, arithmetic, indexing).  It is interpreted by
walking the syntax tree: there is no eval, no attribute access outside a fixed list, no
imports, so a script produced by a model can be run without a sandbox.

Weights are read lazily and written as deltas.  `P("name")[12]` or `neuron(5, 12).w_out`
reads one row of one tensor, wherever the checkpoint lives; assigning to it records a delta
the size of that row.  A script therefore costs what it touches: editing a neuron of a
multi-gigabyte model reads and writes a few thousand numbers.  (Using a whole large tensor as
a value, e.g. `x = P("...")` followed by arithmetic, reads that whole tensor.  On a quantized
(GGUF) model such a read is decoded in row slabs and held as FP32 while the script runs, up to
WHOLE_READ_VALUES at a time; above that the script is refused with the slab loop to write instead.)

Functions (FUNCTION_DOC below is the reference shown to the programmer model).
"""
from __future__ import annotations

from .runtime_events import RuntimeRefusal, edit_preflight

import ast
import builtins
import itertools
import logging
import math
import operator
import re
import string
import sys
import time
import weakref

import numpy as np
import torch
import torch.nn.functional as F

from . import delta as D
from . import selections as S
from . import workspace as W
from .store import region_size
from .workspace import Ref

log = logging.getLogger(__name__)

MAX_STEPS = 3_000_000
PROGRESS_EVERY = 0.5            # seconds between "edit_progress" events of a long edit
# The guards below bound what one statement may touch at once; how large the whole patch may grow is
# wsp.edit_limit's (checked in Interp._add). MAX_ELEMS bounds a tensor a script constructs (zeros, ones,
# randn: big()); MAX_VIEW bounds one read of the weights (read(), base()), a scratch copy the size of the
# view. Generated selection scripts read at most selections.WINDOW (4,000,000) weights at once, so neither
# guard refuses them, whatever size of patch the limit allows.
MAX_ELEMS = 60_000_000
MAX_VIEW = MAX_ELEMS * 8
MAX_ITEMS = 10_000_000          # longest Python sequence or string (list(range(..)), "x" * n, ...) a script may build
# Python values are invisible to memory admission (edit_preflight sees tensors only), so the interpreter counts them
# itself: every operation that builds a list, tuple, dict, set or string charges its estimated size before (or, when
# its size is known only once built, as) it is built, and rebinding a name whose old value nothing else holds gives
# that value back. One script may hold at most this much; a loop or a comprehension of individually small values
# reaches it as surely as one large value does.
MAX_PY_BYTES = 512 << 20
MAX_FORMAT_WIDTH = 10_000       # widest field or precision in a format spec (f"{x:>99999999}" builds that string)
MAX_INT_BITS = 1 << 16          # largest Python integer a script may compute (7 ** 10 ** 9 would take gigabytes)
MAX_PRINT_CHARS = 20_000        # longest line print() keeps; past it the line is cut with a note
_CHUNK = 4096                   # items materialised between two charges (list(zip(..)), list(range(..)))
# Work inside one C call (a scan, a sort, a copy, a hash, a comparison) is charged to the step limit too, so that one
# step cannot do unbounded work: ITEMS_PER_STEP items of it count as one step, and a Python-level loop inside a
# built-in costs PY_ITEM work per item. One comparison, hash, sort or membership test may touch at most MAX_EXPANDED
# items, a container repeated inside another counted each time it occurs.
ITEMS_PER_STEP = 64
PY_ITEM = 16
MAX_EXPANDED = 20_000_000
# key= runs once per item inside sorted / min / max / .sort, so it takes only these value functions (or d.get).
_KEY_FUNCS = ("abs", "len", "float", "int", "str", "round", "bool", "min", "max", "sum", "tuple", "list", "sorted",
              "any", "all")
# A quantized (GGUF) store refuses a single read above 16M values, so that nothing reconstructs a large tensor by
# accident. A script that uses a larger view whole has asked for it: it is decoded in row slabs (Ref.whole) and held
# as FP32. A new whole read is refused when the decoded whole reads the script still holds (and views of them) would
# pass 320M values, 1.28 GB as FP32: three MLP matrices of a 27B model (17,408 x 5,120 each), never its 248,320 x
# 5,120 vocabulary matrix (5 GB). Only those decoded tensors are counted. A tensor computed from one (x * 1, x.clone(),
# x @ y) is not, and nothing totals tensor memory over a whole script: memory admission (edit_preflight) bounds each
# tensor operation on its own, and without a runtime client (tests, the CLI) only big() and the read bounds apply.
WHOLE_READ_VALUES = 320_000_000
_SEQ_METHODS = {list: {"append", "extend", "index", "count", "sort", "reverse", "insert", "pop", "remove", "copy", "clear"},
                tuple: {"index", "count"},
                dict: {"get", "keys", "values", "items", "pop", "setdefault", "update", "copy", "clear"},
                str: {"join", "split", "rsplit", "strip", "lstrip", "rstrip", "format", "upper", "lower", "replace",
                      "startswith", "endswith", "find", "count", "splitlines", "ljust", "rjust", "center", "zfill"}}
_PY_BUILTINS = ("range enumerate zip len int float round bool str list tuple dict set sorted reversed min max sum abs "
                "any all")

FUNCTION_DOC = """\
parts        head(L, h)   -> .Q .K .V [d_model, d_head]  .O [d_head, d_model]  (.b_Q .b_K .b_V if the model has them)
(lazy views, neuron(L, n) -> .w_in [d_model] .b_in .w_out [d_model]     (gated MLPs: .gate .up instead of .w_in)
 same on     mlp(L)       -> .W_in [d_model, d_mlp] (or .W_gate .W_up)  .W_out [d_mlp, d_model]  .b_in .b_out
 any model)  attn(L)      -> .W_Q .W_K .W_V .W_O (all heads)             W_E() [vocab, d_model]   W_U() [d_model, vocab]
             P(name)      -> any tensor by its checkpoint name, in the checkpoint's own layout
             Index a view to narrow it without reading the rest: mlp(3).W_out[17], P(name)[0:8, :]. Assign to edit.
             Using a whole view as a value reads all of it. On a quantized model a whole read is refused when the
             whole reads still held would pass 320M values (the error gives the row-slab loop to use instead); tensors
             computed from them are not counted there, each tensor operation is admitted on its own.
vocabulary   tok(s) -> id (s must be exactly one token)   toks(s) -> [ids]   detok(id) -> str
             E(s|id) -> embedding row [d_model]           U(s|id) -> unembedding vector [d_model]
             n_vocab, n_ctx, d_model, n_heads, d_head, d_mlp, n_layers
build        zeros(*shape) ones(*shape) full(shape, x) eye(n) onehot(i, n) arange(n) linspace(a, b, n)
             randn(*shape, seed=0) tensor([...])
vectors      unit(v) norm(v) dot(a, b) cossim(a, b) outer(a, b) proj(v, onto) reject(v, r)
             cat([..], dim) stack([..], dim) diag(v) transpose(M) reshape(x, shape) where(c, a, b)
math         abs sign sqrt exp log tanh sin cos relu gelu clip(x, lo, hi) maximum minimum
             sum(x, dim=None) mean(x, dim=None) max(x) min(x) argmax(x) argsort(x) topk(x, k) -> (values, indices)
linear alg   svd(M) -> (U, S, Vh)   pinv(M)   lowrank(M, r) -> best rank-r approximation   M @ N
model        act(prompt, hook, pos=-1) -> activation at a position, with the edits made so far in this script applied
                 hooks: blocks.L.hook_resid_pre | hook_resid_post | hook_attn_out | hook_mlp_out | mlp.hook_post | attn.hook_z
             logits(prompt) -> next-token logits [n_vocab]
edits        rank1(view, u, v)              view += outer(u, v), stored as the two vectors (view: a 2-D view or a tensor name)
             lowrank_add(view, A, B)        view += A @ B, stored as the factors
             orthogonalize(r, layers=None)  remove direction r from every matrix that writes to the residual stream:
                                            W <- W - (W r) r^T   (stored as rank-one pieces)
             ablate_direction("name", direction_index=None, weight_distributions={"attn.o_proj": {...}, "mlp.down_proj": {...}},
                              orthogonalize_direction=True, row_normalization="none"|"pre"|"full", full_normalization_lora_rank=3,
                              expert_modules="all"|"shared")
                                            remove a saved behaviour direction (docs/DIRECTIONS.md) from what chosen
                                            layers' attention and MLP output matrices write, with a per-layer weight
                                            {"max_weight", "max_weight_position", "min_weight", "min_weight_distance"}
             ablate_ara("name", start_layer_index=0, end_layer_index=None, preserve_good_behavior_weight=1.0,
                        steer_bad_behavior_weight=0.01, overcorrect_relative_weight=0.0, neighbor_count=1,
                        preserve_row_magnitudes=True, lora_rank=50, n_optimization_steps=5, learning_rate=1.0,
                        max_iter=20, history_size=10, seed=0, expert_modules="all"|"shared")
                                            fit a rank-lora_rank change to each attention and MLP output matrix of the
                                            layers (end exclusive) with L-BFGS, from a measurement saved with module_io:
                                            outputs on the good prompts kept, on the bad prompts moved towards the good
                                            (expert_modules="shared": a mixture-of-experts layer's shared expert only;
                                            ARA refuses routed experts)
             direction("name", k) -> unit direction at index k (0..n_layers) of a saved measurement (projected)
             direction_means("name") -> (good_mean, bad_mean), each [n_layers + 1, d_model]
             copy_head((L, h), (L2, h2))    scale_head((L, h), f)       copy_neuron((L, n), (L2, n2))   scale_neuron((L, n), f)
             revert(view)                   put a view, head(L, h) or neuron(L, n) back to its base values
             revert_neurons(L, [n, ...])    put every weight of several neurons of one layer back to base
             base(view) -> the view's values before any patch (reads are otherwise of the current weights)
atlas        select(concept=.., word=.., token=.., mech=.., role=.., layers=[a, b], effect="promote"|"suppress", top=N)
                 -> [(L, n), ...] neurons chosen from the atlas, strongest first (needs a built atlas; see the atlas tool)
             scale_neurons(L, [n, ...], f)  scale what several neurons of one layer write (thousands at once are fine);
                                            f is a number or a [d_model] vector applied to every neuron, not one per neuron
             shift_tokens(tokens, nats)     move the output score of tokens (strings or ids) by a fixed amount everywhere
             word_tokens("red") -> ids of every spelling of a word (case, leading space)
output       print(...)  values are returned with the result, so a script can also be used to inspect
python       for / if, list / dict comprehensions and generator expressions, f-strings (with format specs: f"{x:.3f}"),
             lists, tuples, dicts ({k: v}, d[k], d[k] += 1), `in`, `is None`, pi
             built-ins: range enumerate zip len int float round bool str list tuple dict set
                        sorted(xs, reverse=True, key=f) reversed min max sum abs any all  (max/min take key= too)
             methods:   list .append .extend .insert .pop .remove .sort(reverse=, key=) .reverse .index .count .copy .clear
                        dict .get .keys .values .items .pop .setdefault .update   str .join .split .strip .format .replace ...
             key= takes a value function by name (abs len float int str round bool min max sum tuple list sorted any
             all) or a dict's .get (sorted(d, key=d.get)); not a function that runs the model or edits weights. There
             is no lambda, so to sort by a score put the score first: sorted([(score, L, n) for ...], reverse=True)
             bounds:    a list, tuple or string holds at most 10M items; the Python values of one script (lists, strings,
                        dicts) at most 512 MB, so keep many numbers in a tensor; print keeps 20,000 characters of a line;
                        a scan, sort, copy or hash of n items counts n/64 steps of the step limit, each key= call a
                        step; one comparison or hash touches at most 20M items (a value repeated inside itself counts
                        each time it occurs)
             (no while, def, lambda, import, or attribute access beyond tensor methods such as .T .sum() .norm())
"""

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.MatMult: operator.matmul, ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
        ast.Is: operator.is_, ast.IsNot: operator.is_not}
_ORDER = (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge)
_TENSOR_ATTRS = {"T", "shape", "mT", "ndim"}
_TENSOR_METHODS = {"sum", "mean", "norm", "abs", "max", "min", "argmax", "argmin", "clone", "t", "flatten",
                   "reshape", "view", "std", "var", "sign", "clamp", "clip", "sqrt", "exp", "log", "tanh", "item",
                   "tolist", "numel", "dim", "size", "transpose", "permute", "squeeze", "unsqueeze", "topk",
                   "argsort", "sort", "float", "double", "int", "long", "nonzero", "any", "all", "cumsum", "dot",
                   "outer", "softmax", "log_softmax", "relu", "pow", "square", "roll", "flip", "copy",
                   "diagonal", "trace", "matmul", "mul", "add", "sub", "div", "neg"}
class ScriptError(Exception):
    def __init__(self, msg: str, lineno: int | None = None):
        super().__init__(f"line {lineno}: {msg}" if lineno else msg)
        self.lineno = lineno


class ScriptStopped(Exception):
    """The caller asked the script to stop (``stopped()`` was true at a progress point). Not a ScriptError: the
    script did nothing wrong, and nothing it made is kept."""


def _spec(spec: str) -> str:
    """A format spec whose widths and precisions are bounded (f"{x:>99999999999}" would build that string)."""
    if any(int(d) > MAX_FORMAT_WIDTH for d in re.findall(r"\d+", spec)):
        raise ScriptError(f"format spec {spec!r} asks for more than {MAX_FORMAT_WIDTH:,} characters; use a smaller width "
                          f"or precision")
    return spec


# A printf-style conversion: %[(name)][flags][width][.precision]type
_PERCENT = re.compile(r"%(?:\([^)]*\))?[-#0 +]*(\*|\d*)(?:\.(\*|\d*))?[a-zA-Z%]")


class _TooLong(Exception):
    """A rendering passed its character bound; `text` is what was written up to it."""
    def __init__(self, text: str):
        super().__init__(len(text))
        self.text = text


_VIEWS = (type({}.keys()), type({}.values()), type({}.items()))


def _render(v, as_repr: bool, cap: int) -> str:
    """str(v) (or repr(v)) written piece by piece, raising _TooLong past `cap` characters, so that it costs what it
    prints and no more. str() of a list builds the whole text in one call, and a list repeated inside itself prints
    each repeat: y = [[x] * 1000] * 1000 holds a thousand references and prints x a million times. Lists, tuples,
    dicts, sets and dict views are written as Python writes them; anything else is its own str() / repr()."""
    out: list[str] = []
    n = 0
    active: set[int] = set()

    def put(s: str):
        nonlocal n
        if n + len(s) > cap:
            out.append(s[:cap - n])
            raise _TooLong("".join(out))
        n += len(s)
        out.append(s)

    def items(xs):
        for i, x in enumerate(xs):
            if i:
                put(", ")
            walk(x, False)

    def walk(x, top: bool):
        t = type(x)
        if t is str:
            if top and not as_repr:
                put(x)
            else:
                put(repr(x) if len(x) <= cap else repr(x[:cap]))       # cut before repr builds a long copy
            return
        if t not in (list, tuple, dict, set) and t not in _VIEWS:
            put(str(x) if top and not as_repr else repr(x))
            return
        if id(x) in active:                                         # a list that contains itself
            put("[...]" if t is list else "{...}" if t is dict else "(...)")
            return
        active.add(id(x))
        if t is list:
            put("["), items(x), put("]")
        elif t is tuple:
            put("("), items(x), put(",)" if len(x) == 1 else ")")
        elif t is set:
            if x:
                put("{"), items(x), put("}")
            else:
                put("set()")
        elif t is dict:
            put("{")
            for i, (k, val) in enumerate(x.items()):
                if i:
                    put(", ")
                walk(k, False), put(": "), walk(val, False)
            put("}")
        else:
            put(f"{t.__name__}(["), items(x), put("])")
        active.discard(id(x))

    walk(v, True)
    return "".join(out)


def _short(v) -> str:
    """repr(v) for an error message: at most 200 characters, however large v is."""
    try:
        return _render(v, True, 200)
    except _TooLong as e:
        return e.text + "..."


def _small_int(v) -> bool:
    return type(v) is int and -5 <= v <= 256                    # cached by Python: costs nothing new


def _value_bytes(v) -> int:
    """Estimated bytes of a value made for a container element: a number, or a tuple and the numbers made for it.
    Strings and containers are charged when they are built, tensors by memory admission (here only the object)."""
    t = type(v)
    if t is int or t is float:
        return 0 if _small_int(v) else sys.getsizeof(v)
    if t is tuple:
        return sys.getsizeof(v) + sum(sys.getsizeof(m) for m in v if type(m) in (int, float) and not _small_int(m))
    if isinstance(v, torch.Tensor):
        return 256
    return 0


def _item_bytes(v) -> int:
    """One new element: its slot (8 bytes) and the value made for it."""
    return 8 + _value_bytes(v)


def _seq_bytes(proto, n: int, *parts) -> int:
    """Estimated bytes of a new list, tuple or string of n items or characters like `proto` (a string made from
    `proto` and `parts` takes 4 bytes a character if any of them is not ASCII)."""
    if isinstance(proto, str):
        ascii_ = proto.isascii() and all(p.isascii() for p in parts if isinstance(p, str))
        return 49 + n * (1 if ascii_ else 4)
    return 56 + 8 * n


class _TooBig(Exception):
    """A value expands to more items than an operation may touch."""


_CONTAINERS = (list, tuple, dict, set, frozenset)
_SCALARS = frozenset({int, float, bool, type(None)})


def _expanded(v, limit: int) -> tuple[int, int]:
    """(items, nodes walked one by one) for comparing, hashing, sorting or writing v: every element of every container,
    a container that occurs several times inside another counted each time (t = (t, t) forty times is 2 ** 40 items
    in one ==, hash or set insertion), a string as one item per 8 characters. One walk: memoised by id for a container
    held in more than one place, so it costs what v holds, not what it expands to, and done in C for a container of
    plain numbers or strings. Raises _TooBig as soon as the count passes `limit`."""
    memo: dict[int, int] = {}
    walked = 0

    def flat(xs):
        types = set(map(type, xs))
        if types <= _SCALARS:
            return len(xs)
        if types <= _SCALARS | {str}:
            chars = sum(map(len, xs)) if types == {str} else sum(len(x) for x in xs if type(x) is str)
            return len(xs) + chars // 8
        return None

    def size(x, shared: bool) -> int:
        nonlocal walked
        t = type(x)
        if t is str:
            return 1 + len(x) // 8
        if t not in _CONTAINERS and t not in _VIEWS:
            return 1
        k = id(x)
        if k in memo:
            return memo[k]
        memo[k] = 1                                     # in progress: a list inside itself counts once
        n = 1
        for part in ((x.keys(), x.values()) if t is dict else (x,)):
            f = flat(part)
            if f is not None:
                n += f
            else:
                for m in part:
                    walked += 1
                    n += size(m, sys.getrefcount(m) > 3)   # held elsewhere too (the slot, `m`, the argument)
                    if n > limit:
                        raise _TooBig
            if n > limit:
                raise _TooBig
        if shared:
            memo[k] = n
        else:
            del memo[k]
        return n

    return size(v, True), walked


class _Formatter(string.Formatter):
    """str.format without the attribute and item lookups of replacement fields ("{0.__class__}", "{0[k]}"):
    a field is a position or a keyword, nothing more. Each field is rendered within bounds and charged to the
    script, and the whole result is at most MAX_ITEMS characters."""
    def __init__(self, interp):
        super().__init__()
        self.it = interp
        self.total = 0

    def vformat(self, format_string, args, kwargs):
        self.total = len(format_string)
        return super().vformat(format_string, args, kwargs)

    def get_field(self, field_name, args, kwargs):
        if "." in field_name or "[" in field_name:
            raise ScriptError("format fields may only name an argument ({0} or {name}); put the value in a variable first")
        return super().get_field(field_name, args, kwargs)

    def convert_field(self, value, conversion):
        if conversion in ("r", "a"):
            return self.it._text(value, as_repr=True, ascii_=conversion == "a")
        if conversion == "s":
            return self.it._text(value)
        return super().convert_field(value, conversion)

    def format_field(self, value, format_spec):
        spec = _spec(format_spec)
        s = self.it._text(value) if not spec else format(value, spec)
        self.total += len(s)
        if self.total > MAX_ITEMS:
            raise ScriptError(f"format() would build a string of more than {MAX_ITEMS:,} characters")
        self.it._charge(_seq_bytes(s, len(s)), "format()")
        return s


def _percent_size(fmt: str, args, text) -> int:
    """Characters `fmt % args` may build, at most: the format, plus for each conversion its width and precision and
    the longest argument (a format of many %s can repeat one argument)."""
    specs = _PERCENT.findall(fmt)
    if any("*" in w + p for w, p in specs):
        raise ScriptError("a * width or precision in a % format is not available; write the number in the format")
    for w, p in specs:
        _spec(w + "." + p)
    vals = [args, *args.values()] if isinstance(args, dict) else list(args) if isinstance(args, tuple) else [args]
    longest = max((len(text(v)) for v in vals), default=0)
    return len(fmt) + sum(int(w or 0) + int(p or 0) + longest for w, p in specs)


class _Path:
    """An unfinished tensor name such as model.layers[3].mlp"""
    def __init__(self, s: str):
        self.s = s


class _NS:
    """A named bundle of views (what head(), neuron(), mlp() return)."""
    def __init__(self, what: str, **members):
        self.what = what
        self.members = members


class Interp:
    stopped = None                                          # also for an interpreter made without __init__

    def __init__(self, ws, extra: dict | None = None, request: str | None = None, stopped=None):
        self.ws = ws
        self.request = request                              # names this edit's progress events (see progress)
        self.stopped = stopped                              # stopped() true at a progress point: ScriptStopped
        self.struct = ws.struct
        self.extra: dict[str, list[D.Piece]] = {k: list(v) for k, v in (extra or {}).items()}
        self.locals: dict = {}
        self.out: list[str] = []
        self.steps = 0
        self.roots = {n.split(".", 1)[0] for n in ws.names()}
        self.prefixes: set[str] | None = None
        self._compact_at: dict[str, int] = {}
        self._n = sum(p.numel() for ps in self.extra.values() for p in ps)     # weights stored so far (see _stored)
        self._base_n = self._n                              # stored before this script: not its own patch
        self._cap = None                                    # edit_limit.Limit, worked out at the first write
        self._cap_measured = False                          # ... past SMALL_EDIT (memory and disk measured)
        self.line, self.lines, self._said = 0, 0, 0.0
        self.reverted: set[str] = set()                     # tensors a revert wrote to (see Session.apply)
        self.calls: list[dict] = []                         # measurements read, in order (the reproduction record)
        self.fn = self._functions()

    # ------------------------------------------------------------------ public
    def run(self, script: str) -> dict:
        try:
            tree = ast.parse(script)
        except SyntaxError as e:
            raise ScriptError(f"syntax error: {e.msg}", e.lineno) from None
        self.lines = len(tree.body)
        for i, stmt in enumerate(tree.body):
            self.line = i
            self.progress()
            self._stmt(stmt)
        deltas = {}
        for name, pieces in self.extra.items():
            pieces = D.compact(pieces, self.ws.shape(name))
            for p in pieces:
                if not p.finite():
                    raise ScriptError(f"{name} would contain NaN or inf")
            if pieces:
                deltas[name] = pieces
        return {"deltas": deltas, "output": "\n".join(self.out), "reverted": sorted(self.reverted & set(deltas)),
                "calls": list(self.calls)}

    def progress(self, window: int | None = None, windows: int | None = None) -> None:
        """Tell the browser how far a long edit has got (statement, and window within it), at most twice a second.
        Runs before each statement and before each component of ablate_direction / ablate_ara, so it is also where
        a caller's ``stopped()`` stops the script."""
        if self.stopped is not None and self.stopped():
            raise ScriptStopped()
        now = time.monotonic()
        if self.lines < 2 and window is None or now - self._said < PROGRESS_EVERY:
            return
        self._said = now
        data = {"line": self.line + 1, "lines": self.lines}
        if self.request:                                    # the browser shows only its own edit's progress
            data["request"] = self.request
        if window is not None:
            data |= {"window": window + 1, "windows": windows}
        data["stored"] = self._stored() - self.__dict__.get("_base_n", 0)     # weights this edit has written so far
        try:
            self.ws.emit("edit_progress", **data)
        except Exception:                                   # progress is a courtesy; it never fails an edit
            log.debug("edit progress", exc_info=True)

    # ------------------------------------------------------------------ lazy weights
    def read(self, ref: Ref, variant: str = "current") -> torch.Tensor:
        if self.ws.kind == "gguf" and ref.size > W.STREAM_ELEMS:
            return self._whole(ref, variant)
        if ref.size > MAX_VIEW:
            raise ScriptError(f"{ref.address()} is {ref.size:,} weights; index it to the part you need, e.g. "
                              f"{_slab_loop(ref, MAX_VIEW)}")
        edit_preflight(self.ws,self._stored(),scratch=ref.size*12)
        return ref.read(variant, self.extra if variant == "current" else None)

    def _held_values(self) -> int:
        """Values of whole quantized reads that the script still holds (a name, a list or a view of one keeps the
        decoded tensor alive; a tensor computed from it does not)."""
        held = getattr(self, "_held", None)
        if held is None:
            held = self._held = []
        held[:] = [(r, n) for r, n in held if r() is not None]
        return sum(n for _, n in held)

    def _whole(self, ref: Ref, variant: str) -> torch.Tensor:
        """A view above STREAM_ELEMS of a quantized model, decoded in row slabs (the store refuses one larger read) and
        held as FP32; refused, with the slab loop to write instead, past WHOLE_READ_VALUES held at once."""
        held = self._held_values()
        if held + ref.size > WHOLE_READ_VALUES:
            raise ScriptError(f"{ref.address()} is {ref.size:,} values; this script already holds {held:,} read whole and "
                              f"may hold at most {WHOLE_READ_VALUES:,} on a quantized model, so read it in row slabs "
                              f"instead: {_slab_loop(ref, WHOLE_READ_VALUES - held)}")
        edit_preflight(self.ws, self._stored(), scratch=ref.size * 12 + held * 4)
        t = ref.whole(variant, self.extra if variant == "current" else None)
        self._held.append((weakref.ref(t._base if t._base is not None else t), ref.size))
        return t

    def _binary(self, fn, a, b):
        if getattr(self.ws,'_runtime_client',None) and (isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor)):
            # Meta tensors infer the broadcast/matmul result shape without data.
            meta=lambda x:torch.empty(x.shape,dtype=x.dtype,device='meta') if isinstance(x,torch.Tensor) else x
            result=fn(meta(a),meta(b))
            if isinstance(result,torch.Tensor):edit_preflight(self.ws,scratch=result.numel()*max(12,result.element_size()*3))
        return self._sized(fn, a, b)

    def write(self, ref: Ref, val, op=None) -> None:
        if not self.ws.editable(ref.name):
            raise ScriptError(f"{ref.name} is not a floating-point tensor and cannot be edited")
        edit_preflight(self.ws,self._stored()+ref.size,scratch=ref.size*12)
        cur = self.read(ref)
        new = op(cur, val) if op is not None else val
        new = torch.as_tensor(new, dtype=torch.float32)
        try:
            fits = tuple(torch.broadcast_shapes(new.shape, cur.shape)) == tuple(cur.shape)
        except RuntimeError:
            fits = False
        if not fits:
            raise ScriptError(f"shape mismatch: target is {list(cur.shape)}, value is {list(new.shape)}")
        delta = new.expand(cur.shape) - cur
        if bool((delta != 0).any()):
            self._add(ref.name, ref.piece(delta))

    def _add(self, name: str, piece: D.Piece) -> None:
        self._n = self._stored() + piece.numel()
        lst = self.extra.setdefault(name, [])
        lst.append(piece)
        if len(lst) > self._compact_at.get(name, 256):
            self._compact(name)
        written = self._n - self.__dict__.get("_base_n", 0)
        if written > self._unchecked() and self._limit(written).exceeded(written):
            self._compact(name)                             # overlapping writes count twice until compacted
            written = self._n - self.__dict__.get("_base_n", 0)
            if over := self._limit(written).exceeded(written):
                from . import edit_limit
                raise ScriptError(edit_limit.refusal(written, over))

    def _compact(self, name: str) -> None:
        # Large tensors keep their pieces through compaction, so wait until the list has doubled
        # before tidying again: re-tidying on every addition made thousands of edits quadratic.
        lst = self.extra[name]
        old = sum(p.numel() for p in lst)
        self.extra[name] = lst = D.compact(lst, self.ws.shape(name))
        self._n += sum(p.numel() for p in lst) - old
        self._compact_at[name] = max(256, 2 * len(lst))

    def _unchecked(self) -> int:
        """A patch up to this size needs no limit check: memory and disk never refuse SMALL_EDIT weights or fewer
        (edit_limit.Limit.exceeded) and the setting, when there is one, is above it."""
        if "_free" not in self.__dict__:
            from . import edit_limit
            s = edit_limit.setting(self.ws)
            self._free = min(edit_limit.SMALL_EDIT, edit_limit.setting_weights(s) if s is not None else 1 << 62)
        return self._free

    def _limit(self, written: int):
        """How large this script's patch may grow (wsp.edit_limit): the setting; above SMALL_EDIT weights also
        disk, and the machine's memory when no runtime supervisor admits the edit (under one, every write's
        edit_preflight decides memory). Never the old fixed fallback: a script had no total cap before, so
        without a memory reading it is bounded only by the setting and disk. Measured once, at the first write
        past SMALL_EDIT."""
        from . import edit_limit
        measured = written > edit_limit.SMALL_EDIT
        if self.__dict__.get("_cap") is None or (measured and not self.__dict__.get("_cap_measured")):
            self._cap = edit_limit.allowed(self.ws, written, memory=getattr(self.ws, "_runtime_client", None) is None)
            self._cap_measured = measured
        return self._cap

    def _stored(self) -> int:
        """Weights the edit stores so far: a running count (summing every piece on each call was quadratic),
        worked out from the pieces the first time on an interpreter built without __init__."""
        if "_n" not in self.__dict__:
            self._n = sum(p.numel() for ps in self.extra.values() for p in ps)
        return self._n

    def _revert_positions(self, name: str, template: tuple, axis: int, pos: np.ndarray) -> None:
        """Add -delta for the given positions along one axis of a tensor (only where it is not zero)."""
        pieces = S.pieces_of(self.ws, name, self.extra)
        if not pieces or len(pos) == 0:
            return
        if not self.ws.editable(name):
            raise ScriptError(f"{name} is not a floating-point tensor and cannot be edited")
        cand = pos[S.axis_hits(pieces, template, axis, pos)]
        k = S.kept_before(template, axis)
        reader = S.DeltaReader(self.ws, name, axis, self.extra)    # the patch pieces only: no checkpoint read
        ws_ = S.windows(cand, S.other_size(template, axis), gap=None)
        for i, w in enumerate(ws_):
            self.progress(i, len(ws_))
            lo, hi = int(w[0]), int(w[-1]) + 1
            region = S.with_axis(template, axis, range(lo, hi))
            stored = self._stored()
            edit_preflight(self.ws, stored, scratch=region_size(region) * 12)
            d = reader.read(region)
            for a, b in S.runs(w):
                sub = d.narrow(k, a - lo, b - a)
                if bool(sub.any()):
                    stored += sub.numel()
                    edit_preflight(self.ws, stored)
                    self._add(name, D.trim(D.block(S.with_axis(region, axis, range(a, b)), -sub)))
                    self.reverted.add(name)

    def revert(self, view) -> None:
        if isinstance(view, _NS):
            for ref in view.members.values():
                if isinstance(ref, Ref):
                    self.revert(ref)
            return
        ref = self._ref_of(view)
        name, region = ref.name, ref.region
        pieces = S.pieces_of(self.ws, name, self.extra)
        if not pieces:
            return
        axis = S.best_axis(pieces, region)
        if axis is None:                                    # one weight
            d = self.ws.read(name, region, "delta", self.extra)
            if bool(d.any()):
                self._add(name, D.block(region, -d))
                self.reverted.add(name)
            return
        self._revert_positions(name, region, axis, np.sort(np.asarray(region[axis], dtype=np.int64)))

    def _ref_of(self, target) -> Ref:
        if isinstance(target, Ref):
            return target
        if isinstance(target, str):
            if target not in self.ws.store:
                raise ScriptError(f"unknown tensor {target!r}")
            return self.ws.ref(target)
        raise ScriptError("expected a view of weights (for example mlp(2).W_out) or a tensor name")

    # ------------------------------------------------------------------ Python values (bounded; see MAX_PY_BYTES)
    def _charge(self, nbytes: int, what: str = "this value") -> None:
        """Count the estimated bytes of a Python value about to be built; refuse past MAX_PY_BYTES held."""
        used = getattr(self, "_py", 0) + int(nbytes)
        if used > MAX_PY_BYTES:
            raise ScriptError(f"{what} would bring the Python lists, strings and dicts this script holds to about "
                              f"{used / 2 ** 20:,.1f} MB, past the {MAX_PY_BYTES / 2 ** 20:,.1f} MB a script may hold; "
                              f"keep numbers in a tensor (tensor([...]), topk) or keep fewer of them")
        self._py = used

    def _work(self, items: int, what: str = "this operation") -> None:
        """Charge `items` of work inside one C call to the step limit (ITEMS_PER_STEP items to a step)."""
        self.steps = getattr(self, "steps", 0) + int(items) // ITEMS_PER_STEP
        if self.steps > MAX_STEPS:
            raise ScriptError(f"script ran too long (step limit; {what} touches {int(items):,} items)")

    def _deep(self, v, what: str = "this value") -> int:
        """Items comparing, hashing or sorting v touches (see _expanded); charges the walk; refuses past MAX_EXPANDED."""
        t = type(v)
        if t in _SCALARS:
            return 1
        if t is str:
            return 1 + len(v) // 8
        if isinstance(v, torch.Tensor):
            return v.numel()
        try:
            n, walked = _expanded(v, MAX_EXPANDED)
        except _TooBig:
            raise ScriptError(f"{what} touches more than {MAX_EXPANDED:,} items when compared, hashed or sorted (a value "
                              f"repeated inside another counts each time it occurs); use smaller values") from None
        self._work(walked * PY_ITEM, what)
        return n

    def _hash(self, k) -> None:
        """Charge hashing a dict key or set member (a tuple is hashed through everything inside it)."""
        if type(k) not in _SCALARS:
            self._work(self._deep(k, "this key"), "hashing this key")

    def _key(self, key):
        """key= for sorted / min / max / .sort: a value function from _KEY_FUNCS or a dict's .get, charged a step per
        call (it runs inside one C call, once per item)."""
        if key is None:
            return None
        if not (any(key is self.fn.get(n) for n in _KEY_FUNCS) or getattr(key, "_dict_get", False)):
            raise ScriptError(f"key= takes a value function ({' '.join(_KEY_FUNCS)}) or d.get, by name; it runs once "
                              f"per item, so a function that runs the model or edits weights is not a key (there is no "
                              f"lambda: put the sort value first in a tuple instead)")

        def k(x):
            self._tick(None)
            return key(x)
        return k

    def _contains(self, a, b) -> bool:
        """`a in b`, with the work of the test charged first. A range answers numbers by arithmetic, as Python does for
        an int (0.5 in range(10 ** 12) would otherwise scan it in C); an unsized iterable is read as items, bounded."""
        if type(b) is range:
            if isinstance(a, torch.Tensor) and a.numel() == 1:
                a = a.item()
            if type(a) in (int, bool):
                return a in b
            if type(a) is float:
                return a.is_integer() and int(a) in b
            if not isinstance(a, torch.Tensor):
                return False                              # no other value equals an int (str, None, list, tuple, ...)
            return a in b                                 # a tensor of several values: Python raises at the first
        if isinstance(b, (dict, set, frozenset)) or type(b) is _VIEWS[0]:
            self._hash(a)
            return a in b
        if isinstance(b, str):
            if isinstance(a, str):
                self._work((len(a) + len(b)) // 8 + 1, "this substring test")
            return a in b
        if isinstance(b, torch.Tensor):
            self._work(b.numel(), "this membership test")
            return a in b
        if not isinstance(b, (list, tuple)) and type(b) not in _VIEWS:
            b = self._items(b)                            # zip, enumerate, ...: their items, at most MAX_ITEMS
        self._work(self._deep(b, "this container") + self._deep(a), "this membership test")
        return a in b

    def _release(self, old) -> None:
        """Give back what a value cost when the name holding it is rebound and nothing else holds it (it is freed
        then): the container and the elements made for it that only it holds."""
        t = type(old)
        if t not in (list, tuple, dict, set, str) or sys.getrefcount(old) > 3:   # the caller's `old`, ours, the argument
            return
        n = sys.getsizeof(old)
        if t is not str and t is not dict:
            self._work(len(old) * PY_ITEM, "freeing this value")
            for m in old:
                if sys.getrefcount(m) <= 3:                     # the container's slot, `m`, the argument
                    n += _value_bytes(m)
        self._py = max(0, getattr(self, "_py", 0) - n)

    def _items(self, x) -> list:
        """The items of an iterable argument of a Python built-in, at most MAX_ITEMS of them (a range or a zip of ranges
        is refused before it is built); items made on the way (from range, zip, enumerate, a tensor) are charged."""
        if hasattr(x, "__len__") and len(x) > MAX_ITEMS:
            raise ScriptError(f"sequence of {len(x):,} items is too long (at most {MAX_ITEMS:,})")
        if isinstance(x, (list, tuple, str, dict)):
            return x
        out: list = []
        it = iter(x)
        while chunk := list(itertools.islice(it, _CHUNK)):
            self._charge(sum(_item_bytes(v) for v in chunk), "this sequence")
            self._work(len(chunk) * PY_ITEM, "this sequence")
            out.extend(chunk)
            if len(out) > MAX_ITEMS:
                raise ScriptError(f"sequence is too long (at most {MAX_ITEMS:,} items)")
        return out

    def _text(self, v, as_repr: bool = False, ascii_: bool = False) -> str:
        """str(v) / repr(v) / ascii(v), at most MAX_ITEMS characters, charged."""
        try:
            s = _render(v, as_repr, MAX_ITEMS // 10 if ascii_ else MAX_ITEMS)
        except _TooLong:
            raise ScriptError(f"this value is more than {MAX_ITEMS:,} characters long as text; print a slice of it "
                              f"(x[:20]) or its len()") from None
        self._work(len(s) * 4, "writing this value as text")
        if ascii_ and not s.isascii():
            s = s.encode("ascii", "backslashreplace").decode("ascii")
        if s is not v:
            self._charge(_seq_bytes(s, len(s)), "this text")
        return s

    def _measure(self, v) -> str:
        """repr(v) within MAX_ITEMS characters: its work charged to the step limit, its bytes not (it is only measured)."""
        try:
            s = _render(v, True, MAX_ITEMS)
        except _TooLong:
            raise ScriptError(f"a value in this format is more than {MAX_ITEMS:,} characters long as text") from None
        self._work(len(s) * 4, "writing this value as text")
        return s

    def _show(self, v) -> str:
        """What print() writes for a Python value: its text, cut at MAX_PRINT_CHARS with a note."""
        try:
            s = _render(v, False, MAX_PRINT_CHARS)
            self._work(len(s) * 4, "printing this value")
            return s
        except _TooLong as e:
            size = f"{len(v):,} characters" if isinstance(v, str) else \
                f"{len(v):,} items" if hasattr(v, "__len__") else "more"
            return e.text + f" ... (cut at {MAX_PRINT_CHARS:,} characters; it has {size}: print a slice, e.g. x[:20])"

    def _formatter(self) -> _Formatter:
        f = getattr(self, "_fmt", None)
        if f is None:
            f = self._fmt = _Formatter(self)
        return f

    def _sort_cost(self, xs) -> int:
        """Items a sort of xs touches: each comparison can walk an element, about log2(n) comparisons an element."""
        return self._deep(xs, "this sort") * max(1, len(xs).bit_length())

    def _method(self, base, attr: str):
        """A whitelisted method of a list, tuple, dict or str: what it builds bounded and charged (_method_inner),
        and the work it does in its one call charged to the step limit first."""
        inner = self._method_inner(base, attr)
        kind = type(base)
        if kind is str and attr not in ("format", "join"):         # format renders, join counts, within themselves
            def cost(*a, **kw):
                return (len(base) + sum(len(x) for x in (*a, *kw.values()) if isinstance(x, str))) // 8 + 1
        elif attr in ("index", "count", "remove"):                # each element compared with the value
            def cost(v=None, *a):
                return self._deep(base, f"{attr}()") + self._deep(v)
        elif attr == "sort":
            def cost(*a, **kw):
                return self._sort_cost(base)
        elif kind is dict and attr in ("get", "pop", "setdefault"):
            def cost(k=None, *a):
                return self._deep(k, "this key")
        elif attr == "update":
            def cost(*a, **kw):
                return sum(self._deep(x, "update()") for x in a) + len(kw)
        elif attr in ("insert", "pop", "reverse", "copy", "clear", "extend"):    # moves or copies every element
            def cost(*a, **kw):
                return len(base) + (sum(len(x) for x in a if hasattr(x, "__len__")) if attr == "extend" else 0)
        else:                                                     # append, keys, values, items: constant
            def cost(*a, **kw):
                return 0

        def method(*a, **kw):
            self._work(cost(*a, **kw), f"{attr}()")
            return inner(*a, **kw)
        if kind is dict and attr == "get":
            method._dict_get = True                               # a key function sorted() accepts
        return method

    def _method_inner(self, base, attr: str):
        """A whitelisted method of a list, tuple, dict or str, with what it builds bounded and charged."""
        fn = getattr(base, attr)
        kind = type(base)
        if attr in ("extend", "update"):
            def grow(*a, **kw):
                a = [self._items(x) for x in a]
                added = sum(len(x) for x in a) + len(kw)
                if len(base) + added > MAX_ITEMS:
                    raise ScriptError(f"{kind.__name__} would hold more than {MAX_ITEMS:,} items")
                self._charge((8 if kind is list else 80) * added, f"{attr}()")
                return fn(*a, **kw)
            return grow
        if attr in ("append", "insert"):
            def add(*a):
                self._charge(_item_bytes(a[-1]) if a else 0, f"{attr}()")
                return fn(*a)
            return add
        if attr == "setdefault":
            def setdefault(k, v=None):
                if k not in base:
                    self._charge(80 + _value_bytes(k) + _value_bytes(v), "setdefault()")
                return fn(k, v)
            return setdefault
        if attr == "copy":
            def copy():
                self._charge(sys.getsizeof(base), "copy()")
                return fn()
            return copy
        if attr == "index" and kind is list:            # its ValueError would hold the value it looked for, whole
            def index(v, *a):
                hay = base[slice(*a)] if a else base
                if v not in hay:
                    raise ScriptError(f"{_short(v)} is not in the list")
                return fn(v, *a)
            return index
        if attr == "pop" and kind is dict:              # likewise its KeyError
            def pop(k, *a):
                if not a and k not in base:
                    raise ScriptError(f"key {_short(k)} is not in the dict; use d.pop(k, default)")
                return fn(k, *a)
            return pop
        if attr == "replace":
            def replace(old, new, *a):
                n = len(base) + (base.count(old) * max(0, len(new) - len(old)) if old else (len(base) + 1) * len(new))
                if n > MAX_ITEMS:
                    raise ScriptError(f"replace() would build a string of more than {MAX_ITEMS:,} characters")
                self._charge(_seq_bytes(base, n, new), "replace()")
                return fn(old, new, *a)
            return replace
        if attr == "sort":
            return lambda key=None, reverse=False: fn(key=self._key(key), reverse=bool(reverse))
        if attr in ("ljust", "rjust", "center", "zfill"):
            def pad(width, *a):
                if int(width) > MAX_ITEMS:
                    raise ScriptError(f"{attr}() width {int(width):,} is too large")
                self._charge(_seq_bytes(base, max(len(base), int(width)), *a), f"{attr}()")
                return fn(width, *a)
            return pad
        if attr in ("upper", "lower", "strip", "lstrip", "rstrip"):
            def same(*a):
                self._charge(_seq_bytes(base, len(base)), f"{attr}()")
                return fn(*a)
            return same
        if attr in ("split", "rsplit", "splitlines"):
            def split(*a, **kw):                        # at most one piece per character, each a new string
                self._charge(len(base) + (len(base) + 1) * 8 + (len(base) // 2 + 1) * 50, f"{attr}()")
                return fn(*a, **kw)
            return split
        if attr == "format":
            return lambda *a, **kw: self._formatter().format(base, *a, **kw)
        if attr == "join":
            def join(xs):
                xs = self._items(xs)
                strs = [x for x in xs if isinstance(x, str)]
                n = len(base) * max(0, len(xs) - 1) + sum(len(x) for x in strs)
                if n > MAX_ITEMS:
                    raise ScriptError(f"join() would build a string of {n:,} characters (at most {MAX_ITEMS:,})")
                self._charge(_seq_bytes(base, n, *strs), "join()")
                self._work(len(xs) + n // 8, "join()")
                return fn(xs)
            return join
        return fn

    def _sized(self, fn, a, b):
        """Refuse Python operations whose result alone would exhaust memory or time ([0] * 10**10, "x" * 10**10,
        10 ** 10 ** 8, x * x repeated, "%9999d" * 10**6 % ...), and charge the lists and strings they build
        (tensors are bounded by edit_preflight instead)."""
        if fn in _ORDER and not isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor) \
                and (type(a) not in _SCALARS or type(b) not in _SCALARS):
            # ==, <, ... walk both values element by element: two long lists, or t = (t, t) repeated
            self._work(min(self._deep(a, "this comparison"), self._deep(b, "this comparison")), "this comparison")
            return fn(a, b)
        if fn is operator.mul and isinstance(a, int) and isinstance(b, (list, tuple, str)):
            a, b = b, a
        if fn is operator.mul and isinstance(a, (list, tuple, str)) and isinstance(b, int) and not isinstance(b, bool):
            n = len(a) * max(b, 0)
            if n > MAX_ITEMS:
                raise ScriptError(f"repeating a sequence of {len(a):,} items {b:,} times is too long (at most {MAX_ITEMS:,})")
            self._charge(_seq_bytes(a, n), "this repeat")
            self._work(n // 8 + 1 if isinstance(a, str) else n, "this repeat")
        elif fn is operator.add and isinstance(a, (list, tuple, str)) and type(a) is type(b):
            n = len(a) + len(b)
            if n > MAX_ITEMS:
                raise ScriptError(f"joining sequences of {len(a):,} and {len(b):,} items is too long (at most {MAX_ITEMS:,})")
            self._charge(_seq_bytes(a, n, b), "this concatenation")
            self._work(n // 8 + 1 if isinstance(a, str) else n, "this concatenation")
        elif type(a) is int and type(b) is int:
            if fn is operator.pow and b > 0 and abs(a) > 1 and b * a.bit_length() > MAX_INT_BITS:
                base = a if a.bit_length() < 64 else "x"
                raise ScriptError(f"integer power {base} ** {b} is too large (more than {MAX_INT_BITS:,} bits); "
                                  f"use a float (e.g. {base}.0 ** {b})")
            if fn is operator.mul and a.bit_length() + b.bit_length() > MAX_INT_BITS:
                raise ScriptError(f"integer product is too large (more than {MAX_INT_BITS:,} bits); use floats")
        elif fn is operator.mod and isinstance(a, str):
            n = _percent_size(a, b, self._measure)
            if n > MAX_ITEMS:
                raise ScriptError(f"this % format would build a string of more than {MAX_ITEMS:,} characters")
            self._charge(_seq_bytes(a, n), "this % format")
            self._work(n // 8 + 1, "this % format")
        elif fn is operator.sub and isinstance(a, set):
            self._charge(sys.getsizeof(a), "this set difference")
            self._work(len(a) + (len(b) if isinstance(b, set) else 0), "this set difference")
        return fn(a, b)

    # ------------------------------------------------------------------ statements
    def _tick(self, node):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise ScriptError("script ran too long (step limit)", getattr(node, "lineno", None))

    def _stmt(self, node):
        self._tick(node)
        try:
            if isinstance(node, ast.Assign):
                val = self._eval(node.value)              # a view assigned to a name stays a view (nothing is read)
                if isinstance(val, _Path):
                    raise ScriptError(f"{val.s} is not a tensor; finish the path")
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self._setlocal(t.id, val)
                    else:
                        self._assign(t, self._force(val), None)
            elif isinstance(node, ast.AugAssign):
                if type(node.op) not in _BIN:
                    raise ScriptError("unsupported operator")
                self._assign(node.target, self._val(node.value), _BIN[type(node.op)])
            elif isinstance(node, ast.For):
                for item in self._val(node.iter):
                    self._bind(node.target, item)
                    for s in node.body:
                        self._stmt(s)
            elif isinstance(node, ast.If):
                for s in (node.body if self._truth(self._val(node.test)) else node.orelse):
                    self._stmt(s)
            elif isinstance(node, ast.Expr):
                self._eval(node.value)
            elif isinstance(node, ast.Pass):
                pass
            else:
                raise ScriptError(f"{type(node).__name__} statements are not part of the weight-script language")
        except ScriptError as e:
            if e.lineno is None:
                raise ScriptError(str(e), node.lineno) from None
            raise
        except RuntimeRefusal:
            raise
        except ScriptStopped:
            raise
        except Exception as e:  # torch shape errors etc.
            raise ScriptError(f"{type(e).__name__}: {e}", node.lineno) from None

    @staticmethod
    def _truth(v) -> bool:
        return bool(v.all()) if isinstance(v, torch.Tensor) else bool(v)

    def _bind(self, target, val):
        if isinstance(target, ast.Name):
            self._setlocal(target.id, val)
        elif isinstance(target, (ast.Tuple, ast.List)):
            vals = list(itertools.islice(iter(val), len(target.elts) + 1))    # a, b = range(10 ** 12) builds 3, not all
            if len(vals) != len(target.elts):
                raise ScriptError("cannot unpack")
            for t, v in zip(target.elts, vals):
                self._bind(t, v)
        else:
            raise ScriptError("bad loop target")

    def _setlocal(self, name: str, val):
        if name.startswith("_"):
            raise ScriptError("names may not start with an underscore")
        old = self.locals.get(name)
        self.locals[name] = val
        if old is not None and old is not val:
            self._release(old)

    def _assign(self, target, val, op):
        if isinstance(target, ast.Name):                  # only reached for augmented assignment
            cur = self.locals.get(target.id)
            if isinstance(cur, Ref):                      # `w *= 0` on a name bound to a view edits the weights, as in numpy
                self.write(cur, val, op)
                return
            self._setlocal(target.id, self._binary(op, self._val(target), val) if op is not None else val)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and op is None:
            self._bind(target, val)
            return
        if isinstance(target, (ast.Attribute, ast.Call)):
            ref = self._eval(target)
            if not isinstance(ref, Ref):
                raise ScriptError("assignment target is not a view of weights")
            self.write(ref, val, op)
            return
        if isinstance(target, ast.Subscript):
            base = self._eval(target.value)
            if isinstance(base, dict):                    # d[k] = v, d[k] += v (keys are values, not tensor indices)
                key = self._val(target.slice)
                self._hash(key)
                if op is not None and key not in base:
                    raise ScriptError(f"key {_short(key)} is not in the dict; set it first (d[k] = 0) or use d.get(k, 0)")
                if key not in base:
                    self._charge(80 + _value_bytes(key) + _value_bytes(val), "this dict entry")
                base[key] = self._binary(op, base[key], val) if op is not None else val
                return
            idx = self._index(target.slice)
            if isinstance(base, Ref):
                if _basic(idx):
                    self.write(base.sub(idx), val, op)
                else:                                     # boolean / list index: read the view, change it, write it back
                    cur = self.read(base)
                    new = cur.clone()
                    new[idx] = torch.as_tensor(op(cur[idx], val) if op is not None else val, dtype=torch.float32)
                    self.write(base, new, None)
            elif isinstance(base, torch.Tensor):
                base[idx] = torch.as_tensor(op(base[idx], val) if op is not None else val, dtype=base.dtype)
            elif isinstance(base, list):
                base[idx] = self._binary(op, base[idx], val) if op is not None else val
            else:
                raise ScriptError("cannot assign into this value")
            return
        raise ScriptError("unsupported assignment target")

    # ------------------------------------------------------------------ expressions
    def _index(self, node):
        if isinstance(node, ast.Tuple):
            return tuple(self._index1(e) for e in node.elts)
        return self._index1(node)

    def _index1(self, node):
        if isinstance(node, ast.Slice):
            return slice(*(None if x is None else int(self._val(x)) for x in (node.lower, node.upper, node.step)))
        v = self._val(node)
        if isinstance(v, torch.Tensor) and v.dim() == 0 and v.dtype != torch.bool:
            return int(v)
        if isinstance(v, torch.Tensor) and v.dtype not in (torch.bool, torch.int64, torch.int32):
            v = v.long()
        if isinstance(v, (list, tuple)):
            v = torch.tensor(v)
        if isinstance(v, float):
            v = int(v)
        return v

    def _name(self, name: str):
        if name in self.locals:
            return self.locals[name]
        if name in self.fn:
            return self.fn[name]
        if name in self.roots:
            return self._path(name)
        if name in ("np", "torch", "numpy", "math"):
            return _Path("@fn")
        if name in ("map", "filter"):
            raise ScriptError(f"{name}() is not available; use a list comprehension: [f(x) for x in xs if ...]")
        if hasattr(builtins, name):
            raise ScriptError(f"the Python built-in {name!r} is not part of weight scripts; the built-ins are: "
                              f"{_PY_BUILTINS}")
        raise ScriptError(f"unknown name {name!r}")

    def _path(self, full: str):
        if full in self.ws.store:
            return self.ws.ref(full)
        if self.prefixes is None:
            self.prefixes = set()
            for n in self.ws.names():
                parts = n.split(".")
                for i in range(1, len(parts)):
                    self.prefixes.add(".".join(parts[:i]))
        if full in self.prefixes:
            return _Path(full)
        raise ScriptError(f"unknown tensor path {full!r}")

    def _eval(self, node):
        """Evaluate without forcing a read: weights come back as lazy Ref views."""
        self._tick(node)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool)) or node.value is None:
                return node.value
            raise ScriptError("unsupported constant")
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.Attribute):
            v = node.value
            if isinstance(v, ast.Name) and v.id in self.roots and v.id not in self.locals and v.id in self.fn:
                base = self._path(v.id)                   # a tensor-name root that happens to share a function's name
            else:
                base = self._eval(v)
            attr = node.attr
            if attr.startswith("_"):
                raise ScriptError("private attributes are not available")
            if isinstance(base, _Path):
                if base.s == "@fn":
                    if attr in ("linalg", "nn", "functional"):
                        return base
                    if attr in self.fn:
                        return self.fn[attr]
                    raise ScriptError(f"function {attr!r} is not available; see the function list")
                return self._path(f"{base.s}.{attr}")
            if isinstance(base, _NS):
                if attr not in base.members:
                    raise ScriptError(f"{base.what} has no part {attr!r}; it has {sorted(base.members)}")
                return base.members[attr]
            if isinstance(base, Ref):
                if attr == "shape":
                    return tuple(base.shape)
                base = self.read(base)
            if isinstance(base, torch.Tensor):
                if attr in _TENSOR_ATTRS:
                    v = getattr(base, attr)
                    return tuple(v) if attr == "shape" else v
                if attr in _TENSOR_METHODS:
                    if attr in ('outer','matmul','dot','mul','add','sub','div','pow'):
                        return lambda other,attr=attr,base=base:self._binary(lambda a,b:getattr(a,attr)(b),base,other)
                    fn=base.clone if attr=='copy' else getattr(base,attr)
                    def checked(*args,**kwargs):
                        if attr == "tolist":                  # Python numbers: about 32 bytes each, not 4
                            self._charge(base.numel() * 32 + 64 * (base.numel() // max(1, base.shape[-1] if base.dim() else 1)), "tolist()")
                        edit_preflight(self.ws,scratch=base.numel()*max(12,base.element_size()*3))
                        return fn(*args,**kwargs)
                    return checked
                raise ScriptError(f"tensor attribute {attr!r} is not available")
            if type(base).__module__ == "torch.return_types" and attr in ("values", "indices"):
                return getattr(base, attr)                # x.max(dim=0).values, x.topk(5).indices
            kind = type(base) if type(base) in _SEQ_METHODS else None
            if kind is not None:
                if attr in _SEQ_METHODS[kind]:
                    return self._method(base, attr)
                raise ScriptError(f"{kind.__name__} has no method {attr!r} here; available: "
                                  f"{' '.join(sorted(_SEQ_METHODS[kind]))}")
            raise ScriptError(f"attribute access on {type(base).__name__} is not available")
        if isinstance(node, ast.Subscript):
            base = self._eval(node.value)
            if isinstance(base, dict):
                key = self._val(node.slice)
                self._hash(key)
                if key not in base:
                    raise ScriptError(f"key {_short(key)} is not in the dict; use d.get(k, default) for a key that may "
                                      f"be missing")
                return base[key]
            idx = self._index(node.slice)
            if isinstance(base, _Path):
                if isinstance(idx, int) and base.s != "@fn":
                    return self._path(f"{base.s}.{idx}")
                raise ScriptError(f"{base.s} is not a tensor; finish the path")
            if isinstance(base, Ref):
                if _basic(idx):
                    try:
                        return base.sub(idx)
                    except IndexError as e:
                        raise ScriptError(str(e)) from None
                return self.read(base)[idx]
            if isinstance(base, (list, tuple, str)) and isinstance(idx, slice):      # a copy: x[:] in a loop adds up
                n = len(range(*idx.indices(len(base))))
                self._charge(_seq_bytes(base, n), "this slice")
                self._work(n // 8 + 1 if isinstance(base, str) else n, "this slice")
            if isinstance(base, (torch.Tensor, list, tuple, str)):
                return base[idx]
            raise ScriptError("value is not indexable")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN:
                raise ScriptError("unsupported operator")
            return self._binary(_BIN[type(node.op)],self._val(node.left),self._val(node.right))
        if isinstance(node, ast.UnaryOp):
            v = self._val(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, ast.Not):
                return not self._truth(v)
            raise ScriptError("unsupported unary operator")
        if isinstance(node, ast.Compare):
            left = self._val(node.left)
            res = True
            for op, comp in zip(node.ops, node.comparators):
                right = self._val(comp)
                if type(op) not in _CMP:
                    raise ScriptError("unsupported comparison")
                if isinstance(op, (ast.In, ast.NotIn)):
                    res = self._contains(left, right) == isinstance(op, ast.In)     # its work charged first
                elif isinstance(op, (ast.Is, ast.IsNot)):
                    res = _CMP[type(op)](left, right)     # a bool; no shape to check
                else:
                    res = self._binary(_CMP[type(op)],left,right)
                if not isinstance(res, torch.Tensor) and not res:
                    return False
                left = right
            return res
        if isinstance(node, ast.BoolOp):
            vals = [self._truth(self._val(v)) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.IfExp):
            return self._val(node.body if self._truth(self._val(node.test)) else node.orelse)
        if isinstance(node, (ast.List, ast.Tuple)):
            # Literals are not charged: each element costs a step, so MAX_STEPS bounds them; a literal that is kept
            # is charged where it is kept (append, a comprehension, a dict entry).
            vals = [self._val(e) for e in node.elts]
            return vals if isinstance(node, ast.List) else tuple(vals)
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise ScriptError("{**d} is not available; copy with d.copy() and add with d.update(other)")
            d = {}
            for k, v in zip(node.keys, node.values):
                key = self._val(k)
                self._hash(key)
                d[key] = self._val(v)
            return d
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)) and \
                not any(g.is_async for g in node.generators):
            # A generator expression is evaluated eagerly, as a list: sum(x for x in xs) and any(...) work the same.
            out: list = []
            item = (lambda: (self._val(node.key), self._val(node.value))) if isinstance(node, ast.DictComp) else \
                (lambda: self._val(node.elt))
            slot = 80 if isinstance(node, ast.DictComp) else 64 if isinstance(node, ast.SetComp) else 8

            def comp(i: int):
                if i == len(node.generators):
                    v = item()                            # [[0] * n for ...] adds up: each item is charged
                    if slot != 8:
                        self._hash(v[0] if slot == 80 else v)
                    self._charge(slot + (_value_bytes(v[0]) + _value_bytes(v[1]) if slot == 80 else _value_bytes(v)),
                                 "this comprehension")
                    out.append(v)
                    return
                gen = node.generators[i]
                for x in self._val(gen.iter):
                    self._tick(node)
                    self._bind(gen.target, x)
                    if all(self._truth(self._val(c)) for c in gen.ifs):
                        comp(i + 1)

            comp(0)
            return dict(out) if isinstance(node, ast.DictComp) else set(out) if isinstance(node, ast.SetComp) else out
        if isinstance(node, ast.Lambda):
            raise ScriptError("there is no lambda; to sort by a score put it first in a tuple "
                              "(sorted([(score, x) for x in xs], reverse=True)) or pass a function by name (key=abs)")
        if isinstance(node, ast.Set):
            out = set()
            for e in node.elts:
                v = self._val(e)
                self._hash(v)
                out.add(v)
            return out
        if isinstance(node, ast.Call):
            fn = self._eval(node.func)
            if not callable(fn):
                raise ScriptError("not a function")
            lazy = getattr(fn, "lazy", False)             # functions that take views without reading them
            args = [(self._eval(a) if lazy else self._val(a)) for a in node.args]
            kwargs = {k.arg: self._val(k.value) for k in node.keywords if k.arg}
            return fn(*args, **kwargs)
        if isinstance(node, ast.JoinedStr):
            parts, n = [], 0
            for v in node.values:
                s = self._formatted(v) if isinstance(v, ast.FormattedValue) else v.value
                n += len(s)
                if n > MAX_ITEMS:                         # f"{x}{x}{x}..." of a long x
                    raise ScriptError(f"this f-string would be more than {MAX_ITEMS:,} characters long")
                parts.append(s)
            return "".join(parts)
        raise ScriptError(f"{type(node).__name__} expressions are not part of the weight-script language")

    def _formatted(self, node: ast.FormattedValue) -> str:
        """One {value!conversion:spec} of an f-string, rendered within bounds and charged."""
        v = self._val(node.value)
        conv = node.conversion
        if conv in (ord("r"), ord("a"), ord("s")):
            v = self._text(v, as_repr=conv != ord("s"), ascii_=conv == ord("a"))
        spec = _spec(self._val(node.format_spec)) if node.format_spec is not None else ""
        if not spec:
            return self._text(v)
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            v = v.item()                                  # f"{x.norm():.3f}" formats the number
        s = format(v, spec)                               # a width is bounded by _spec; a container takes no spec
        self._charge(_seq_bytes(s, len(s)), "this f-string")
        return s

    def _val(self, node):
        """Evaluate to a value: views are read."""
        return self._force(self._eval(node))

    def _force(self, v):
        if isinstance(v, Ref):
            return self.read(v)
        if isinstance(v, _Path):
            raise ScriptError(f"{v.s} is not a tensor; finish the path")
        if isinstance(v, _NS):
            raise ScriptError(f"{v.what} is a bundle of parts ({sorted(v.members)}); pick one, e.g. .{sorted(v.members)[0]}")
        return v

    # ------------------------------------------------------------------ functions
    def _functions(self) -> dict:
        ws, st, sem = self.ws, self.struct, self.ws.sem
        T = torch.as_tensor

        def big(*shape):
            shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
            n = 1
            for x in shape:
                n *= int(x)
            if n > MAX_ELEMS:
                raise ScriptError("tensor too large")
            edit_preflight(ws,scratch=n*12)
            return tuple(int(x) for x in shape)

        def need(cond: bool, what: str):
            if not cond:
                raise ScriptError(f"{what} is not available: {st.describe().splitlines()[0]}")

        def tok(s):
            if isinstance(s, (int, float)) or (isinstance(s, torch.Tensor) and s.dim() == 0):
                i = int(s)
                if not 0 <= i < max(st.d_vocab, 1):
                    raise ScriptError(f"token id {i} out of range")
                return i
            ids = ws.tok.encode_token(s)
            if len(ids) != 1:
                pieces = [ws.tok.token_str(i) for i in ids]
                raise ScriptError(f"{s!r} is {len(ids)} tokens {pieces}, not one; use toks() or pick one piece")
            return ids[0]

        def E(s):
            need(st.recognised, "E()")
            return sem.embed_row(tok(s), extra=self.extra)

        def U(s):
            need(st.recognised, "U()")
            return sem.unembed_col(tok(s), extra=self.extra)

        def pos(i):
            need(st.pos is not None, "pos() (this model has no position-embedding tensor)")
            return sem.W_pos().sub(int(i)).read("current", self.extra)

        def unit(v):
            v = T(v, dtype=torch.float32)
            n = v.norm()
            return v / n if n > 0 else v

        def proj(v, onto):
            u = unit(onto)
            return (v @ u)[..., None] * u if v.dim() > 1 else (v @ u) * u

        def layer(l):
            l = int(l)
            if not -st.n_layers <= l < st.n_layers:
                raise ScriptError(f"layer {l} out of range (model has {st.n_layers})")
            return l % st.n_layers

        def head(l, h):
            l = layer(l)
            need(st.recognised and st.has_heads(l), f"head({l}, {h})")
            if not 0 <= int(h) < st.n_heads:
                raise ScriptError(f"head {h} out of range (layer has {st.n_heads})")
            return _NS(f"head({l}, {h})", **sem.head(l, int(h)))

        def neuron(l, n):
            l = layer(l)
            need(st.recognised and st.has_neurons(l), f"neuron({l}, {n})")
            if not 0 <= int(n) < st.mlp_width(l):
                raise ScriptError(f"neuron {n} out of range (layer has {st.mlp_width(l)})")
            N = sem.neuron(l, int(n))
            m = {"w_out": N["write"]}
            for r in N["reads"]:
                key = {"in": "w_in", "gate": "gate", "up": "up"}[r["kind"]]
                m[key] = r["w"]
                if r["b"] is not None:
                    m["b_" + r["kind"]] = r["b"]
            return _NS(f"neuron({l}, {n})", **m)

        def mlp(l):
            l = layer(l)
            need(st.recognised and bool(st.layers[l].mlp), f"mlp({l})")
            return _NS(f"mlp({l})", **sem.mlp(l))

        def attn(l):
            l = layer(l)
            need(st.recognised and bool(st.layers[l].attn), f"attn({l})")
            return _NS(f"attn({l})", **sem.attn(l))

        def P(name):
            if name not in ws.store:
                raise ScriptError(f"unknown tensor {name!r}")
            return ws.ref(name)

        def rank1(target, u, v):
            ref = self._ref_of(target)
            u, v = T(self._mat(u), dtype=torch.float32), T(self._mat(v), dtype=torch.float32)
            if len(ref.shape) != 2:
                if len(ref.shape) > 2 and tuple(ref.shape) == tuple(self.ws.shape(ref.name)):      # e.g. [head, d_head, d_model]
                    self.write(ref, torch.outer(u, v).reshape(ref.shape), operator.add)
                    return
                raise ScriptError(f"rank1 needs a 2-D view; this one has shape {list(ref.shape)}")
            edit_preflight(ws,self._stored()+u.numel()+v.numel())
            self._add(ref.name, ref.lowrank_piece(u, v))

        def lowrank_add(target, A, B):
            ref = self._ref_of(target)
            a,b=T(self._mat(A),dtype=torch.float32),T(self._mat(B),dtype=torch.float32)
            edit_preflight(ws,self._stored()+a.numel()+b.numel())
            self._add(ref.name, ref.lowrank_piece(a,b))

        def orthogonalize(r, layers=None):
            need(st.recognised, "orthogonalize()")
            r = unit(self._mat(r))
            for ref in sem.writers(None if layers is None else [layer(l) for l in layers]):
                if len(ref.shape) == 1:
                    b = self.read(ref)
                    self.write(ref, b - (b @ r) * r)
                else:
                    wr = ref.contract(r, axis=1, extra=self.extra)
                    if bool(wr.any()):
                        edit_preflight(ws,self._stored()+wr.numel()+r.numel())
                        self._add(ref.name, ref.lowrank_piece(-wr, r))

        def _measurement(name):
            from . import directions as DR
            try:
                meta, good, bad = DR.load(ws, name)
                DR.matches(ws, meta, name)
            except DR.DirectionError as e:
                raise ScriptError(str(e)) from None
            edit_preflight(ws, scratch=(good.numel() + bad.numel()) * 12)
            return good, bad

        def _called(function, method, parameters, name, result):
            """List one successful call that read a measurement (wsp.directions_record writes them into the
            patch's reproduction record): what was asked, the measurement's hashes, and what it did."""
            from . import directions as DR
            from . import directions_search as DSR
            try:
                identity = DSR.measurement_identity(ws, name)
            except DR.DirectionError as e:
                raise ScriptError(str(e)) from None
            self.calls.append(DSR._plain({"function": function, "method": method, "parameters": parameters,
                                          "measurement": identity, "result": result}))

        def ablate_direction(name, direction_index=None, weight_distributions=None, orthogonalize_direction=True,
                             row_normalization="none", full_normalization_lora_rank=3, expert_modules="all"):
            """Remove a saved behaviour direction from what the chosen layers' attention and MLP output matrices
            write (wsp.directions): low-rank pieces added here, so Session.apply commits, scopes and tests them."""
            from . import directions as DR
            need(st.recognised, "ablate_direction()")
            params = {"direction": name, "direction_index": direction_index, "weight_distributions": weight_distributions,
                      "orthogonalize_direction": orthogonalize_direction, "row_normalization": row_normalization,
                      "full_normalization_lora_rank": full_normalization_lora_rank}
            if expert_modules != "all":                  # the default stays out of the parameters (and the record)
                params["expert_modules"] = expert_modules

            def add(ref, A, B):
                if not ws.editable(ref.name):
                    raise ScriptError(f"{ref.name} is not a floating-point tensor and cannot be edited")
                edit_preflight(ws, self._stored() + A.numel() + B.numel())
                self._add(ref.name, ref.lowrank_piece(A, B))
            try:
                rep = DR.ablate(ws, params, extra=self.extra, add=add, stored=self._stored, progress=self.progress)
            except DR.DirectionError as e:
                raise ScriptError(str(e)) from None
            comps = rep["components"]
            line = (f"ablate_direction({rep['direction']!r}): {_counted(comps)} in {len({c['layer'] for c in comps})} "
                    f"layers, {rep['stored']:,} values stored, row_normalization {rep['row_normalization']}, "
                    f"{len(rep['skipped'])} skipped")
            if len(self.out) < 400:
                self.out.append(line)
            for c in comps:
                if "full_error" in c and len(self.out) < 400:
                    self.out.append(f"  {_where(c, comps)}: weight {c['weight']:.6g}, full_error {c['full_error']:.6g}")
            p = DR.normalize(params)
            _called("ablate_direction", "ablation", p, p["direction"],
                    {"stored": rep["stored"], "components": rep["components"], "skipped": rep["skipped"]})

        def ablate_ara(name, start_layer_index=0, end_layer_index=None, preserve_good_behavior_weight=1.0,
                       steer_bad_behavior_weight=0.01, overcorrect_relative_weight=0.0, neighbor_count=1,
                       preserve_row_magnitudes=True, lora_rank=50, n_optimization_steps=5, learning_rate=1.0,
                       max_iter=20, history_size=10, seed=0, expert_modules="all"):
            """Fit a low-rank change to the output matrices of a layer range from a measurement's saved module inputs
            and outputs (wsp.directions_ara): pieces added here, so Session.apply commits, scopes and tests them."""
            from . import directions as DR
            from . import directions_ara as ARA
            need(st.recognised, "ablate_ara()")
            params = {"method": "ara", "direction": name, "start_layer_index": start_layer_index,
                      "end_layer_index": end_layer_index, "preserve_good_behavior_weight": preserve_good_behavior_weight,
                      "steer_bad_behavior_weight": steer_bad_behavior_weight,
                      "overcorrect_relative_weight": overcorrect_relative_weight, "neighbor_count": neighbor_count,
                      "preserve_row_magnitudes": preserve_row_magnitudes, "lora_rank": lora_rank,
                      "n_optimization_steps": n_optimization_steps, "learning_rate": learning_rate,
                      "max_iter": max_iter, "history_size": history_size, "seed": seed}
            if expert_modules != "all":                  # the default stays out of the parameters (and the record)
                params["expert_modules"] = expert_modules

            def add(ref, A, B):
                if not ws.editable(ref.name):
                    raise ScriptError(f"{ref.name} is not a floating-point tensor and cannot be edited")
                edit_preflight(ws, self._stored() + A.numel() + B.numel())
                self._add(ref.name, ref.lowrank_piece(A, B))
            try:
                rep = ARA.ablate(ws, params, extra=self.extra, add=add, stored=self._stored, progress=self.progress)
            except DR.DirectionError as e:
                raise ScriptError(str(e)) from None
            comps = rep["components"]
            line = (f"ablate_ara({rep['direction']!r}): {_counted(comps)} in {len({c['layer'] for c in comps})} "
                    f"layers, {rep['stored']:,} values stored, lora_rank {rep['lora_rank']}, {len(rep['skipped'])} skipped")
            if len(self.out) < 400:
                self.out.append(line)
            every = comps + rep["skipped"]
            for c in sorted(every, key=lambda c: (c["layer"], list(DR.COMPONENTS).index(c["component"]))):
                if len(self.out) >= 400:
                    break
                if c["skipped"] is not None:
                    self.out.append(f"  {_where(c, every)}: skipped ({c['skipped']})")
                else:
                    self.out.append(f"  {_where(c, every)}: loss {c['loss_before']:.6g} -> {c['loss_after']:.6g}, "
                                    f"scale {c['scale_min']:.6g}..{c['scale_max']:.6g}, mismatch good "
                                    f"{c['mismatch_good']:.3g} bad {c['mismatch_bad']:.3g}")
            p = ARA.normalize(params)
            _called("ablate_ara", "ara", p, p["direction"],
                    {"stored": rep["stored"], "components": rep["components"], "skipped": rep["skipped"]})

        def direction(name, k):
            good, bad = _measurement(name)
            from . import directions as DR
            k = int(k)
            if not 0 <= k <= st.n_layers:
                raise ScriptError(f"index {k} out of range (a measurement has indices 0..{st.n_layers})")
            out = DR.table(good, bad, True)[k].clone()
            _called("direction", None, {"name": name, "k": k}, name, None)
            return out

        def direction_means(name):
            good, bad = _measurement(name)
            _called("direction_means", None, {"name": name}, name, None)
            return good.clone(), bad.clone()

        def copy_head(src, dst):
            a, b = head(*src).members, head(*dst).members
            for k in a:
                if k in b:
                    self.write(b[k], self.read(a[k]))

        def scale_head(lh, f):
            self.write(head(*lh).members["O"], f, operator.mul)

        def copy_neuron(src, dst):
            a, b = neuron(*src).members, neuron(*dst).members
            for k in a:
                if k in b:
                    self.write(b[k], self.read(a[k]))

        def scale_neuron(ln, f):
            self.write(neuron(*ln).members["w_out"], f, operator.mul)

        def _neuron_ids(l, ns):
            l = layer(l)
            need(st.recognised and st.has_neurons(l), f"neurons of layer {l}")
            ids = np.unique(np.asarray([int(n) for n in ns], dtype=np.int64))
            w = st.mlp_width(l)
            bad = ids[(ids < 0) | (ids >= w)]
            if len(bad):
                raise ScriptError(f"neuron {int(bad[0])} out of range (layer has {w})")
            return l, ids

        def scale_neurons(l, ns, f):
            """What each neuron writes, times f: one read per window of neurons, one stored piece per run."""
            l, ids = _neuron_ids(l, ns)
            part = st.layers[l].mlp["W_out"]
            name = part.name
            axis, positions, template = S.neuron_axis(st, part)
            k = S.kept_before(template, axis)
            vec = [len(r) for d, r in enumerate(template) if isinstance(r, range) and d != axis]
            f = _neuron_factor(f, vec, k)
            if not len(ids) or (isinstance(f, float) and f == 1.0):
                return
            if not ws.editable(name):
                raise ScriptError(f"{name} is not a floating-point tensor and cannot be edited")
            pos = np.sort(positions.start + ids * positions.step)
            ws_ = S.read_windows(ws, name, axis, pos, S.other_size(template, axis))   # by packed block if quantized
            for i, w in enumerate(ws_):
                self.progress(i, len(ws_))
                lo, hi = int(w[0]), int(w[-1]) + 1
                region = S.with_axis(template, axis, range(lo, hi))
                cur = self.read(Ref(ws, name, region))
                stored = self._stored()
                for a, b in S.runs(w):
                    c = cur.narrow(k, a - lo, b - a)
                    d = c * f - c
                    if bool(d.any()):
                        stored += d.numel()
                        edit_preflight(ws, stored)
                        self._add(name, D.block(S.with_axis(region, axis, range(a, b)), d))

        def _neuron_factor(f, vec, k):
            """A number, or factors for each weight of one neuron's write vector (shape `vec`), the same for every
            neuron -- what scale_neuron((L, n), f) has always taken. One factor per neuron is not supported."""
            if isinstance(f, (int, float)) or (isinstance(f, (torch.Tensor, np.ndarray)) and f.ndim == 0):
                return float(f)
            try:
                t = torch.as_tensor(np.asarray(f, dtype=np.float32) if not isinstance(f, torch.Tensor) else f,
                                    dtype=torch.float32)
                t = t.expand(vec)
            except (RuntimeError, TypeError, ValueError):
                raise ScriptError(f"scale_neurons takes a number, or factors shaped like one neuron's write vector "
                                  f"{list(vec)} (the same for every neuron); for one factor per neuron, call "
                                  f"scale_neurons once per factor") from None
            return t.unsqueeze(k)

        def revert_neurons(l, ns):
            l, ids = _neuron_ids(l, ns)
            for p in S.neuron_parts(st, l):
                axis, positions, template = S.neuron_axis(st, p)
                self._revert_positions(p.name, template, axis, np.sort(positions.start + ids * positions.step))

        def revert(view):
            """Put a view (or every part of head(L, h) / neuron(L, n)) back to its base values."""
            self.revert(view)

        def base(view):
            return self.read(self._ref_of(view), "base")

        def select(**sel):
            """Neurons chosen from the atlas, strongest first, as (layer, index) pairs."""
            from .atlas import Atlas, select as _select
            try:
                r = _select(Atlas.of(ws), sel)
            except (ValueError, KeyError) as e:
                raise ScriptError(str(e)) from None
            out = [(int(a), int(b)) for a, b in zip(r["layer"], r["idx"])]
            self._work(len(out) * PY_ITEM, "select()")
            self._charge(sys.getsizeof(out) + sum(_value_bytes(t) for t in out), "select()")
            return out

        def shift_tokens(tokens, nats):
            """Raise (lower) the output score of tokens by `nats` in every context."""
            from .atlas import guide
            need(st.recognised, "shift_tokens()")
            ids = [tok(t) for t in (tokens if isinstance(tokens, (list, tuple)) else [tokens])]
            plan = guide.shift_plan(ws)
            if plan["mode"] == "bias":
                ref = sem.ref(st.unembed_bias)
                for t in ids:
                    self.write(ref.sub(t), float(nats), operator.add)
            elif plan["mode"] == "direction":
                w = guide.bias_direction(ws)
                for t in ids:
                    self.write(sem.W_U().sub((slice(None), t)), float(nats) * w, operator.add)
            else:
                raise ScriptError(plan["note"])

        def word_tokens(word):
            """Every token that is this word up to case and surrounding spaces."""
            from .atlas.select import token_ids
            out = [int(i) for i in token_ids(ws, word=word)]
            self._work(st.d_vocab, "word_tokens()")                # a pass over the vocabulary
            self._charge(sys.getsizeof(out) + sum(_value_bytes(i) for i in out), "word_tokens()")
            return out

        def toks(s):
            out = ws.tok.encode_token(s)
            self._work((len(s) if isinstance(s, str) else 0) + len(out), "toks()")
            self._charge(sys.getsizeof(out) + sum(_value_bytes(i) for i in out), "toks()")
            return out

        def _ids(prompt):
            return ws.encode(prompt) if isinstance(prompt, str) else [int(i) for i in prompt]

        def act(prompt, hook, pos=-1):
            r = ws.runner
            if not r.available(hook):
                raise ScriptError(f"hook {hook!r} is not available for this model")
            _, cache = r.run(_ids(prompt)[-(r.max_ctx or 10 ** 9):], "current", self.extra, names=[hook])
            if hook not in cache:
                raise ScriptError(f"unknown hook {hook!r}")
            return cache[hook][0, pos].clone()

        def logits(prompt):
            # Only the last row is asked for (and crosses a model-worker boundary); the copy
            # keeps no larger [tokens, vocabulary] result alive while the script holds it.
            row = ws.runner.logits(_ids(prompt), "current", self.extra, tail=1)[-1]
            edit_preflight(ws, scratch=row.numel() * max(12, row.element_size() * 3))
            return row.clone()

        def fmt(v):
            if isinstance(v, Ref):
                if v.size > 4_000_000:
                    return f"view {v.address()} shape={list(v.shape)} (index it to print values)"
                v = self.read(v)
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return f"{float(v):.5g}"
                if v.numel() <= 64:
                    return str([round(float(x), 4) for x in v.flatten()]) + (f" shape={list(v.shape)}" if v.dim() > 1 else "")
                return (f"tensor{list(v.shape)} mean={float(v.float().mean()):.4g} std={float(v.float().std()):.4g} "
                        f"min={float(v.min()):.4g} max={float(v.max()):.4g}")
            return f"{v:.5g}" if isinstance(v, float) else self._show(v)

        def _print(*a):
            if len(self.out) < 400:
                parts, n = [], 0
                for x in a:                               # each part is cut at MAX_PRINT_CHARS; so is the line
                    if n > MAX_PRINT_CHARS:
                        parts.append(f"... ({len(a) - len(parts)} more values; cut at {MAX_PRINT_CHARS:,} characters)")
                        break
                    parts.append(fmt(x))
                    n += len(parts[-1]) + 1
                self.out.append(" ".join(parts))

        def lowrank(M, r):
            u, s, vh = torch.linalg.svd(M, full_matrices=False)
            return (u[:, :r] * s[:r]) @ vh[:r]

        def randn(*shape, seed=0):
            return torch.randn(big(*shape), generator=torch.Generator().manual_seed(int(seed)))

        def tensor(x):
            def count(value):
                if isinstance(value,torch.Tensor):return value.numel()
                if isinstance(value,(list,tuple)):return sum(count(v) for v in value)
                return 1
            if getattr(ws,'_runtime_client',None):big(count(x))
            return T(x,dtype=torch.float32)

        def where(condition,*values):
            if getattr(ws,'_runtime_client',None):
                if values:
                    tensors=[x for x in (condition,*values) if isinstance(x,torch.Tensor)]
                    shape=torch.broadcast_shapes(*(x.shape for x in tensors))
                    edit_preflight(ws,scratch=math.prod(shape)*max(12,*(x.element_size()*3 for x in tensors)))
                else:edit_preflight(ws,scratch=condition.numel()*max(1,condition.ndim)*8)
            return torch.where(condition,*values)

        def arange(*args):
            start,end,step=(0,args[0],1) if len(args)==1 else (args[0],args[1],args[2] if len(args)>2 else 1)
            if getattr(ws,'_runtime_client',None):big(max(0,math.ceil((end-start)/step)))
            return torch.arange(*args)

        def combine(fn,xs,dim=0,**kw):
            xs=list(xs);edit_preflight(ws,scratch=sum(x.numel()*max(12,x.element_size()*3) for x in xs))
            return fn(xs,kw.get('axis',dim))

        def diag(v):
            v=fl(v)
            if getattr(ws,'_runtime_client',None):big(v.numel()**2 if v.ndim==1 else min(v.shape))
            return torch.diag(v)

        def red(f):
            def g(x, dim=None, **kw):
                dim = kw.get("axis", dim)
                x = T(x) if not isinstance(x, torch.Tensor) else x
                return f(x) if dim is None else f(x, dim)
            return g

        def extreme(name, py):
            def f(*a, key=None, **kw):
                if len(a) == 1 and isinstance(a[0], torch.Tensor) and key is None and "default" not in kw:
                    return getattr(a[0], name)() if not kw else getattr(a[0], name)(**kw).values
                if kw.keys() - {"default"}:
                    raise ScriptError(f"{name}() of Python values takes key= and default=; dim= is for one tensor")
                cands = a if len(a) > 1 else self._items(a[0]) if a else ()
                self._work(self._deep(cands, f"{name}()"), f"{name}()")
                return py(*([cands] if len(a) <= 1 and a else a), key=self._key(key), **kw)
            return f

        _max, _min = extreme("max", builtins.max), extreme("min", builtins.min)
        _tsum = red(torch.sum)

        def _sum(x, *a, start=0, **kw):
            """sum(tensor, dim=None) as before; sum of a list of tensors adds them (sum([E(t) for t in ...]), and
            sum(xs, start) as in Python); sum of numbers is a 0-d tensor, as it has always been."""
            if isinstance(x, torch.Tensor) or kw:
                return _tsum(x, *a, **kw)
            xs = self._items(x)
            vectors = any(isinstance(v, torch.Tensor) and v.dim() > 0 for v in xs)
            if not vectors and (not isinstance(start, (int, float)) or start != 0):
                vectors = True
            if not vectors:                               # numbers: a positional second argument is dim, as before
                self._work(len(xs), "sum()")
                return _tsum(list(xs), *a)
            total = a[0] if a else start
            for v in xs:
                self._tick(None)                          # one addition, maybe of two vectors, per item
                total = self._binary(operator.add, total, v)
            return total

        def truth(name, py):
            def f(x):
                if isinstance(x, torch.Tensor):
                    return bool(getattr(x, name)())
                xs = self._items(x)
                self._work(len(xs) * PY_ITEM, f"{name}()")
                return py(self._truth(v) for v in xs)
            return f

        def _list(x=(), make=list, slot=8, what="list()"):
            """list / tuple / set of an iterable: its items (bounded) in a new container, charged."""
            xs = self._items(x)
            self._charge(64 + slot * len(xs), what)
            self._work(self._deep(xs, what) if make is set else len(xs), what)       # a set hashes every item
            return xs if make is list and xs is not x and type(xs) is list else make(xs)

        def _sorted(x, key=None, reverse=False):
            xs = self._items(x)
            self._charge(64 + 8 * len(xs), "sorted()")
            self._work(self._sort_cost(xs), "sorted()")
            return builtins.sorted(xs, key=self._key(key), reverse=bool(reverse))

        def _reversed(x):
            return x.flip(0) if isinstance(x, torch.Tensor) else _list(x, lambda xs: list(reversed(xs)), 8, "reversed()")

        def _int(x=0, *a):
            if isinstance(x, str):
                self._work(len(x), "int()")
            return int(x, *a)

        def _float(x=0.0):
            if isinstance(x, str):
                self._work(len(x), "float()")
            return float(x)

        def _dict(*a, **kw):
            if len(a) > 1:
                raise ScriptError("dict() takes at most one positional argument")
            xs = self._items(a[0]) if a else ()
            self._charge(64 + 80 * (len(xs) + len(kw)), "dict()")
            self._work(self._deep(xs, "dict()") + len(kw), "dict()")
            return {**dict(xs), **kw}

        for f in (rank1, lowrank_add, _print, revert, base):
            f.lazy = True
        w_e = lambda: sem.W_E()  # noqa: E731
        w_u = lambda: sem.W_U()  # noqa: E731
        fl = lambda x: T(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x  # noqa: E731
        return {
            "tok": tok, "toks": toks, "detok": lambda i: ws.tok.token_str(int(i)),
            "E": E, "U": U, "pos": pos, "P": P, "head": head, "neuron": neuron, "mlp": mlp, "attn": attn,
            "W_E": w_e, "W_U": w_u,
            "n_vocab": st.d_vocab, "n_ctx": st.n_ctx, "d_model": st.d_model, "n_heads": st.n_heads,
            "d_head": st.d_head, "d_mlp": st.d_mlp, "n_layers": st.n_layers, "pi": math.pi, "inf": math.inf,
            "zeros": lambda *s: torch.zeros(big(*s)), "ones": lambda *s: torch.ones(big(*s)),
            "full": lambda s, x: torch.full(big(s if isinstance(s, (tuple, list)) else (s,)), float(x)),
            "eye": lambda n: torch.eye(big(n, n)[0]), "arange": arange,
            "onehot": lambda i, n: F.one_hot(torch.tensor(int(i)), big(n)[0]).float(),
            "linspace": lambda a, b, n: torch.linspace(a, b, big(n)[0]), "randn": randn,
            "tensor": tensor, "array": tensor,
            "unit": unit, "normalize": unit, "norm": lambda v, *a, **k: fl(v).norm(*a, **k),
            "dot": lambda a,b:self._binary(operator.matmul,fl(a),fl(b)), "matmul": lambda a,b:self._binary(operator.matmul,a,b),
            "cossim": lambda a, b: F.cosine_similarity(fl(a), fl(b), dim=-1),
            "outer": lambda a,b:self._binary(torch.outer,fl(a),fl(b)), "proj": proj,
            "reject": lambda v, r: v - proj(v, r),
            "cat": lambda xs, dim=0, **k: combine(torch.cat,xs,dim,**k),
            "concatenate": lambda xs, dim=0, **k: combine(torch.cat,xs,dim,**k),
            "stack": lambda xs, dim=0, **k: combine(torch.stack,xs,dim,**k),
            "diag": diag, "transpose": lambda m: m.transpose(-1, -2),
            "reshape": lambda x, s: x.reshape(s), "where": where,
            "abs": lambda x: x.abs() if isinstance(x, torch.Tensor) else abs(x),
            "sign": lambda x: torch.sign(fl(x)), "sqrt": lambda x: torch.sqrt(fl(x)), "exp": lambda x: torch.exp(fl(x)),
            "log": lambda x: torch.log(fl(x)), "tanh": lambda x: torch.tanh(fl(x)), "sin": lambda x: torch.sin(fl(x)),
            "cos": lambda x: torch.cos(fl(x)), "relu": lambda x: F.relu(fl(x)), "gelu": lambda x: F.gelu(fl(x)),
            "clip": lambda x, lo=None, hi=None: torch.clamp(fl(x), lo, hi),
            "clamp": lambda x, lo=None, hi=None: torch.clamp(fl(x), lo, hi),
            "maximum": lambda a,b:self._binary(torch.maximum,fl(a),fl(b)), "minimum": lambda a,b:self._binary(torch.minimum,fl(a),fl(b)),
            "sum": _sum, "mean": red(torch.mean), "max": _max, "min": _min,
            "any": truth("any", builtins.any), "all": truth("all", builtins.all), "sorted": _sorted,
            "reversed": _reversed, "tuple": lambda x=(): _list(x, tuple, 8, "tuple()"), "dict": _dict,
            "set": lambda x=(): _list(x, set, 64, "set()"),
            "argmax": lambda x, *a: int(x.argmax()) if not a else x.argmax(*a),
            "argmin": lambda x, *a: int(x.argmin()) if not a else x.argmin(*a),
            "argsort": lambda x, descending=False: torch.argsort(x, descending=descending),
            "topk": lambda x, k: tuple(torch.topk(x, int(k))), "softmax": lambda x, dim=-1: torch.softmax(x, dim),
            "svd": lambda m: tuple(torch.linalg.svd(m, full_matrices=False)), "pinv": torch.linalg.pinv,
            "lowrank": lowrank, "act": act, "logits": logits, "orthogonalize": orthogonalize, "rank1": rank1,
            "ablate_direction": ablate_direction, "ablate_ara": ablate_ara, "direction": direction,
            "direction_means": direction_means,
            "lowrank_add": lowrank_add, "copy_head": copy_head, "scale_head": scale_head, "copy_neuron": copy_neuron,
            "scale_neuron": scale_neuron, "scale_neurons": scale_neurons, "select": select, "shift_tokens": shift_tokens,
            "word_tokens": word_tokens, "revert": revert, "revert_neurons": revert_neurons, "base": base,
            "print": _print, "range": range, "enumerate": enumerate, "zip": zip, "len": len, "int": _int,
            "float": _float, "round": round, "list": lambda x=(): _list(x), "str": lambda x="": self._text(x), "bool": bool,
            "shape": lambda x: tuple(x.shape),
        }

    def _mat(self, v):
        """An argument that may be a view: read it."""
        return self.read(v) if isinstance(v, Ref) else v


def _counted(comps) -> str:
    """The summary line's count of an ablate call's report: "<n> components", one per (layer, component, module) as
    the plan counts them (a layer's routed experts together); "<m> modules (<n> components)" when the modules are
    more (routed experts, each its own module)."""
    from . import directions as DR
    rows = len({(c["layer"], c["component"], DR.routed_path(c.get("module") or "")) for c in comps})
    return f"{rows} components" if rows == len(comps) else f"{len(comps)} modules ({rows} components)"


def _where(c, entries) -> str:
    """``L<layer> <component>`` of a per-module line, with the module path when the layer has more than one module
    of that component among ``entries``."""
    n = sum(1 for e in entries if (e["layer"], e["component"]) == (c["layer"], c["component"]))
    return f"L{c['layer']} {c['component']}" + (f" {c['module']}" if n > 1 and c.get("module") else "")


def _slab_loop(ref: Ref, limit: int) -> str:
    """The loop that reads `ref` in row slabs along its first kept axis, each slab under `limit` values (one line)."""
    rows = ref.region[ref.kept[0]] if ref.kept else range(1)
    row = max(1, ref.size // max(1, len(rows)))
    step = max(0, int(limit) // row)
    if step < 1:
        return (f"one row is {row:,} values and the values held already leave no room; rebind the names that hold "
                f"earlier whole reads (x = 0) first")
    lo, hi, step = rows.start, rows.stop, min(step, len(rows))
    parts = []
    for d, r in enumerate(ref.region):
        if d == (ref.kept[0] if ref.kept else -1):
            parts.append(f"a:min(a + {step}, {hi})")
        elif isinstance(r, range):
            full = r.start == 0 and r.step == 1 and r.stop == ref.ws.shape(ref.name)[d]
            parts.append(":" if full else f"{r.start}:{r.stop}" + (f":{r.step}" if r.step != 1 else ""))
        else:
            parts.append(str(int(r)))
    return (f'for a in range({lo}, {hi}, {step}): X = P("{ref.name}")[{", ".join(parts)}]  '
            f'(at most {step:,} rows of {row:,} values per slab)')


def _basic(idx) -> bool:
    items = idx if isinstance(idx, tuple) else (idx,)
    return all(isinstance(i, (int, slice)) and not isinstance(i, bool) for i in items)


def run_script(ws, script: str, extra: dict | None = None, request: str | None = None, stopped=None) -> dict:
    """Run a weight script against the workspace's current weights. Nothing is committed.

    ``stopped()``, when given, is asked at every progress point (before each statement and each component of
    ablate_direction / ablate_ara); when it is true the script raises ScriptStopped.

    Returns {"deltas": {tensor: [pieces]}, "output": printed text}."""
    return Interp(ws, extra, request, stopped).run(script)
