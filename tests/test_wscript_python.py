"""Weight scripts: the Python built-ins and methods they accept, the sandbox they must keep, and whole-view reads
of a quantized model.

Model-free: a tiny random model of this project's kind (tests/test_wsp.py builds the same), and the fake GGUF store
of tests/selection_fixtures.py with its single-read limit and the slab size lowered so that every matrix is
"large". No quantized file, decoder or real model is used.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import selection_fixtures as F  # noqa: E402
from wsp import workspace as W  # noqa: E402
from wsp import wscript  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402
from wsp.wscript import FUNCTION_DOC, ScriptError, run_script  # noqa: E402


@pytest.fixture()
def ws(tmp_path):
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghij"), "char")
    cfg = Config(d_vocab=10, n_ctx=12, d_model=16, n_layers=2, n_heads=2, d_mlp=32, name="rnd")
    Workspace.create(str(tmp_path / "rnd"), cfg, tok, init_params(cfg, seed=1, std=0.3))
    s = Session(str(tmp_path))
    s.open("rnd")
    return s.ws


def out(ws, script):
    return run_script(ws, script)["output"].splitlines()


# ---------------------------------------------------------------------------------------------- Python additions
def test_sorted_with_reverse_and_key(ws):
    assert out(ws, "print(sorted([3, -5, 1]))\nprint(sorted([3, -5, 1], reverse=True))\n"
                   "print(sorted([3, -5, 1], key=abs))\nprint(sorted([(0.5, 'b'), (2.0, 'a')], reverse=True))") == \
        ["[-5, 1, 3]", "[3, 1, -5]", "[1, 3, -5]", "[(2.0, 'a'), (0.5, 'b')]"]


def test_sorting_neurons_by_a_score_without_lambda(ws):
    lines = out(ws, "s = [(float(neuron(0, n).w_out.norm()), n) for n in range(d_mlp)]\n"
                    "top = sorted(s, reverse=True)[:3]\nprint([n for sc, n in top])\n"
                    "xs = [n for sc, n in s]\nxs.sort(key=abs, reverse=True)\nprint(xs[0])")
    norms = [(float(ws.ref(ws.struct.layers[0].mlp["W_out"].name).read()[n].norm()), n) for n in range(32)]
    assert lines[0] == str([n for _, n in sorted(norms, reverse=True)[:3]])
    assert lines[1] == "31"


def test_min_max_sum_abs_any_all_reversed_tuple(ws):
    assert out(ws, "xs = [4, -7, 2]\nprint(max(xs), min(xs), max(xs, key=abs), min(3, 9), max([], default=0))\n"
                   "print(abs(-2), any([0, 0, 1]), all([1, 0]), any(x > 3 for x in xs))\n"
                   "print(reversed(xs), tuple(xs), list(reversed(range(3))))") == \
        ["4 -7 -7 3 0", "2 True False True", "[2, -7, 4] (4, -7, 2) [2, 1, 0]"]


def test_sum_of_vectors_numbers_and_a_tensor(ws):
    r = out(ws, "v = sum([E('a'), E('b')])\nprint(float((v - E('a') - E('b')).abs().max()) < 1e-6)\n"
                "print(sum([1, 2, 3]))\nprint(sum(x * x for x in range(4)))\nprint(sum(ones(2, 3), dim=0))\n"
                "print(sum([E('a')], E('b')).shape == E('a').shape)")
    assert r == ["True", "6", "14", "[2.0, 2.0, 2.0]", "True"]


def test_tensor_reductions_keep_their_meaning(ws):
    assert out(ws, "x = tensor([[1, 5], [7, 2]])\nprint(max(x), min(x), max(x, dim=0))\n"
                   "print(x.max(dim=1).values, x.max(dim=1).indices, any(x > 6), all(x > 0))\n"
                   "print(reversed(tensor([1, 2, 3])))") == \
        ["7 1 [7.0, 5.0]", "[5.0, 7.0] [1.0, 0.0] True True", "[3.0, 2.0, 1.0]"]


def test_dicts_and_comprehensions(ws):
    assert out(ws, "d = {'a': 1}\nd['b'] = 2\nd['a'] += 10\nprint(d, d.get('z', 0), 'a' in d, sorted(d.keys()))\n"
                   "c = {n: n * n for n in range(3)}\nprint(c, dict(c), dict([(1, 2)], x=3))\n"
                   "for k, v in sorted(d.items()):\n    print(k, v)\n"
                   "s = {x % 2 for x in range(5)}\nprint(sorted(s), set([1, 1]), len({1, 2}))") == \
        ["{'a': 11, 'b': 2} 0 True ['a', 'b']", "{0: 0, 1: 1, 2: 4} {0: 0, 1: 1, 2: 4} {1: 2, 'x': 3}", "a 11", "b 2",
         "[0, 1] {1} 2"]


def test_list_methods_and_item_assignment(ws):
    assert out(ws, "xs = [3, 1, 2]\nxs.append(0)\nxs.insert(0, 9)\nprint(xs.pop(), xs.index(1), xs.count(3))\n"
                   "xs.remove(9)\nxs.reverse()\nprint(xs)\nxs[0] += 5\nys = xs.copy()\nys.clear()\nprint(xs, ys)\n"
                   "t = (1, 2, 1)\nprint(t.count(1), t.index(2))") == \
        ["0 2 1", "[2, 1, 3]", "[7, 1, 3] []", "2 1"]


def test_strings_and_formats(ws):
    assert out(ws, "x = 3.14159\nprint(f'{x:.2f}', f'{7:>4}|', f'{\"a\"!r}', f'{norm(ones(4)):.1f}')\n"
                   "print('{} and {name}'.format(1, name='two'), '{:.3f}'.format(x))\n"
                   "print(','.join(['a', 'b']), ' a '.strip(), 'a-b'.split('-'), 'ab'.startswith('a'), 'x'.zfill(3))\n"
                   "print('%d / %.1f' % (3, 2.25), x is None, 2 not in [1, 3])") == \
        ["3.14    7| 'a' 2.0", "1 and two 3.142", "a,b a ['a', 'b'] True 00x", "3 / 2.2 False True"]


def test_the_reference_names_what_is_available(ws):
    for name in ("sorted(xs, reverse=True, key=f)", "reversed", "any", "all", "tuple", "dict", "set", ".setdefault",
                 ".sort(reverse=, key=)", "no lambda", "320M values"):
        assert name in FUNCTION_DOC, name
    for name in wscript._PY_BUILTINS.split():
        assert name in FUNCTION_DOC, name
        run_script(ws, f"f = {name}")                            # every built-in the reference names exists


def test_the_brief_carries_the_script_reference(tmp_path, ws):
    s = Session(str(tmp_path))
    s.open("rnd")
    brief = s.brief("tools")
    assert "## Weight scripts" in brief and "sorted(xs, reverse=True, key=f)" in brief and "no lambda" in brief


# ---------------------------------------------------------------------------------------------- sandbox
ESCAPES = [
    ("x = ().__class__", "private attributes"),
    ("x = [].__class__.__base__", "private attributes"),
    ("f = sorted\nx = f.__self__", "private attributes"),
    ("x = [1].append.__self__", "private attributes"),
    ("x = [1].append.__name__", "private attributes"),
    ("x = '{0.__class__}'.format(1)", "format fields may only name an argument"),
    ("x = '{0[0]}'.format([1])", "format fields may only name an argument"),
    ("x = '{.real}'.format(1)", "format fields may only name an argument"),
    ("x = 'a'.format_map({})", "str has no method 'format_map'"),
    ("x = {}.fromkeys([1])", "dict has no method 'fromkeys'"),
    ("x = (1).bit_length()", "attribute access on int is not available"),
    ("x = tensor([1.0]).max(dim=0).__class__", "private attributes"),
    ("x = tensor([1.0]).max(dim=0).n_fields", "attribute access on max is not available"),
    ("x = tensor([1.0]).data_ptr()", "tensor attribute 'data_ptr' is not available"),
    ("x = tensor([1.0]).numpy()", "tensor attribute 'numpy' is not available"),
    ("x = (y for y in [1]).gi_frame", "list has no method 'gi_frame'"),
    ("x = open('/etc/passwd')", "built-in 'open' is not part of weight scripts"),
    ("x = getattr(1, 'real')", "built-in 'getattr' is not part of weight scripts"),
    ("x = eval('1')", "built-in 'eval' is not part of weight scripts"),
    ("exec('x = 1')", "built-in 'exec' is not part of weight scripts"),
    ("x = __import__('os')", "built-in '__import__' is not part of weight scripts"),
    ("x = type(1)", "built-in 'type' is not part of weight scripts"),
    ("x = vars()", "built-in 'vars' is not part of weight scripts"),
    ("x = globals()", "built-in 'globals' is not part of weight scripts"),
    ("x = compile('1', 'f', 'eval')", "built-in 'compile' is not part of weight scripts"),
    ("x = map(abs, [1])", "use a list comprehension"),
    ("import os", "Import statements are not part"),
    ("from os import path", "ImportFrom statements are not part"),
    ("x = lambda: 1", "there is no lambda"),
    ("x = sorted([1, 2], key=lambda v: -v)", "there is no lambda"),
    ("def f():\n    pass", "FunctionDef statements are not part"),
    ("while True:\n    pass", "While statements are not part"),
    ("x = {**{}}", "{**d} is not available"),
    ("x = np.load('a')", "function 'load' is not available"),
    ("x = torch.save", "function 'save' is not available"),
    ("x = math.__dict__", "private attributes"),
    ("x = sorted([1], key=3)", "key= takes a value function"),
    ("_x = 1", "names may not start with an underscore"),
]


@pytest.mark.parametrize("script,why", ESCAPES)
def test_sandbox_escapes_stay_refused(ws, script, why):
    with pytest.raises(ScriptError) as e:
        run_script(ws, script)
    assert why in str(e.value), str(e.value)


# Every bound below is lowered before its test runs, so that a run with the bound broken (a mutant, a regression)
# builds at most about 1 MB: the refused request is a few thousand items, never the size the shipped bound exists
# for. (A test that asked for 10 ** 9 items and relied on the guard took a mutant run to a 612 GB footprint.)
@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(wscript, "MAX_ITEMS", 1000)
    monkeypatch.setattr(wscript, "MAX_FORMAT_WIDTH", 1000)
    monkeypatch.setattr(wscript, "MAX_INT_BITS", 2000)
    monkeypatch.setattr(wscript, "MAX_PY_BYTES", 50_000)
    monkeypatch.setattr(wscript, "MAX_PRINT_CHARS", 100)


RESOURCE = [
    ("x = [0] * 5000", "too long"),
    ("x = 5000 * 'a'", "too long"),
    ("x = 'ab'\nfor i in range(12):\n    x = x + x", "too long"),
    ("x = [1]\nfor i in range(12):\n    x += x", "too long"),
    ("x = [1]\nfor i in range(12):\n    x.extend(x)", "would hold more than"),
    ("x = list(range(5000))", "too long"),
    ("x = sorted(range(5000))", "too long"),
    ("x = list(zip(range(5000), range(5000)))", "too long"),
    ("x = tuple(x for x in range(5000))", "too long"),
    ("x = 'x'.ljust(5000)", "width"),
    ("x = ('a' * 100).replace('a', 'b' * 100)", "replace() would build"),
    ("x = f'{1:>2000}'", "format spec"),
    ("x = '{:>2000}'.format(1)", "format spec"),
    ("x = '%2000d' % 1", "format spec"),
    ("x = '%(a)2000d' % {'a': 1}", "format spec"),
    ("x = '%*d' % (5000, 1)", "* width"),
    ("x = ('%900d' * 10) % tuple(range(10))", "% format would build"),
    ("x = ('{0:>900}' * 10).format(1)", "format() would build"),
    ("s = 'a' * 400\nx = f'{s}{s}{s}'", "f-string would be"),
    ("x = ','.join(['a' * 400] * 10)", "join() would build"),
    ("x = 7 ** 1000", "integer power"),
    ("x = 7 ** 200\ny = x ** 200", "integer power"),
    ("x = 2 ** 900\nfor i in range(10):\n    x = x * x", "integer product"),
]


@pytest.mark.parametrize("script,why", RESOURCE)
def test_resource_bombs_are_refused(ws, script, why, small, monkeypatch):
    monkeypatch.setattr(wscript, "MAX_PY_BYTES", 1 << 20)       # each bound on its own; the total still caps a broken one
    with pytest.raises(ScriptError) as e:
        run_script(ws, script)
    assert why in str(e.value), str(e.value)


# Many small values add up: each of these stays under every per-value bound and is refused only by MAX_PY_BYTES
# (50 KB here; each would build about 0.1 to 1 MB unbounded).
ACCUMULATE = [
    "x = []\nfor i in range(100):\n    x.append([0] * 900)",
    "x = [[0] * 900 for i in range(100)]",
    "y = list(range(900))\nx = [y[:] for i in range(100)]",
    "s = 'a' * 900\nx = [s.upper() for i in range(100)]",
    "x = [list(zip(range(900), range(900))) for i in range(5)]",
    "x = [sorted(range(900)) for i in range(20)]",
    "d = {}\nfor i in range(5000):\n    d[i] = i * 1.5",
    "x = {i: [0] * 500 for i in range(200)}",
    "x = [(i * 0.5, i) for i in range(5000)]",
    "x = [ones(900).tolist() for i in range(10)]",
    "y = list(range(200))\nx = [str(y) for i in range(100)]",
    "x = [f'{i:>900}' for i in range(100)]",
    "s = 'a b ' * 200\nx = [s.split() for i in range(30)]",
    "s = 'a' * 900\nx = ['%s' % s for i in range(100)]",
    "s = 'a' * 900\nx = ['{}'.format(s) for i in range(100)]",
    "y = ['ab'] * 300\nx = [','.join(y) for i in range(100)]",
    "y = [0] * 900\nx = [y.copy() for i in range(100)]",
    "y = [0] * 900\nx = [list(y) for i in range(100)]",
    "y = [0] * 900\nx = [tuple(y) for i in range(100)]",
    "y = [0] * 900\nx = [reversed(y) for i in range(100)]",
    "y = [(i, i) for i in range(100)]\nx = [dict(y) for i in range(100)]",
    "s = set(range(300))\nx = [s - {1} for i in range(100)]",
    "z = [0] * 900\nx = [[] for i in range(100)]\nfor y in x:\n    y.extend(z)",
    "d = {}\nfor i in range(5000):\n    d.setdefault(i, i * 1.5)",
    "s = 'abcdefghij' * 90\nx = [toks(s) for i in range(100)]",
    "x = []\nfor i in range(5000):\n    x.append(i * 1.5)",
    "s = 'a' * 90\nx = [s.replace('a', 'bb') for i in range(500)]",
    "x = ['a'.ljust(900) for i in range(100)]",
    "y = [0] * 400\nx = [y + y for i in range(100)]",
    "y = [0] * 900\nx = [sorted(y) for i in range(100)]",
]


@pytest.mark.parametrize("script", ACCUMULATE)
def test_python_values_are_bounded_in_all(ws, script, small):
    with pytest.raises(ScriptError, match="a script may hold"):
        run_script(ws, script)


def test_rebinding_a_name_gives_its_value_back(ws, small):
    """The bound counts what the script holds, not what it ever built: a name rebound in a loop frees its old value."""
    r = run_script(ws, "for i in range(200):\n    x = [0] * 900\ns = ''\nfor i in range(90):\n    s = s + 'abcdefghij'\n"
                       "res = []\nfor n in range(300):\n    res = res + [(n * 0.5, n)]\nprint(len(x), len(s), len(res))")
    assert r["output"] == "900 900 300"


def test_text_of_a_value_that_repeats_itself_is_bounded(ws, small):
    """[x] * 30 holds 30 references; as text it is x written 30 times. str() / f-strings / format / % render within
    MAX_ITEMS characters (1,000 here; this one is 81 KB), print cuts the line, an error message shows 200 characters."""
    y = "x = [[0] * 30] * 30\ny = [x] * 30\n"
    for script in ("z = str(y)", "z = f'{y}'", "z = f'{y!r}'", "z = '{}'.format(y)", "z = '{!r}'.format(y)",
                   "z = '%s' % (y,)", "z = '%r' % (y,)"):
        with pytest.raises(ScriptError, match="long as text"):      # refused before it is written, not after
            run_script(ws, y + script)
    line = out(ws, y + "print(y)")[0]
    assert line.startswith("[[[0, 0") and len(line) < 300 and "cut at 100 characters; it has 30 items" in line
    assert len(out(ws, "print('a' * 900)")[0]) < 200
    assert out(ws, "s = 'a' * 90\nprint(s, s, s, s, s)")[0].endswith("... (3 more values; cut at 100 characters)")
    for script in ("[1].index(y)", "t = (0,) * 30\nu = (t,) * 30\nv = (u,) * 30\nd = {}\nd.pop(v)",
                   "t = (0,) * 30\nu = (t,) * 30\nv = (u,) * 30\nz = {}[v]"):
        with pytest.raises(ScriptError) as e:
            run_script(ws, y + script)
        assert len(str(e.value)) < 400, len(str(e.value))


def test_format_conversions_render_within_the_bound_not_after(ws, small):
    """'{!r}'.format(y) stops writing y at the bound; it does not let repr() write all of y and then refuse."""
    calls = []

    class Leaf:
        def __repr__(self):
            calls.append(1)
            return "x"
    y = [Leaf()] * 1000                                          # 1,000 leaves, 3,000 characters as text
    it = wscript.Interp(ws)
    for conv in ("{!r}", "{!a}", "{!s}", "{}"):
        calls.clear()
        with pytest.raises(ScriptError, match="long as text"):
            it._formatter().format(conv, y)
        assert len(calls) < 400, (conv, len(calls))


def test_text_renders_as_python_does(ws):
    xs = [[1, 2.5, "a'b", None, True], (3,), (), {"k": [1, (2,)]}, set(), {4}, {}, [[]]]
    lines = out(ws, "a = [[1, 2.5, \"a'b\", None, True], (3,), (), {'k': [1, (2,)]}, set(), {4}, {}, [[]]]\n"
                    "print(a)\nprint(str(a) == f'{a}')\nd = {'x': 1}\nprint(d.keys(), d.values(), d.items())\n"
                    "b = []\nb.append(b)\nprint(b, str(b), f'{\"q\"!r}', str('q'))")
    assert lines == [str(xs), "True", "dict_keys(['x']) dict_values([1]) dict_items([('x', 1)])",
                     "[[...]] [[...]] 'q' q"]


def test_unpacking_takes_only_what_it_binds():
    """Unpacking into two names takes three items from what it unpacks (to see that it does not fit), never all of
    it: counted on a generator of 5,000, so a broken guard takes 5,000 small ints, not an unbounded range."""
    import ast
    taken = []

    def gen():
        for i in range(5000):
            taken.append(i)
            yield i
    it = object.__new__(wscript.Interp)
    it.locals = {}
    with pytest.raises(ScriptError, match="cannot unpack"):
        it._bind(ast.parse("a, b = 0").body[0].targets[0], gen())
    assert len(taken) == 3


def test_the_shipped_bounds():
    """The shipped values, stated in the reference; checked as numbers, never by building what they refuse."""
    assert (wscript.MAX_ITEMS, wscript.MAX_PY_BYTES, wscript.MAX_FORMAT_WIDTH, wscript.MAX_INT_BITS,
            wscript.MAX_PRINT_CHARS, wscript.WHOLE_READ_VALUES) == (10_000_000, 512 << 20, 10_000, 1 << 16, 20_000,
                                                                    320_000_000)
    assert "10M items" in FUNCTION_DOC and "512 MB" in FUNCTION_DOC and "20,000 characters" in FUNCTION_DOC


def test_refusals_say_what_to_do_in_one_line(ws):
    for script in ("x = filter(abs, [1])", "x = isinstance(1, int)", "x = [1].frobnicate()", "x = {}['k']",
                   "d = {}\nd['k'] += 1"):
        with pytest.raises(ScriptError) as e:
            run_script(ws, script)
        assert "\n" not in str(e.value)
    with pytest.raises(ScriptError, match="the built-ins are: range enumerate"):
        run_script(ws, "x = isinstance(1, int)")
    with pytest.raises(ScriptError, match=r"available: append clear copy count extend"):
        run_script(ws, "x = [1].frobnicate()")
    with pytest.raises(ScriptError, match=r"d.get\(k, default\)"):
        run_script(ws, "x = {}['k']")


def test_comprehension_steps_are_bounded(ws, monkeypatch):
    monkeypatch.setattr(wscript, "MAX_STEPS", 1000)
    with pytest.raises(ScriptError, match="step limit"):
        run_script(ws, "x = sum(i for i in range(5000))")          # 5,000 items if the step limit were broken


# ---------------------------------------------------------------------------------------------- quantized whole reads
@pytest.fixture()
def quant(tmp_path, monkeypatch):
    """A fake GGUF workspace whose store refuses a read above 100 values (every matrix is 384), slabs of 100."""
    monkeypatch.setattr(F, "MAX_READ", 100)
    monkeypatch.setattr(W, "STREAM_ELEMS", 100)
    monkeypatch.setattr(wscript, "WHOLE_READ_VALUES", 1000)
    ws = F.quant_workspace(tmp_path / "q")
    return ws


def test_a_whole_quantized_matrix_is_read_in_slabs(quant):
    gate = "model.layers.0.mlp.gate_proj.weight"
    quant.store.reads.clear()
    r = out(quant, "x = mlp(0).W_gate\nprint(list(x.shape), float(x.sum()))")
    want = F.llama_tensors()[gate]
    assert r[0].startswith("[16, 24]") and abs(float(r[0].split()[-1]) - float(want.sum())) < 1e-3
    assert quant.store.reads and max(n for _, n in quant.store.reads) <= 100


def test_base_and_writes_of_a_whole_quantized_view(quant):
    r = run_script(quant, "b = base(mlp(0).W_up)\nmlp(0).W_up *= 0.5\nprint(float(b.sum()))")
    up = "model.layers.0.mlp.up_proj.weight"
    want = F.llama_tensors()[up]
    assert abs(float(r["output"]) - float(want.sum())) < 1e-3
    assert up in r["deltas"] and max(n for _, n in quant.store.reads) <= 100


def test_whole_reads_past_the_bound_are_refused_with_the_slab_loop(quant):
    with pytest.raises(ScriptError) as e:
        run_script(quant, "a = mlp(0).W_gate.T\nb = mlp(0).W_up.T\nc = mlp(1).W_gate.T")
    msg = str(e.value)
    # 768 held of 1000: 232 values left, rows of 16 -> 14 rows per slab
    assert "line 3" in msg and "384 values" in msg and "already holds 768" in msg and "\n" not in msg
    loop = 'for a in range(0, 24, 14): X = P("model.layers.1.mlp.gate_proj.weight")[a:min(a + 14, 24), :]'
    assert loop in msg
    # the loop it gives runs where the whole read did not, holding what was already held
    run_script(quant, "a = mlp(0).W_gate.T\nb = mlp(0).W_up.T\n" + loop + "; print(list(X.shape))")


def test_whole_reads_that_are_let_go_no_longer_count(quant):
    run_script(quant, "a = mlp(0).W_gate.T\nb = mlp(0).W_up.T\na = 0\nb = 0\nc = mlp(1).W_gate.T\nd = mlp(1).W_up.T")
    run_script(quant, "for L in range(2):\n    a = mlp(L).W_gate * 2\n    b = mlp(L).W_up.T")
    with pytest.raises(ScriptError, match="already holds 768"):          # a view (.T) of a read still holds it
        run_script(quant, "a = mlp(0).W_gate.T\nb = [mlp(0).W_up]\nc = mlp(1).W_gate.T")


def test_whole_reads_tell_memory_admission(quant, monkeypatch):
    seen = []
    monkeypatch.setattr(wscript, "edit_preflight", lambda ws, stored=0, scratch=0, **k: seen.append(scratch))
    run_script(quant, "a = mlp(0).W_gate.T\nb = mlp(0).W_up.T")
    assert 384 * 12 in seen and 384 * 12 + 384 * 4 in seen


def test_a_float_store_keeps_its_own_bound(ws, monkeypatch):
    monkeypatch.setattr(wscript, "MAX_VIEW", 320)                        # the bound on one read (was MAX_ELEMS * 8)
    with pytest.raises(ScriptError) as e:
        run_script(ws, "x = mlp(0).W_in.T")
    assert "index it to the part you need, e.g. for a in range(" in str(e.value)
    monkeypatch.setattr(wscript, "WHOLE_READ_VALUES", 1)                 # the quantized bound does not apply
    monkeypatch.setattr(wscript, "MAX_VIEW", 480_000_000)
    run_script(ws, "x = mlp(0).W_in.T\ny = mlp(1).W_in.T")


# ---------------------------------------------------------------------------------------------- one step, bounded work
# A single step must not do unbounded work inside one C call (a scan, a sort, a copy, a hash, a comparison). Each case
# runs an operation on a value of at most a few thousand items 100 times: the steps of the loop itself (measured with
# work made free) fit the limit set here, the work charged for the operation does not. Small values, short runs: a
# broken charge makes the case pass where it should be refused, never run long.
SETUP = ("x = list(range(2000))\nx2 = list(x)\ns = 'ab' * 1000\nsl = 'ab' * 2400\nt = tuple(x)\n"
         "d = {i: i for i in range(2000)}\nw = ['ab'] * 1000\nv = list(range(500))\nv60 = list(range(60))\n"
         "pairs = list(zip(x, x))\nn1 = '1' * 1000\ntx = 'abcdefghij' * 200\ns4 = 'ab' * 200\nfl = 'a' * 4000 + '%d'\n"
         "kt = tuple(x)\ndt = {kt: 1}\n")
# Each case is a bare expression where it can be, so no other charge (freeing the old value of a name, reading an
# iterator) does the work the case is about; mutants of the interpreter were used to check this.
WORK = [
    "0.5 in x", "0.5 not in t", "'zz' in sl", "x.count(5)", "x.index(1999)", "t.count(3)",
    "x == x2", "sorted(x)", "max(x)", "min(t)", "sum(x)", "any(x)", "all(x)",
    "reversed(x)", "tuple(x)", "list(t)", "set(x)", "dict(pairs)", "','.join(w)",
    "s.upper()", "s.count('a')", "s.replace('a', 'b')", "s.split('a')", "x[:]", "s * 2",
    "x + x", "x.copy()", "int(n1)", "toks(tx)", "d.copy()", "dt[kt]",
    "fl % 1",                                  # the format's own length
    "'%s' % s4",                               # writing the argument as text (the result is under a step)
    "str(v)", "f'{v}'", "'{}'.format(v)",
    "y = v60[:]",                              # freeing the old y walks it (the slice itself is under a step)
    "sum(range(60))",                          # items read from range (the sum itself is under a step)
]


def run_free_then_charged(ws, setup, loop, monkeypatch):
    """`loop` hits the step limit with room for every step it takes but none for the work charged inside its calls.
    The setup's own charges are spent before the limit is set, so they cannot do the case's work for it."""
    monkeypatch.setattr(wscript, "ITEMS_PER_STEP", 10 ** 12)          # work free: the loop's own steps
    it = wscript.Interp(ws)
    it.run(setup)
    before = it.steps
    it.run(loop)
    free = it.steps - before
    monkeypatch.setattr(wscript, "ITEMS_PER_STEP", 64)
    it = wscript.Interp(ws)
    it.run(setup)
    monkeypatch.setattr(wscript, "MAX_STEPS", it.steps + free + 200)   # room for the loop, not for its work
    with pytest.raises(ScriptError, match="step limit"):
        it.run(loop)


@pytest.mark.parametrize("op", WORK)
def test_one_step_cannot_do_unbounded_work(ws, op, monkeypatch):
    monkeypatch.setattr(wscript, "MAX_ITEMS", 5000)
    run_free_then_charged(ws, SETUP, "for i in range(100):\n    " + op, monkeypatch)


def test_walking_a_nested_value_is_charged(ws, monkeypatch):
    """Comparing 2,000 one-item tuples touches 4,001 items (62 steps) but walks 2,001 containers in Python first."""
    monkeypatch.setattr(wscript, "MAX_ITEMS", 5000)
    run_free_then_charged(ws, "p = [(j,) for j in range(2000)]\nq = [(j,) for j in range(2000)]\n", "y = p == q",
                          monkeypatch)


def test_a_range_answers_membership_by_arithmetic(ws):
    """As Python answers: an int or an integral float by position, nothing else is ever equal to an int."""
    import math as m
    r = range(-7, 50, 3)
    for v in (-7, 2, 3, 47, 50, True, False, 2.0, 2.5, -7.0, m.nan, m.inf, "2", None, (2,), [2]):
        it = wscript.Interp(ws)
        assert it._contains(v, r) == (v in r), v
        assert it._contains(torch.tensor(float(v)), r) == (v in r) if isinstance(v, (int, float)) else True
    assert out(ws, "print(0.5 in range(10), 3.0 in range(10), 'a' in range(10), 5 not in range(4))") == \
        ["False True False True"]


def test_a_range_is_not_scanned_for_a_float(ws):
    """0.5 in range(N) costs the same for any N; Python's own test scans the range (about 3 s for 10 ** 8)."""
    import time
    it = wscript.Interp(ws)
    t0 = time.perf_counter()
    assert it._contains(0.5, range(10 ** 8)) is False and it._contains(10 ** 8 - 1.0, range(10 ** 8)) is True
    assert it._contains("a", range(10 ** 8)) is False and it._contains((1,), range(10 ** 8)) is False
    assert time.perf_counter() - t0 < 0.5 and it.steps == 0


KEY_REFUSED = ["sorted([1, 2], key=logits)", "sorted(['a'], key=act)", "max([1, 2], key=revert)",
               "min([1, 2], key=print)", "x = [1, 2]\nx.sort(key=P)", "sorted([1], key=scale_neuron)",
               "sorted([1], key=E)", "sorted([1], key=shift_tokens)"]


@pytest.mark.parametrize("script", KEY_REFUSED)
def test_key_takes_only_value_functions(ws, script):
    with pytest.raises(ScriptError, match="key= takes a value function"):
        run_script(ws, script)


def test_value_keys_and_dict_get_work(ws):
    assert out(ws, "d = {'a': 3, 'b': 1, 'c': 2}\nprint(sorted(d, key=d.get), max(d, key=d.get))\n"
                   "print(sorted(['bb', 'a', 'ccc'], key=len), sorted([(1, 2), (1,)], key=sum), min([-3, 2], key=abs))\n"
                   "x = ['10', '9']\nx.sort(key=int)\nprint(x, sorted([2.7, 1.2], key=round))") == \
        ["['b', 'c', 'a'] a", "['a', 'bb', 'ccc'] [(1,), (1, 2)] 2", "['9', '10'] [1.2, 2.7]"]


def test_each_key_call_is_a_step(ws, monkeypatch):
    it = wscript.Interp(ws)
    it.run("x = list(range(300))")
    before = it.steps
    it.run("y = sorted(x, key=abs)")
    assert it.steps - before >= 300                                     # one step per key call, besides the sort
    monkeypatch.setattr(wscript, "MAX_STEPS", it.steps + 100)
    with pytest.raises(ScriptError, match="step limit"):
        wscript.Interp(ws).run("x = list(range(300))\ny = sorted(x, key=abs)\nz = sorted(x, key=abs)")


# A tuple that holds another twice, sixteen times over: 17 tuples in memory, 196,607 items when compared or hashed.
NEST = "t = (0,)\nu = (0,)\nfor i in range(16):\n    t = (t, t)\n    u = (u, u)\n"
NESTED = ["y = t == u", "y = t != u", "y = t < u", "y = {t: 1}", "y = {t}", "y = set([t])", "y = dict([(t, 1)])",
          "d = {}\nd[t] = 1", "d = {}\ny = d.get(t)", "d = {}\nd.setdefault(t, 1)", "y = t in {}", "y = t in [u]",
          "y = [t].count(u)", "y = [u, t].index(t)", "y = sorted([t, u])", "y = max([t, u])", "y = {k: 1 for k in [t]}",
          "y = {k for k in [t]}", "y = [t] == [u]", "y = (t, 1) in {(u, 1): 2}"]


@pytest.mark.parametrize("op", NESTED)
def test_values_repeated_inside_themselves_are_refused_before_they_are_compared_or_hashed(ws, op, monkeypatch):
    monkeypatch.setattr(wscript, "MAX_EXPANDED", 50_000)               # 196,607 items here: refused at 4 x the bound
    with pytest.raises(ScriptError, match="touches more than 50,000 items"):
        run_script(ws, NEST + op)


def test_the_expanded_size_walk_is_memoised():
    t = (0,)
    for _ in range(16):
        t = (t, t)
    n, walked = wscript._expanded(t, 10 ** 9)
    assert n == 3 * 2 ** 16 - 1 and walked < 100                     # 17 distinct tuples, not 196,607 items
    assert wscript._expanded([1, 2.5, None, True], 100) == (5, 0)     # a flat container is counted in C
    assert wscript._expanded(["ab" * 40, "c"], 100) == (3 + 10, 0)    # strings: one item per 8 characters
    with pytest.raises(wscript._TooBig):
        wscript._expanded([[0] * 10] * 10, 50)
