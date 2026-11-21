"""scripts/formats_shared.py rounded_sums in row chunks: the same numbers as the whole-tensor arithmetic it replaced.

Found on Qwen3.5-4B: on a [2560, 9216] matrix the whole-tensor version held about 3 GiB of temporaries, and with the
model on a GPU the comparison ended at the memory guard's budget.
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "formats_shared.py"
_SPEC = importlib.util.spec_from_file_location("formats_shared_rounded_under_test", _PATH)
FS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FS)


def whole(W, d1, d2):
    """rounded_sums as it was first written, every temporary the size of the tensor."""
    W = W.to(torch.float32)
    a = (W + d1.to(torch.float32)).to(torch.bfloat16)
    b = (W + d2.to(torch.float32)).to(torch.bfloat16)
    s = FS.bf16_steps(a, b)
    out = {"elements": int(s.numel()), "equal": int((s == 0).sum()), "one_step": int((s == 1).sum()),
           "more_than_one": int((s > 1).sum()), "max_steps": int(s.max()) if s.numel() else 0}
    af, bf = a.float(), b.float()
    babs = bf.abs()
    diff = (af - bf).abs()
    m = float(babs.median()) if babs.numel() else 0.0
    if m == 0.0:
        nz = babs[babs > 0]
        m = float(nz.min()) if nz.numel() else 0.0
    s_m = FS.step_at(m) if m > 0 else None
    mx = float(diff.max()) if diff.numel() else 0.0
    far = s > 1
    largest = None
    if bool(far.any()):
        largest = float(babs[far].max()) / m if m > 0 else math.inf
    differs = s != 0
    allowed = torch.maximum(FS.spacing(af), FS.spacing(bf))
    if s_m is not None:
        allowed = torch.clamp(allowed, min=s_m)
    if bool(differs.any()):
        excess = diff[differs] / allowed[differs]
        outside, max_excess = int((excess > 1).sum()), float(excess.max())
    else:
        outside, max_excess = 0, 0.0
    out.update({"median_abs": m, "step_at_median": s_m, "max_abs_difference": mx,
                "max_abs_difference_in_median_steps": (mx / s_m) if s_m else (0.0 if mx == 0 else math.inf),
                "differing": int(differs.sum()), "largest_abs_over_median": largest,
                "outside": outside, "max_excess": max_excess})
    return out


def cases():
    g = torch.Generator().manual_seed(7)
    W = (0.05 * torch.randn(37, 53, generator=g)).to(torch.bfloat16)
    small = 1e-4 * torch.randn(37, 53, generator=g)
    yield "two close deltas", W, small, small + 3e-6 * torch.randn(37, 53, generator=g)
    yield "the same delta", W, small, small.clone()
    big = small.clone()
    big[5, 7] += 0.5                                   # far outside the rule, above the median
    big[30, 2] -= 0.25
    yield "elements far apart", W, small, big
    near = W.clone()
    near[::2] = 0                                      # more than half the elements are zero: the median is 0
    yield "a median of zero", near, small * (near != 0), (small + 1e-5) * (near != 0)
    Z = torch.zeros(9, 11, dtype=torch.bfloat16)
    yield "all zero, equal", Z, torch.zeros(9, 11), torch.zeros(9, 11)
    yield "all zero in b only", Z, 1e-3 * torch.ones(9, 11), torch.zeros(9, 11)
    tiny = W.clone()
    tiny[3] = 1e-9                                     # near zero: an own step is far below the difference
    yield "elements near zero", tiny, small, small + 2e-6
    yield "one dimension", W[0], small[0], small[0] + 1e-5
    yield "three dimensions", W.reshape(37, 53, 1), small.reshape(37, 53, 1), (small + 4e-6).reshape(37, 53, 1)
    yield "no elements", torch.zeros(0, 4, dtype=torch.bfloat16), torch.zeros(0, 4), torch.zeros(0, 4)
    yield "a scalar", torch.tensor(0.5, dtype=torch.bfloat16), torch.tensor(1e-3), torch.tensor(1.1e-3)


@pytest.mark.parametrize("name,W,d1,d2", list(cases()), ids=[c[0] for c in cases()])
def test_row_chunks_give_the_numbers_of_the_whole_tensor(name, W, d1, d2):
    want = whole(W.reshape(1) if W.dim() == 0 else W, d1.reshape(1) if d1.dim() == 0 else d1,
                 d2.reshape(1) if d2.dim() == 0 else d2)
    for chunk in (1, 53, 54, 53 * 7 + 1, 10 ** 9, None):
        assert FS.rounded_sums(W, d1, d2, chunk=chunk) == want, (name, chunk)


def test_the_cases_cover_what_the_rule_distinguishes():
    got = {name: FS.rounded_sums(W, d1, d2, chunk=64) for name, W, d1, d2 in cases()}
    assert got["two close deltas"]["differing"] > 0 and got["two close deltas"]["outside"] == 0
    assert got["the same delta"]["differing"] == 0 and got["the same delta"]["max_excess"] == 0.0
    far = got["elements far apart"]
    assert far["outside"] == 2 and far["more_than_one"] >= 2 and far["largest_abs_over_median"] > 1
    assert got["a median of zero"]["median_abs"] > 0 and got["a median of zero"]["step_at_median"] is not None
    assert got["all zero, equal"]["step_at_median"] is None and FS.one_step_ok(got["all zero, equal"])
    assert got["all zero in b only"]["step_at_median"] is None and not FS.one_step_ok(got["all zero in b only"])
    assert got["no elements"]["elements"] == 0 and got["a scalar"]["elements"] == 1


def test_the_default_chunk_bounds_a_large_matrix():
    """A [2560, 9216] matrix is cut into 12 row chunks of at most 2**21 elements."""
    assert FS.ROUNDED_CHUNK == 1 << 21
    per = max(1, FS.ROUNDED_CHUNK // 9216)
    assert per == 227 and math.ceil(2560 / per) == 12 and per * 9216 <= FS.ROUNDED_CHUNK
