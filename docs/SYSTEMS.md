# Bounded Systems analysis

Systems scores describe linear writer/reader composition, ignoring intervening
normalization and nonlinearities. They are not causal or behavioral validation.
Hybrid recurrent attention is explicitly omitted from this graph; ordinary full
attention heads and mapped MLPs remain represented. A recurrent layer is not
invented as a set of conventional attention heads.

Previously `/api/wiring` called `ml["W_out"].read()` and each MLP reader's `.read()`
before projection, and `sem.stack()` materialized whole attention projections.
A5120×17408 MLP contains89,128,960 logical values, exceeding the GGUF16M read limit.
This is an analysis-allocation bug, not evidence of a corrupt checkpoint.

The reductions now stream canonical `Ref` slices (including active deltas and
GGUF inverse permutations). Writer Gram and reader Gram/norm sums are accumulated
from at most262,144 float values per read (1MiB); head products are reduced locally.
The base tensor cache is bypassed. A dynamic projection rank bounds retained Gram,
projection, stack and temporary buffers to a conservative128MiB estimate. Scores
retain seed0 Gaussian projection and bias correction; reduction order can produce
small floating-point differences. The top4000 edges are retained, with the total
pair count reported separately.

Large models default to a narrow layer window. Above400M parameters the UI requires
**Analyze window**, with explicit first-layer/count controls (maximum3), progress
and Cancel. A metadata preflight rejects work exceeding512M logical values before
reading weights. A120-second deadline and cancellation are checked between slabs;
explicit Cancel and IDE shutdown request cancellation. Matching requests share
the same bounded work, and navigation/disconnect do not cancel it on behalf of
other observers. Cancel stops that shared calculation; retry is explicit.
Workspace/version/group/suite scope checks and bounded pending/completed caches
from PR8 remain enforced. The API also enforces the slab/work/time bounds. Cache belongs to the workspace and invalidates on
committed edits, enable/disable, scaling and undo. Whole-model scans are not an
implicit side effect of entering the large-model Systems tab.

Validation uses tiny dense/reference multiplication, seed-matched sketches,
permuted GGUF+patch reductions, lazy89M-element shape rejection/cancellation and
API/cache tests. This fix does not validate a27B full tensor scan or inference.

Small real browser evidence: Qwen2.5-0.5B-named Q4_0 GGUF actually contains
630,167,424 tensor parameters. A single-layer window read17,088,512 values in slabs
of at most261,632 values, with a133,988,352-byte retained-work estimate. The initial
browser reload retained an old module; after explicit cache refresh, the new
Analyze/Cancel controls and graph rendered correctly. No27B payload scan was run.
