"""pytest session setup for `vlm_pipeline/tests/` (fix round 1, Task 3,
consequence-steering-code plan, spec 2026-08-01).

On a many-core node (observed: 256 cores, HPC login/compute nodes with
`OMP_NUM_THREADS` unset), torch's default CPU intra-op thread pool launches
one OMP thread per core for EVERY op -- for the many tiny (~3-4e4 param)
`ConsequenceModel` forward passes this suite runs, thread-launch overhead
alone stalled the full suite past 500s (observed hang) where the same tests
run in ~5s with threading pinned to 1. This is exactly the fix already
applied at the call site in `consequence_select.run_ensemble_select`
(`torch.set_num_threads(1)`); this conftest is the belt-and-braces version
for the whole suite (covers `consequence_model.py`'s own training/eval
tests too, which call torch directly and don't go through
`consequence_select`), applied before any test collects/runs so os.environ
and torch's thread pool are set process-wide from the start.

Guarded: torch is optional in this test env (see `consequence_model.py`'s
module docstring); if unimportable, this is a no-op and the rest of the
suite's torch-skip guards (`pytest.importorskip("torch")`) still apply.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass
