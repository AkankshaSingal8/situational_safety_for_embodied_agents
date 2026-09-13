# Public-release restructure plan

Safety net already in place — everything below is reversible:

| Ref | Points at | Purpose |
|---|---|---|
| `paper/v1-pre-restructure` (tag) | `9f720f33` | Immutable snapshot of main before the restructure |
| `archive/pre-release-2026-09` (branch) | `9f720f33` | Full pre-restructure tree; nothing is ever "moved to" it |

Nothing is deleted from history. Archived content stays reachable on the archive
branch at zero cost; pruning `main` does **not** shrink a clone. A genuinely
smaller repo would need `git filter-repo`, which is mutually exclusive with
preserving history — history preservation is the better trade here since the
result dirs are small (mostly one JSON each).

---

## Release blockers — fix before any public push

These are independent of layout and each breaks a reader on their first command.

1. **`vlsa-aegis` submodule points at a commit that does not exist publicly.**
   The working tree is at `e348187`, which `git branch -a --contains` shows on
   local `main` only — no `origin/*`. The committed pointer is `9838b4fa`.
   `git clone --recurse-submodules` (the README's first instruction) would fail
   against `e348187`.
   **Action:** leave the committed pointer at the public `9838b4fa`, do NOT commit
   the dirty `+`, and make `patches/vlsa-aegis.patch` a mandatory README step.
   `run_libsafety_eval_aegis_gt.py:67` imports `compute_h_coeffs_3d` from that
   modified submodule, so the patch is load-bearing, not optional.

2. **`SafeLIBERO` and `openvla-oft` show `-` in `git submodule status`** (not
   registered) despite populated directories. Verify `.git/modules` bookkeeping
   before claiming the clone path works.

3. **`results_tables/fol_v17_final_results.md` and `fol_v16_spatial_results.md`
   cite directories that exist nowhere on `main`** (`fol_v15/v16/v17/v172_*_n50`).
   They are on the `fol-safety-filter` branch only. These are the strongest
   tables in the repo (v17: 43.0/55.5 L1, 42.0/47.0 L2), and
   `fol_final_campaign_results.md` uses the v17.2 oracle row as its
   perception-error upper bound.
   **Action:** cherry-pick those six paths from `fol-safety-filter`, or annotate
   both tables as not reproducible from `main`. Do **not** merge that branch —
   it carries a 16 MB copyrighted PDF and `vlm_pipeline/semantic-safety-filter.zip`
   that `main` does not have.

---

## Highest-severity reorg risk: a silent wrong-answer path

`fol_safety_filter/cbf_mapper.py` imports `SuperquadricParams` via
`from vlm_pipeline.semantic_cbf_filter import ...`, which resolves only because
`vlm_pipeline/` has **no `__init__.py`** (PEP-420 namespace package). It wraps
that import in `except ImportError` with a **fallback stub dataclass**.

Renaming `vlm_pipeline/` therefore does not raise — it silently substitutes stub
superquadric parameters and produces wrong CBF geometry. Any rename of that
directory MUST delete the stub fallback in the same change so a breakage is loud.

Meanwhile `fol_safety_filter/filter.py:34-36` imports the *same file* the other
way (flat `from semantic_cbf_filter import ...` after inserting `../vlm_pipeline`),
and `vlm_pipeline/vlm_grounding_accuracy.py` imports back into
`fol_safety_filter` — a bidirectional cycle across ~50 `sys.path.insert` calls
repo-wide.

**Recommendation:** add a `pyproject.toml` and `pip install -e .` so those
`sys.path` lines become *deletable* rather than *relocatable*. Without it, a
`src/` layout is pure cost. Note the Python-version split: the SafeLIBERO stack
is 3.8, LIBERO-Safety requires >=3.10.

---

## Target layout

Flat packages at root (not `src/`): five git submodules already occupy the root
and must stay there, and `docs/experimental_setup_appendix.md` is built on
`file:line` citations that every path change invalidates.

```
fol_safety_filter/        the method (name unchanged — it is the paper's method)
safelibero_eval/          runnable half of vlm_pipeline/ (drivers + env/CBF lib)
libsafety_eval/           the six root-level run_libsafety_*.py + flow_cbf
vlm_prompt_runner/        unchanged; only results/ moves out
prompts/                  unchanged path — ~15 CWD-relative refs, half in docstrings
scripts/                  aggregators and one-shot CLIs, nothing importable
setup/                    the 11 loose root *_setup.sh, grouped by benchmark
slurm/{safelibero,libsafety}/   pruned to reported conditions only
data/                     prompt_tuning_benchmark_set (an INPUT, not a result)
results/                  see below
docs/                     curated; superpowers/plans -> archive
patches/                  unchanged
<5 submodules>            paths unchanged
```

`results/` is split by benchmark deliberately: the two benchmarks define safety
oppositely (appendix §: "SafeLIBERO TSR and LIBERO-Safety SR are not
commensurable and must never share a column"). Splitting at the filesystem level
makes it structurally harder to put them in one table.

```
results/
  README.md                    NEW: dir -> table/figure -> SLURM job id
  safelibero/
    policy_baselines/{openvla_oft,pi05,cosmos,fastwam}/
    fol/{baseline_og10,v18,v19,v19f,v20,v21d2}/{L1,L2}/
  libsafety/{cosmos,fastwam}/
  vlm_prompts/                 p3_candidate_list + vlm_prompt_runner/results
  grounding_accuracy/
  tables/                      <- results_tables/
```

---

## Classification summary

**KEEP on main (33 result dirs).** Criterion: the numbers appear in a tracked
table on `main`. Being globbed by `final_results.py` does NOT count — that script
reads only the superseded n=40 generation.

**ARCHIVE (22 dirs):** the 14 n=40 FOL-generation dirs, `fol_v19e_*`,
`fol_v22_*`, `fol_v23b/c_*`, `fastwam_benchmark_libsafety` (n=1 smoke),
the stale `results/` M1/M2/M3 demo JSONs, `vlm_outputs_with_different_prompt`.

**`slurm/`:** 184 -> ~35. Of the 184, **43 are smoke/debug runs** backing no
number, ~55 are superseded FOL versions, and 9 (`v22`, `v23a/b/c`) are cited by
no table. All 184 hardcode cluster paths; 156 reference `.worktrees/` paths
absent from a clone. Kept scripts need `REPO="${REPO:-$(git rev-parse --show-toplevel)}"`
and the worktree indirection removed.

**Docs:** ~4,000 lines across root `CLAUDE.md`, `vlm_pipeline/CLAUDE.md`,
`progress.md`, `vlm_pipeline_readme.md`, `CBF_construction.md` and
`integrated_eval_readme.md` describe the **retired M1/M2/M3 paradigm** as if it
were current; every reported number came from the FOL filter.
`vlm_pipeline/CLAUDE.md` references `benchmark_prompts.py`, which does not exist.
Four files are prompts rather than documentation
(`Hierarchical_symbolic_safety.md`, `vlm_pipeline/evaluation_openvla_oft.md`,
`prompt_writing_plan.md`, `assumptions.md`).

---

## Needs an owner decision

| Item | Why it cannot be decided from the code |
|---|---|
| `semantic_cbf/` | Self-consistent 2D prototype with mocked VLM calls; appears in no results table. Archive unless the paper has a latent-CBF section. |
| `epistemic_uncertainty/` | Clean, 23 files, real tests — but in no results table or memory entry. |
| `fol_v22`, `fol_v23b`, `fol_v23c` | Dated 2026-07-12, one day AFTER the campaign table, coinciding with the ICRA-pivot note. Cited by nothing. |
| Which paper `main` accompanies | The submitted ReS AI paper draws on `results_tables/fgd_tier_gt_results.md` and `ls_rewrite_fix/runs/` in the `flow-guidance-tier-gt` worktree — **none of it is on `main`**. So "final results under results/" covers the SafeLIBERO/LIBERO-Safety boards only, not the submitted paper's tables. |

---

## Execution order

Never combine a move with an edit: git has no rename records, and `--follow`
relies on content-similarity detection that degrades when content changes in the
same commit.

```
1..5   pure `git mv` only, grouped (results/, setup/, libsafety_eval/,
       safelibero_eval/, docs/)
6      imports, sys.path removals, argparse defaults, pyproject.toml
7      deletions (byte-identical duplicates + archived dirs)
8      README + results/README.md rewrite
9      appendix path sweep
```

Verify after each move commit:
```bash
git log --follow --oneline -- <new/path>      # reaches the original
git show --stat -M90% HEAD | grep -c '=>'      # renames detected
```

**Post-move sweep that bites later:** `docs/experimental_setup_appendix.md` is
built on `file:line` citations. Line numbers survive a pure move; **paths do
not**. Every rename invalidates at least one appendix citation. Schedule this
explicitly — the cost lands in the paper, not in CI.

**Silent-failure traps to fix during the move**, not after:
- `final_results.py` globs 12 hardcoded sibling dirs and fails **silently**
  (`load_latest` returns None, rows print `N/A`). All 12 are on the archive list,
  so rewrite it against the v18-v21 campaign rather than moving it.
- `final_results.py`, `compare_results.py`, `scripts/aggregate_results.py` all
  select runs via `sorted(glob)[-1]` — **renaming or moving files inside a kept
  directory can silently change which run a table reports.**
- `aggregate_table3_replication.py:25` hardcodes `RESULTS_DIR = "LIBERO-Safety/results"`,
  writing *inside a submodule*.
- `epistemic_uncertainty/run_safelibero_uncertainty_eval.py:345` defaults to
  `results/uncertainty_eval`, which **collides with the new results/ tree**.
- Moving `run_libsafety_*.py` into `libsafety_eval/` changes `Path(__file__).parent`
  depth by one for three drivers — centralise in `libsafety_eval/_paths.py`.
