# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

Semantic safety filtering for vision-language-action (VLA) robot policies. A VLM
proposes open-vocabulary safety predicates from robot observations; those are
grounded into parametric spatial constraints, assembled into control barrier
functions (CBFs), and used to filter the policy's nominal action each timestep.

**The live method is `fol_safety_filter/`** — first-order-logic predicates →
knowledge base → CBF mapper → runtime filter. It produced every reported number
and is invoked by `vlm_pipeline/run_safelibero_fol_openvla_eval.py`.

> Historical note: earlier revisions of this file described an "M1 (Seg+VLM) /
> M2 (VLM-only) / M3 (3D+VLM)" architecture. That paradigm was retired. Its
> prototypes survive in `semantic_cbf/` (2D simulation, mocked VLM calls,
> no eval path) and parts of `vlm_pipeline/`. Do not treat them as current.

## Two benchmarks, opposite safety semantics

This is the single most important fact when touching metrics:

| | SafeLIBERO (arXiv:2512.11891) | LIBERO-Safety (arXiv:2606.23686) |
|---|---|---|
| On violation | episode continues | **episode terminates, scored a failure** |
| Primary metric | TSR (completion, collisions ignored) | SR (completion **with no** violation) |
| Collision signal | >1 mm hazard displacement | BDDL `:constraints` predicates |

**SafeLIBERO TSR and LIBERO-Safety SR are not commensurable and must never share
a column.** See `docs/experimental_setup_appendix.md` for the full protocol with
every value traced to `file:line`.

## Layout

| Path | Contents |
|---|---|
| `fol_safety_filter/` | The method: predicates, KB, rule composer, VLM grounding, CBF mapper, runtime filter |
| `vlm_pipeline/` | SafeLIBERO eval drivers + env helpers. Mixed: also holds retired CBF prototypes |
| `run_libsafety_*.py`, `libsafety_*.py`, `flow_cbf_*.py` | LIBERO-Safety eval drivers and the flow-CBF barrier |
| `vlm_prompt_runner/` | Multi-model VLM prompt harness (Anthropic/OpenAI/Gemini/Qwen) with majority voting |
| `semantic_cbf/` | Retired 2D prototype. Self-contained, on no eval path, kept for reference |
| `epistemic_uncertainty/` | Self-rooted: own SLURM entry points and tests. Not part of the headline method |
| `results/` | All results, split `safelibero/` vs `libsafety/` so the two cannot be conflated |
| `slurm/{safelibero,libsafety}/` | Job scripts for the reported conditions only |
| `setup/` | Environment bootstrap, grouped by benchmark |

> **Do not delete `vlm_pipeline/semantic_cbf_filter.py`.** It reads as a retired
> prototype but is a runtime dependency of the live method: `fol_safety_filter/filter.py:36`
> imports it, and `:368` catches the resulting `ImportError` with a
> `"Geometry-only mode"` warning. Removing it does not raise -- it silently
> downgrades the filter to geometry-only and changes every reported number.
> `vlm_pipeline/save_vlm_inputs.py` is similar: no importer, but it is the only
> way to regenerate the gitignored `vlm_inputs/` tree the accuracy studies read.

## Known issues — read before trusting a metric

`docs/code_review_findings.md` records 18 defects, several unfixed on this branch.
The ones that change numbers:

- **`CheckRobotContact` can never fire** (appends geom *indices*, tested by
  *name*). The arm-hits-hazard channel yields zero detections on all four
  LIBERO-Safety suites. `affordance` has no `:constraints` block at all, so its
  CAR ≡ 1.000 vacuously. Use `libsafety_contact.py` for a real robot-contact
  metric; `patches/LIBERO-Safety.patch` fixes the predicate but is not applied.
- Several `fol_*_n50` numbers were produced by an uncommitted working tree and
  are not reproducible from any commit. See `docs/restructure_plan.md`.

## Conventions

- `MUJOCO_GL=egl` for all headless rendering.
- The simulator seed moves **object positions** even with a fixed initial state,
  so every method in a comparison table must share `env_seed`. It is deliberately
  separate from the torch/numpy `seed`.
- Tests: `python -m pytest fol_safety_filter/tests vlm_prompt_runner/tests`.
  `epistemic_uncertainty/tests` additionally needs `torch`.
