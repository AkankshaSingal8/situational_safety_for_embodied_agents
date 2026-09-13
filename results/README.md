# Results

Split by benchmark, deliberately. SafeLIBERO and LIBERO-Safety define safety
**oppositely** — SafeLIBERO lets an episode continue after a collision and scores
TSR on completion alone; LIBERO-Safety terminates on any violation and scores SR
as completion *without* violation. The two are **not commensurable and must never
share a column**. Keeping them in separate trees makes that hard to get wrong.

Full protocol, with every value traced to `file:line`:
[`../docs/experimental_setup_appendix.md`](../docs/experimental_setup_appendix.md).

## Layout

```
safelibero/
  policy_baselines/{openvla_oft,pi05,cosmos,fastwam,fastwam_50ep}/
  fol/
    baseline_og10/{L1,L2}      no-filter reference for the campaign table
    v1_clean/{suite}_L{1,2}    the 8-condition v1 sweep (4 suites x 2 levels)
    v15,v16,v17,v172/{L1,L2}   development series; v172 is the GT oracle
    v18,v19,v19f,v20,v21d2/{L1,L2}   the campaign table rows
libsafety/{cosmos,fastwam}/
grounding_accuracy/            VLM detection/grounding accuracy studies
vlm_prompts/                   obstacle-ID prompt-study outputs
tables/                        aggregated markdown tables
```

## Which directory backs which table

| Table in `tables/` | Backed by |
|---|---|
| `summary_table.md` (OVL / π0.5 boards) | `safelibero/policy_baselines/{openvla_oft,pi05}/` |
| `fol_final_campaign_results.md` | `safelibero/fol/{baseline_og10,v18,v19,v19f,v20,v21d2}/` + the v172 oracle row |
| `fol_v17_final_results.md`, `fol_v16_spatial_results.md` | `safelibero/fol/{v16,v17,v172}/` |
| `all_per_task_tables.md` | `safelibero/policy_baselines/` |

## Caveats that apply to the numbers here

1. **LIBERO-Safety collision columns are partly unmeasurable.** The benchmark's
   `CheckRobotContact` predicate can never fire, so the arm-hits-hazard channel
   yields zero detections on all four suites. `affordance` declares no
   `:constraints` at all, so its CAR ≡ 1.000 vacuously, and `human_safety`'s is
   1.000 because the only predicate present cannot fire. The two obstacle suites
   retain a live `CheckContact` channel that measures **object**-hazard contact,
   never arm-hazard. Details: [`../docs/code_review_findings.md`](../docs/code_review_findings.md).

2. **LIBERO-Safety runs here are 3 trials × 1 seed**; the paper specifies
   10 trials × 3 seeds (45 vs 450 episodes per suite). Not directly comparable
   to published numbers.

3. **The `fol/v1_clean` and campaign numbers were produced by an uncommitted
   working tree** and are not reproducible from any commit in this repo. They
   need re-running under a tagged commit before publication. See
   [`../docs/restructure_plan.md`](../docs/restructure_plan.md).

4. `safelibero/policy_baselines/fastwam/` is n=200 for seven conditions but n=40
   for spatial/level-I — do not present it as a uniform board.

5. Aggregation scripts select runs with `sorted(glob)[-1]`, so **renaming or
   adding a file inside one of these directories can silently change which run a
   table reports.**
