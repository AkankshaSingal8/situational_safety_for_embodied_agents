# Where the recipe can fail: three independent critiques (2026-10-20)

Three agents reviewed the text-conditioned residual flow recipe separately. One looked at the
method as a learning algorithm, one at what changes when moving to OopsieVerse, and one read the
dummy-env experiments the way a CoRL/ICRA reviewer would. Their reports are summarised below,
merged where they overlap. The critics checked their claims against `data_v2/`, `runs_v2/` and the
OopsieVerse code. I re-checked two myself: the exact decomposition in blocker 1 (all 4,100 scenes)
and the clearance numbers in item 7. A claim marked *predicted* has not been run.

## Bottom line

The dummy env shows that a text embedding can switch a learned correction on a fixed scripted
base. It does not yet show the three things the recipe needs before OopsieVerse:

1. **Composition** is untested. The A+B reference correction equals the A correction plus the B
   correction exactly, so stacking or adding can only succeed here.
2. **Language understanding** is weaker than the numbers suggest. A nearest-neighbour lookup over
   the five training wordings labels every held-out wording correctly, while the learned text
   pathway reaches about 63% on the laptop constraint.
3. **The setting** is not the OopsieVerse setting. OopsieVerse uses a stochastic, chunked,
   closed-loop VLA. Its RoboCasa track scores only mechanical damage (contact and impact force,
   mostly through the gripper), and it has no analytic safe trajectory to train on.

None of these is a reason to drop the recipe. Each has a cheap dummy-env test, listed at the end.

## Blockers (at least two critics agree)

**1. Composition holds by construction.** `safe_AB` is `safe_A` with the tilt zeroed
(`src/traj.py`). On all 4,100 scenes, `dAB − (dA + dB)` is exactly zero (re-checked), and the B
correction is exactly minus the base tilt. A moves only x and y,
and B changes only the tilt. So "clauses in turn" and "corrections added" must work whenever
each clause works, and their 0.96 against 0.98 says nothing about composition. The far-wording
drop (about 0.53) is entirely laptop violations (viol B 0.003), a paraphrase failure carried
forward. The direct sentence scores 0 of 2,400 per split, and the clause split is a hand-written
oracle. *Needed:* two constraints that act on the same action dimensions, so the right joint
correction is not the sum. An example is a second obstacle on the laptop's cheaper detour side,
or a conflicting pair such as "don't crush" against "don't drop" on the gripper.

**2. The text pathway is a soft three-way switch.**
- BGE puts all 32 held-out wordings nearer their own class's training wordings, so a
  nearest-neighbour lookup feeding the existing model should score near the training-wording
  level of about 98% (*predicted*). The learned map gets 62–65% on laptop wordings.
- Success tracks the cosine margin between the two classes (Spearman 0.74 over the 16 laptop
  wordings; n = 16, indicative only).
- Ambiguous wordings get blended corrections: a detour 20–80% of the needed size, plus some tilt
  removed.
- "Never transport the beverage above the device" is routed to the upright correction.
- The upright constraint is not a language test at all. Its target is exactly minus the base
  tilt, which the model receives as input.
- Text enters through global FiLM with no object binding. "Avoid X" works only because the
  laptop pose is hard-wired into the 10 scene features.

*Needed:* a nearest-neighbour baseline and a one-hot class baseline. If either matches, report
the result as classification.

**3. The correction is planned once, open loop, against a base that never reacts.** The 40
waypoints are corrected in one shot, and the base is a pure function of the scene. GR00T in the
local cache predicts 16-step chunks and executes 8, re-planning from what it observes, for about
90 chunks per 720-step RoboCasa episode. After a detour, the next base chunk heads straight for
the target and undoes it, or re-crosses the obstacle. The residual was trained only on
uncorrected base rollouts, so errors compound (a DAgger-shaped problem). The obstacle can also
sit beyond the chunk horizon, so the correction has to start before the base shows a violation.

**4. There is no oracle at OopsieVerse scale.** DamageSim thresholds MuJoCo contact forces and
finite-differenced accelerations. It is not differentiable, but it is queryable every step
(`info["damage_info"]`), and MuJoCo state can be saved and restored. The workable oracle is
best-of-K:
- fix GR00T's noise so A_base is the chunk actually sampled;
- sample K sibling chunks or perturb grip and speed;
- roll each forward past the 8 executed steps, since damage often lands later;
- take the safest one.

Two caveats. ΔA* then mixes base sampling noise into the safety correction, and for some states
no safe sibling exists. Once you can run this search online, the recipe has to beat simply
running that search as a filter.

**5. The constraints don't exist in OopsieVerse's RoboCasa track.** The track has only
mechanical damage (`damagesim/robosuite/evaluators/mechanical.py`). Thermal and fluid damage
exist only in OmniGibson, which needs Isaac Sim and is out of reach on this cluster. "Keep
upright" and "avoid the laptop" have no damage signal there. Crush comes from the gripper,
which the dummy env doesn't model. Text swaps reduce to choosing which object to be gentle
with, on multi-object tasks (prepare_breakfast, dishes_to_sink, shelve_item). GR00T is itself
language-conditioned, so appending the safety clause to its prompt is a mandatory baseline.

## Major

6. **It is never trained on scenes that are already safe.** The scene filter keeps only bases
   that clearly break both constraints, so safety text always comes with a nonzero correction.
   Most VLA chunks are already safe, so the model will probably intervene on every chunk
   (*predicted*). This cost
   can't show up here, because success is 0.96–1.00 for every condition: the grasp is a weld and
   the arm passes through objects.
7. **Corrections sit on the boundary.** The oracle searches for the smallest detour that keeps
   3 cm of clearance. Executed on test scenes, the model's A corrections have a median clearance
   of 2.85 cm, a 5th percentile of 1.3 cm and a 1st percentile of −1.7 cm (1.5% of runs below
   zero); the reference correction's median is 2.9 cm. Any shrinkage from blending, perception
   noise or drift becomes a violation.
8. **The model sees privileged state.** It gets exact laptop pose and size. OopsieVerse gives
   cameras and proprioception, and crush also needs grasp state (aperture and contact force).
9. **The metrics are geometric labels, not damage.** "Spill" means tilt over 15°, ignoring
   acceleration and fill. "Over the laptop" means the mug centre is over the footprint at any
   height. The binary metric hides lost clearance.
10. **The statistics overstate certainty.**
    - The unit for language generalization is the wording. Clustered by wording, pooled laptop
      paraphrase success is 0.65 ± 0.29 and 0.62 ± 0.39, against reported Wilson intervals about
      0.03 wide.
    - Training-seed variance is now measured and small (seeds 0, 7 and 42,
      `runs_v2/seeds/seeds.md`), so it is not what limits the claims. On test scenes:
      - laptop paraphrases: 0.67 ± 0.04 and 0.63 ± 0.02;
      - far-set clauses in turn: 0.56 ± 0.04;
      - far-set combined training: 0.24 ± 0.01;
      - direct combined sentence: 0.000 on every seed.

      Every paired comparison points the same way on all three seeds.
    - Two held-out sets were written after seeing results.
11. **Language robustness is untested.** Negation ("you may tilt it"), irrelevant text ("be
    quick"), instructions about objects that aren't present, and conflicting instructions are not
    tested. The model has never seen non-null text with no correction, and BGE barely separates
    negations.
12. **The compute doesn't fit.** Best-of-K oracle data needs about 500 episodes at roughly
    1.5–3 min each plus sibling rollouts, about 15–30 GPU-hours. Paired evaluation needs n ≥ 30
    per cell. Together they exceed the roughly 40 SU left, unless the simulation runs on CPU
    nodes and only the policy server holds a GPU. These figures are estimates; nothing reports
    OopsieVerse throughput.

**Minor:**
- The mean-residual baseline is a straw man, because left and right detours cancel.
- The simulated scene filter adds selection bias (it keeps at least 80% of scenes).
- Train and test come from one sampler, so "train vs test" shows interpolation only.
- `DamageSim`'s timestep is hardcoded to 0.005 s, so damage magnitudes depend on matching
  their setup exactly.

## What to change in the dummy env, in order

Each step has a go/no-go check, and the early ones need no new environment.

1. **Baselines and probes on the existing models, no retraining.**
   - Nearest-neighbour text lookup feeding `main`.
   - A one-hot class model.
   - Parsed constraint plus the scripted oracle, with a keyword or LLM parser and automatic
     clause splitting.
   - Already-safe scenes and laptop-off-path scenes.
   - Negated, permissive, irrelevant, absent-object and conflicting instructions.
   - Clearance and tilt distributions with threshold sweeps.

   *Go:* if the lookup or one-hot baseline matches the learned text pathway, treat language as a
   separate parsing problem. That means an LLM or retrieval front end producing a constraint ID
   and target object, rather than the flow learning it.
2. **Closed-loop chunked base.**
   - Re-plan a stochastic scripted base every 8 of 16 waypoints, and apply the residual per chunk.
   - Relabel with the oracle in the loop (DAgger).
   - Require zero correction on already-safe chunks.
   - Add a safety margin so corrections don't sit on the boundary.

   *Go:* violations stay low as the chunk size shrinks, with no over-intervention on safe chunks.
3. **Force constraints and real composition.** This is the expensive step, not a quick CPU job.
   - Replace the weld grasp with real finger contact and a continuous gripper channel. The weld
     exists because round 1 hit a mug that was flung on release and rocked when it landed, so
     this reopens that work. With the weld and the non-colliding arm, there is no contact force to
     measure crush, and a welded mug cannot drop.
   - Then plug OopsieVerse's `RSMechanicalDamageEvaluator` into the dummy MuJoCo scene (it reads
     only contacts and `cvel`). Match its hardcoded 0.005 s timestep or patch it: the dummy runs
     at 0.002 s, so its finite-difference impact forces would be about 2.5× off.
   - Add "don't crush" and "don't drop", which share the gripper.
   - Add a second obstacle that changes the right detour side.
   - Remove the sim filter on the joint reference.

   *Go:* clause stacking or composition beats the direct and trained-combined models against a
   joint oracle where the sum is wrong.
4. **Best-of-K oracle under a non-differentiable scorer** in place of the analytic oracle.
   *Go:* within 10 points of the analytic-oracle model.
5. **Noisy perception features and object-slot conditioning** in place of global FiLM.
   *Go:* a noisy-state model degrades gracefully.
6. **Evaluation hygiene.**
   - Three seeds (done: 0, 7 and 42, spread at most 0.04).
   - Confidence intervals clustered by wording.
   - 30 or more held-out wordings per constraint written blind (by an LLM or another person).
   - Every post-hoc addition declared as post hoc.

## Staged path to OopsieVerse

1. **Dummy v3:** steps 1–4 above. Steps 1, 2 and 4 are mostly CPU work, while step 3 rebuilds
   the grasp and needs days, not hours.
2. **Infrastructure reproduction:**
   - `DamageablePickEgg` plus one multi-object task, driven by a GR00T N1.6 server on a V100
     with eager attention.
   - Reconcile the robosuite and robocasa version pins between OopsieVerse and GR00T's RoboCasa
     setup.

   *Go:* task completion within about 15 points of the paper's GR00T number, and SC strictly
   between 0 and completion.
3. **Does a safe sibling exist?** On logged GR00T rollouts, count how often best-of-K (K ≤ 16,
   with lookahead) rescues a damaging episode. *Go:* at least 50%.
4. **Single-constraint residual on pick_egg** with privileged state. Compare against GR00T
   prompted with the safety clause and a no-text residual. *Go:* safe completion rises with
   completion not worse, paired McNemar p < 0.05.
5. **Text claims** (swap, paraphrase, composition) on the multi-object tasks, then images in
   place of privileged state. The OmniGibson spill and thermal track stays out of scope here.

## Sources

- OopsieVerse: arXiv 2606.31993 and github.com/UT-Austin-RobIn/oopsieverse, cloned read-only
  under `tmp/claude-101024/critic2/ovrepo`.
- GR00T configs in `oopsieverse_eval_cache/Isaac-GR00T`.
- Dummy env: `src/traj.py`, `data_v2/`, `runs_v2/results_v2.md`.
