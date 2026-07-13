# ROCS — Runtime Outcome-Conditioned Steering (spec draft, 2026-07-13)
**The user's framing:** context comes from the scene; steering enforces commonsense right/wrong at the level of *outcomes* ("placing a glass on a laptop is wrong"), not just collisions. If the policy is generating that action at runtime, we steer it away — toward what is contextually right.

## 1. The gap (precise)
All surveyed steering — the SOTA's DCBF spheres, our companion barriers, OmniGuide's repellers, risk fields (2512.08233), Brunke's spatial constraints — constrains the robot's **trajectory geometry**: don't be near X, don't be above Y. But commonsense hazards are **relational outcomes conditioned on action semantics**:
- Glass ON laptop is wrong; glass PASSING OVER laptop in transit is fine. A static "not-above-laptop" barrier (Brunke's `above` constraint) wrongly vetoes transit; no barrier at all wrongly permits placement.
- The event that realizes the hazard is a **release** (gripper opens) whose predicted landing region is the laptop — an *action-semantic* event, detectable in the chunk (gripper dim crossing its open threshold) BEFORE it happens, because chunked flow VLAs predict 10 steps ahead.
- Same family: pouring (rotation event while holding liquid over the wrong region), activating (turn on stove while towel ON it), impact-releases (dropping heavy onto fragile).
**Nobody steers generation against predicted OUTCOME relations.** Our chunk rollout already predicts the trajectory; the missing move is predicting the *event structure* of the chunk (where does it release? what lands where? what rotates when?) and constraining THAT.

## 2. The method
### Context → forbidden-outcome rules (Direction 1, episode start + per-chunk refresh)
VLM scene pass (objects + properties) → runtime rule authorship (v16 machinery, built) emits **outcome rules** over a grounded vocabulary:
```
FORBIDDEN_ON(x, y)      e.g. (glass, laptop), (towel, stove), (knife_edge, table_edge)
FORBIDDEN_RELEASE_OVER(x, region)      derived: release of x whose landing set intersects y's support region
FORBIDDEN_TILT(x, θ_max | HOLDING(x) ∧ CONTAINS_LIQUID(x))
SANCTIONED_ON(x, {y1, y2, ...})        the commonsense-right alternatives (table, plate, shelf)
```
Each rule carries its activation condition (HOLDING(x), gripper state, task phase) — **conditional constraints**, active only when the predicate holds, so transit is never wrongly blocked.

### Chunk-level outcome prediction (new, cheap, jittable)
From the denoised chunk: (a) EEF path p_1..p_H (existing rollout); (b) **release index** j* = first j where gripper dim crosses the open threshold (differentiable soft-min over sigmoid crossings); (c) **landing point** ℓ = (x,y) of p_j* projected to the support surface below (region lookup against scene boxes); (d) rotation excursions per step (dims 3:6, axis-angle magnitude). This is an *event parse of the action chunk* — the object no other method computes.

### Steering (Direction 2, three new repair channels inside the existing sweep)
1. **Release barrier**: if ℓ ∈ forbidden region for the held object → repair = (i) clamp gripper dim closed through step j* (delay release — 1-D clamp, trivially jittable), and (ii) add a translational attractor toward the nearest SANCTIONED_ON region so the release relocates rather than never happening. Task completion is served, not traded: the policy still places the glass — next to the laptop.
2. **Conditional keep-out**: rules that forbid even transit (heavy-above-fragile) activate a companion-point barrier over y's region only while HOLDING(x) — the existing DCBF machinery with a predicate gate (one multiply).
3. **Rotation-event lock**: existing rotation-lock design, activated by the same rule engine (tilt barrier on dims 3:6 while holding liquid, everywhere or over forbidden regions only).
All three compose with the current stack (schedule, brake, K-selection: candidates whose event parse is clean win selection; prefix certificate extends to "no forbidden event in executed prefix").

### Why this is jointly safety+task (not a trade)
The repair's degrees of freedom are WHEN and WHERE the outcome happens, not WHETHER the task happens. Delaying a release and attracting it to a sanctioned region *is* task completion under commonsense constraints — the LIBERO-Safety SSR axis nobody executes ("safe alternative execution": published systems only refuse, best F1 0.31-0.46).

## 3. Novelty seams (checked against our full survey map; re-verify at submission)
- **Brunke RA-L25** has "water cup above laptop" as a STATIC spatial constraint (post-hoc CBF): blocks transit, can't distinguish carry-over from place-on, no event structure, no alternatives. We cite as the closest constraint-vocabulary ancestor; delta = event-conditional activation + release/landing prediction + in-generation enforcement + sanctioned-alternative attraction.
- **2512.08233 risk fields**: dense geometric risk conditioned on held object — still WHERE, not WHAT-BECOMES-TRUE; no action-semantic events; planning-time not generation-time.
- **HazardArena option layer**: discrete action gating (allow/block), no continuous steering, no alternative execution.
- **RoboGuard / SSR refusal line**: plan-level veto or refusal; never executes the commonsense-right variant.
- **Our own prior components**: corridor exemption modulates margins by task sanction (WHERE); ROCS constrains outcomes (WHAT) — complementary, same payload/compiler plumbing.
- **Gripper-dim steering is unclaimed anywhere** in the surveyed literature (every method: translational only, rotation rarely, gripper never).

## 4. Evaluation fit
- **LIBERO-Safety SSR L1/L2**: the marquee — "hold the bowl above the candle", "towel on stove", "egg in microwave" are literally FORBIDDEN_ON/RELEASE cases; metric upgrade from refusal-F1 to **safe-alternative execution rate** (pre-registered protocol).
- **HazardArena twins**: safe/unsafe scene pairs isolate context-dependence (same instruction, different scene → different steering).
- **SafeLIBERO**: unchanged (geometric tier); ROCS adds no regression risk (rules inactive when no outcome hazard).
- **OopsieVerse**: FORBIDDEN_RELEASE over hard surfaces for fragile payloads (drop-damage) — partially recovers the drop class previously scoped out.
- **Demo figure**: same instruction "put the glass down", laptop present vs absent — policy alone places on laptop; ROCS relocates release. Zero retraining.

## 5. Cost & risks
Event parse + release channel ≈ 2-3 days (rollout exists; gripper dim is one column; region lookup = box tests). Rule authorship exists (v16). Risks: gripper-threshold calibration (openpi gripper convention — verify against episodes); landing prediction is ballistic-naive v1 (directly-below; fine for tabletop); attractor-vs-policy conflict (reuse corridor line-search + hysteresis); rule quality gates via the same conformal machinery as C2; LIBERO-Safety fine-tuned π0.5 needed for SSR runs (released).
