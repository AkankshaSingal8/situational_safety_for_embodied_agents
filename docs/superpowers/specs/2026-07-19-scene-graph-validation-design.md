# Scene Graph for Semantic Safety: Go/No-Go Validation Design

Date: 2026-07-19. Status: V1 EXECUTED same day (user directed continuous work).
V1 RESULT: 100% agreement (19/19) flat-v3 vs graph-rule identification —
pre-registered NO-GO for SafeLIBERO confirmed. Construction: vlm_pipeline/
scene_graph.py (worktree). V2 gated on LIBERO-Safety priority unparking.
Question: is an explicit scene graph (typed nodes + relational edges) worth adding
to the no-GT identification pipeline, and if so, how do we build it cheaply?

## 1. Current state (what the graph would replace or wrap)

The no-GT pipeline already computes, in scattered form, everything a scene graph's
*nodes* would hold:

- **Perceptual entity map** (Stage 1): per-object name, 3D position (RGB-D
  back-projection off gt_seg or GDINO masks), fail-safe drops. This is a flat
  metric-semantic object list — a scene graph with zero edges.
- **Symbolic identity v3** (Stage 2): hazard_prior × path-proximity Gaussian ×
  mention-penalty. Path proximity is an implicit PATH_BLOCKS edge computed
  on the fly; mention parsing implicitly computes TARGET/DEST edges.
- **parse_entities** (Stage 3): explicit TARGET/DEST binding for the corridor.

So "adding a scene graph" concretely means: reify these scattered computations
into one typed data structure and add *general relational edges* (ON, NEAR,
ABOVE, INSIDE, HELD_BY) plus *semantic hazard rules over edges*
(towel-ON-stove → hazard; knife-NEAR-table_edge → margin+velocity constraint).

## 2. Where a graph could pay — and where it provably cannot

**SafeLIBERO (the current table):** every cell has exactly one active obstacle,
identified from name priors + path proximity. Symbolic identity already scores
39/40 with percep positions and 4/8 no-GT cells beat baseline+reimpl on both
axes. There is no relational hazard in the benchmark: no ON/NEAR-induced danger,
no multi-object interaction. A graph cannot add information here — only
refactor. **Expected result: no table movement.**

**LIBERO-Safety:** hazards are relational and dynamic by construction —
intruding human hand (never task-mentioned, appears mid-episode),
obstacle_avoidance with hazard-adjacent placements, reasoning_safety
(instruction-level). Hand intrusion needs a node that enters at runtime;
hazard-adjacency needs NEAR/ON edges. **This is the arena where edges can
change identification outcomes.**

**Paper framing:** the graph is the "unified stack" generalization argument
(Direction-1 survey gap #2) and the compilation contract
(graph → {keep-out geometry, margins, velocity boxes, rotation locks})
is the C2 FOL-COMPILE story. It earns a paper section only if the
validation below shows measurable lift somewhere.

## 3. Approaches considered

**A. Full 3D scene-graph stack (ConceptGraphs / Hydra style).** Open-vocab
detection + incremental 3D fusion + LLM edge labeling. Rejected: weeks of
integration, GPU-heavy per-frame, and SafeLIBERO/LIBERO-Safety scenes have
≤8 objects on one table — the machinery is wildly oversized.

**B. Lightweight metric-semantic graph over the existing entity map
(recommended).** Nodes = entity-map entries (name, percep 3D pos, mask,
property priors). Metric edges computed by geometric thresholds over percep
positions/extents (ON: z-stack + xy-overlap; NEAR: d < τ; PATH_BLOCKS: the
existing path-segment distance). Task edges from the existing parse
(TARGET/DEST/MENTIONED). Semantic layer = FOL rules over (node-property, edge)
pairs, reusing the v17 grounder + property priors. Cost ≈ 2–3 days because
every ingredient exists; construction is a refactor plus ~200 lines of edge
predicates.

**C. VLM-labeled edges.** Ask a VLM to emit relations per scene. Rejected as
the *primary* path by the fresh VLM-ablation NO-GO (qwen7b 79.5% vs table
100%, all failures fail-unsafe); kept as an optional S-arm inside the offline
benchmark (same harness as vlm_slot_bench) to re-test on *relational* queries,
where tables have no prior — this is the one place VLM could still win.

## 4. Validation plan (pre-registered, cheapest gate first)

**V1 — Offline equivalence test on SafeLIBERO (CPU, ~1 day).**
Build graph constructor; run flat-v3 vs graph-rule identification on the 39
saved capture scenes (+ any new captures). Metric: agreement rate on
(obstacle id, constraint set). Pre-registered expectation: ≥95% agreement.
- If ≥95%: **NO-GO for SafeLIBERO** — graph cannot move the main table;
  do NOT spend GPU on E3-style swaps there. Record and move to V2.
- If <95%: inspect disagreements; any cell where graph fixes a flat mispick
  escalates to a runtime identification-swap smoke (n=10, guidance fixed).

**V2 — LIBERO-Safety offline hazard-identification benchmark (CPU, ~2 days,
gated on assets extraction, job 42405002).**
Label 30–60 LIBERO-Safety scenes (obstacle_avoidance, human_safety) with GT
(hazard object, hazard relation, required response class). Run three arms:
flat-v3 (priors + path proximity only), graph rules (B), graph + VLM edge
labels (C, optional). Metrics: hazard-identification F1, response-class
accuracy, and **error direction** (fraction of errors that fail unsafe — the
E5 fail-safety framing).
- **GO** iff graph beats flat by ≥5pp F1 OR covers a hazard class flat cannot
  express at all (e.g., mid-episode hand intrusion, towel-ON-stove) with
  F1 > 0.7 on that class.
- **NO-GO** otherwise: keep the flat map, mention the graph as future work.

**V3 — Runtime rows (GPU, only on V2 GO).**
LIBERO-Safety method rows (already pre-staged in slurm/libero_safety_rows.slurm)
with identification source = graph vs flat, guidance identical. The graph
earns its paper section iff it moves TSR/CAR (or SSR) on at least one suite
beyond noise.

**Decision timeline:** V1 can start immediately (uses existing captures);
V2 gated only on the assets unzip; total cost before any GPU commitment:
~3 days CPU. Kill criteria are pre-registered above — if both V1 and V2 come
back NO-GO the total sunk cost is ~3 days and we keep the (already winning)
flat map.

## 5. Construction plan (approach B, executed only as far as validation needs)

1. `vlm_pipeline/scene_graph.py`: `SceneGraph` dataclass — `nodes:
   {name: Node(pos, extent, mask_area, priors)}`, `edges: [(src, rel, dst,
   score)]`, built from the existing `estimate_object_positions` output +
   `parse_entities`. Deterministic, no learned components.
2. Metric edge predicates: ON / NEAR / ABOVE / PATH_BLOCKS with thresholds
   calibrated once on 10 held-out scenes (report sensitivity ±50% threshold).
3. Rule layer: `hazard_rules(graph) -> [(node, constraint_class, params)]`
   reusing property priors; rules authored per hazard class, each rule names
   its compilation target (keep-out / margin / velocity / rotation) — the C2
   compilation contract.
4. V1 harness: replay saved scenes, emit agreement report JSON →
   `results_tables/scene_graph_v1_agreement.json`.
5. (V2, post-unzip) LIBERO-Safety scene capture + labeling script, then the
   three-arm benchmark.

## 6. Risks

- Percep position noise (~6cm) can flip NEAR/ON edges → report edge accuracy
  vs GT positions in V1; thresholds must exceed 2× noise floor.
- Labeling GT for V2 is manual effort; cap at 60 scenes.
- Scope creep: the graph must stay an *identification* module behind the same
  interface (`obstacle_id_source`, constraint list) — no changes to guidance.
