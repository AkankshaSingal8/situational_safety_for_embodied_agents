"""Lightweight metric-semantic scene graph over the perceptual entity map
(spec: docs/superpowers/specs/2026-07-19-scene-graph-validation-design.md,
approach B). Deterministic, no learned components: nodes are entity-map
entries; metric edges come from geometric thresholds on (percep) positions;
task edges from the instruction parse. The hazard-rule layer maps
(node property, edge) pairs to guidance constraint classes — the C2
compilation contract.

Thresholds are >= 2x the ~6cm percep noise floor (spec risk section).
"""

import dataclasses
from typing import Optional

import numpy as np

import symbolic_identity as si

NEAR_T = 0.12       # centers closer than this in xy -> NEAR
ON_XY_T = 0.06      # xy alignment for a support relation
ON_DZ = (0.02, 0.25)  # dz band for ON/ABOVE (above support, below hover)
PATH_T = 0.12       # xy distance to a reach segment -> PATH_BLOCKS


@dataclasses.dataclass
class Node:
    name: str
    pos: np.ndarray                      # (3,) metric
    prior: float                         # property prior (table or VLM)
    mention_frac: float = 0.0
    extent: Optional[np.ndarray] = None  # (3,) half-extents when available


@dataclasses.dataclass
class SceneGraph:
    nodes: dict          # name -> Node
    edges: list          # (src, rel, dst, score)
    goals: list          # task-referenced goal names (TARGET/DEST carriers)
    eef_pos: np.ndarray

    def out_edges(self, name, rel=None):
        return [e for e in self.edges
                if e[0] == name and (rel is None or e[1] == rel)]


def build_graph(task_description: str, candidate_positions: dict,
                eef_pos: np.ndarray) -> SceneGraph:
    task = task_description.lower()

    def mention_frac(key):
        toks = [w for w in si._clean_object_name(key).lower().split()
                if len(w) >= 3 and w != "obstacle"]
        if not toks:
            return 0.0
        f = sum(w in task for w in toks) / len(toks)
        if toks[-1] in task:
            f = 1.0
        return f

    nodes = {n: Node(n, np.asarray(p, dtype=float), si._hazard_weight(n),
                     mention_frac(n))
             for n, p in candidate_positions.items()}
    goals = [n for n, nd in nodes.items() if nd.mention_frac >= 0.5]

    edges = []
    names = list(nodes)
    for i, a in enumerate(names):
        pa = nodes[a].pos
        for b in names[i + 1:]:
            pb = nodes[b].pos
            dxy = float(np.linalg.norm(pa[:2] - pb[:2]))
            dz = float(pa[2] - pb[2])
            if dxy < ON_XY_T and ON_DZ[0] < dz < ON_DZ[1]:
                edges.append((a, "ON", b, 1.0 - dxy / ON_XY_T))
            elif dxy < ON_XY_T and ON_DZ[0] < -dz < ON_DZ[1]:
                edges.append((b, "ON", a, 1.0 - dxy / ON_XY_T))
            elif dxy < NEAR_T:
                edges.append((a, "NEAR", b, 1.0 - dxy / NEAR_T))
                edges.append((b, "NEAR", a, 1.0 - dxy / NEAR_T))

    spos = np.asarray(eef_pos, dtype=float)[:2]
    segs = [(spos, nodes[g].pos[:2]) for g in goals] or \
           [(spos, np.array([0.0, 0.15]))]
    for n, nd in nodes.items():
        p = nd.pos[:2]
        best = np.inf
        for s, g in segs:
            v = g - s
            l2 = float(v @ v)
            t = 0.0 if l2 < 1e-9 else float(np.clip((p - s) @ v / l2, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (s + t * v))))
        if best < PATH_T * 2.5:  # keep a soft tail; rule layer re-weights
            edges.append((n, "PATH_BLOCKS", "__path__", float(np.exp(
                -best ** 2 / (2 * 0.25 ** 2)))))
    return SceneGraph(nodes, edges, goals, np.asarray(eef_pos, dtype=float))


def hazard_rules(graph: SceneGraph) -> list:
    """Rules over (property, edge) pairs -> (node, constraint_class, score).

    R1 keep-out: unmentioned node with PATH_BLOCKS edge, scored by
        prior x path score x (1 - 0.8 mention) — the flat-v3 decision
        expressed as a graph query (V1 equivalence target).
    R2 margin: fragile/spillable node NEAR a goal -> margin constraint.
    R3 support: node that a goal is ON -> protected support (never keep-out).
    """
    out = []
    supports = {dst for _, rel, dst, _ in graph.edges if rel == "ON"}
    for name, nd in graph.nodes.items():
        if nd.mention_frac >= 1.0 or name in supports:
            continue
        path = graph.out_edges(name, "PATH_BLOCKS")
        pscore = path[0][3] if path else 0.0
        score = nd.prior * pscore * (1.0 - 0.8 * nd.mention_frac)
        if score > 0.0:
            out.append((name, "keep_out", score))
        if nd.prior >= 0.7 and any(e[2] in graph.goals
                                   for e in graph.out_edges(name, "NEAR")):
            out.append((name, "margin", nd.prior))
    return sorted(out, key=lambda t: -t[2])


def graph_obstacle_id(task_description, candidate_positions, eef_pos):
    """Graph-rule identification: top keep_out node (V1 interface)."""
    g = build_graph(task_description, candidate_positions, eef_pos)
    keep = [r for r in hazard_rules(g) if r[1] == "keep_out"]
    return keep[0][0] if keep else None
