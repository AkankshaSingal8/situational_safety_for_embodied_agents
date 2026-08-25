# Full Methodology Section (expanded §IV)

Drop-in replacement / appendix source for `main.tex` §IV (`sec:method`, currently lines 173--224).
**Not yet spliced into `main.tex`.** Written against `STYLE_RULES.md` (no em-dash asides, no
meta-commentary, sparse italics, formal connectives, third person).

Grounded in the implementation, not in the compressed §IV:

- `vlm_pipeline/symbolic_identity.py` (election layer, `fol_obstacle_id`)
- `vlm_pipeline/guard_set.py` (top-k guard set)
- `openpi_guided/src/openpi/models/pi0_guided.py` (`_prefix_acceptance`, selection
  scalarization, `adjoint_sample_actions`, `flow_decode_actions`)
- `vlm_pipeline/run_guided_safelibero_pi05_eval.py` (receding-horizon client, stall ladder,
  dual controller)

## Two corrections made against the code

- The prior-weighted scorer (`scored_obstacle_id`, with `sigma=0.25`, `_PRIOR_ALPHA=1.5`,
  `_VETO_TAU=0.6`, the `HAZARD_PRIOR` table) is the **baseline** row in `tab:e3`, not the
  deployed rule. The deployed rule is `fol_obstacle_id`, which consults no priors. None of those
  constants appear below as method parameters.
- `NEAR_PATH_THRESHOLD = 0.20` is only the per-replan de-election re-check. At election time the
  rule ranks and the guard-set `k` truncates, so it is not written below as a firing threshold.

The SDEdit/RENOISE entry point in `flow_decode_actions` exists but no run config uses it, so it is
omitted.

## Open decisions

1. **Length.** Roughly 2.5--3 columns against the current §IV's ~1.2, so about 1.5 columns net.
   The paper is already at 12 pages against "8+N". Either replace §IV wholesale, or keep the
   current §IV in the body and move IV-A, IV-E, IV-G, IV-H into an appendix.
2. **A disclosure made rather than assumed.** The current §IV does not mention the HDC lateral-bias
   detour, the scripted lift-retreat, the adaptive-replan cadence, or the integral repulsor
   controller. These are deployed machinery that affects the reported rows, so they are written in
   (IV-H). If they stay out of the body they should at least appear in Limitations, since a
   reviewer who reads the release will find them.
3. **Two soft mismatches corrected rather than preserved.** The current §IV says `Select` "scores
   candidates lexicographically"; the implementation scalarizes with feasibility weighted at
   `1e3`, so it is described below as feasibility-dominant scalarization. Proposition 1 previously
   asserted the chain is satisfied outright, where the acceptance test admits residuals up to
   `1e-4`; the tolerance is now stated. Both are numbers added, not numbers changed, so the
   `main.tex.prerewrite` numeric diff still passes.

## Preserved anchors

`sec:method`, `sec:ident`, `sec:routing`, `sec:enforce`, `eq:fol`, `prop:cert`, `fig:pipeline`,
`tab:ablation`, `sec:e3`, `sec:ls`, `sec:exp`, `sec:limitations`. New labels introduced:
`sec:percep`, `sec:ground`, `sec:barrier`, `sec:cert`, `sec:loop`, `sec:nonclaims`, `eq:barrier`,
`eq:dcbf`.

---

## LaTeX source

```latex
\section{Method}
\label{sec:method}

Fig.~\ref{fig:pipeline} shows the assembled system. Neural components answer
only local, checkable questions; a fixed symbolic layer makes every
safety-critical decision; and a certificate re-checked on the executed actions
makes soundness independent of the search that produced them. We describe the
five layers in the order in which the runtime executes them.

\subsection{Perception tier and what the system reads}
\label{sec:percep}

Every claim in Sec.~\ref{sec:exp} is stated at a declared privilege tier.
\textit{Tier-GT} reads hazard identity and object positions from the simulator,
which is the tier at which prior in-denoising work reports. \textit{Tier-Percep}
reads only the instruction, the RGB and depth images, proprioception, and the
symbolic object name list, so that identity, positions, extents, and corridor
anchors are all derived from perception or from the vision-language model.

At the perception tier, object positions and extents are obtained from
open-vocabulary detection~\cite{gdino} prompted with the scene's name list,
back-projected through the depth image into the metric world frame, with extents
taken as percentile boxes over the back-projected support rather than as raw
bounding-box corners, so that a single depth outlier cannot inflate a guard.
Median localization error is 0.067\,m on SafeLIBERO and 0.036--0.038\,m on
LIBERO-Safety. Localization failures fall back conservatively, retaining the
previous estimate and widening the guard, and are logged per episode so that the
rate is reported rather than absorbed.

One perceptual quantity is not read from images. The set of moving entities is
grounded from inter-frame displacement over a settle window, with a threshold of
more than 1\,cm, and is re-read at every replan. This is what allows a dynamic
intruder to be handled without any runtime model call.

\subsection{Hazard election: a three-literal first-order rule}
\label{sec:ident}

The scene-level decision is made by a fixed rule, not by a model:
\begin{equation}
\begin{aligned}
\textsc{Hazard}(x) \coloneqq{}& \textsc{Protected}(x)\ \vee\ \textsc{Moving}(x)\ \vee \\
&\big(\lnot\textsc{Mentioned}(x)\wedge\textsc{NearPath}(x)\big),
\end{aligned}
\label{eq:fol}
\end{equation}
ranked protected-first, then by distance to the sanctioned reach path.

$\textsc{Protected}$ covers a semantic class comprising body parts and objects
attached to a person. Membership in this class is never removed by mention, on
the grounds that an instruction naming a hand as a destination does not
authorize contact with it, and the path term is floored for protected entities,
because an intruder's distance at episode start carries no information about
where it will be later.

$\textsc{Moving}$ is the displacement predicate of Sec.~\ref{sec:percep}. It
receives the same two overrides as $\textsc{Protected}$, being exempt from
mention exclusion and floored in the path term, and it is the single literal
that handles dynamic human-hand intruders.

$\textsc{Mentioned}$ is decided by content-token overlap between the object's
cleaned name and the instruction, under a head-noun rule: if the final content
token of a multi-word name appears in the instruction, the object is treated as
referenced in full. This rule exists because multi-word visual descriptors
dilute plain token-fraction matching, so that an object the instruction plainly
names can otherwise survive as a partial match and be elected as its own
obstacle. Naming conventions in the benchmark asset files are stripped before
any matching, so no \texttt{\_obstacle} suffix is ever consulted and identity is
decided from the instruction and the geometry alone.

$\textsc{NearPath}$ measures distance to the sanctioned reach segments, which
run from the current end-effector position to the parsed target and then to the
parsed destination. The target is parsed from the instruction by a small set of
verb patterns over the candidate name list. Multiple segments are supported and
the distance is the minimum over them, which matters on the long-horizon suites,
where the obstacle frequently blocks the approach to the second goal rather than
the first. At election time this literal ranks rather than gates: candidates are
ordered by path distance and the guard-set size truncates. A distance threshold
of 0.20\,m is applied only in the per-replan re-check of Sec.~\ref{sec:loop},
which asks whether an already-elected near-path guard still satisfies the
disjunct under which it was elected.

\textbf{From an election to a guard set.} The rule returns a ranking, and the
deployed system guards its top element. A documented top-2 mode is also
available, motivated by the observation that identification errors fail unsafe
(Sec.~\ref{sec:e3}): under every corruption we injected, path geometry alone
keeps the true hazard within the top two, so the set recovers 19/19 recall where
the argmax recovers 3/19. The set costs 12.5 task-success points when it is
engaged unnecessarily, and we therefore report it rather than enabling it by
default.

\textbf{What the rule is not.} It is not synthesized per scene, not retrieved,
and not conditioned on a benchmark-specific table. It is deliberately small
enough to read in one line, and every literal is separately falsifiable, which
is what makes the ablation of Sec.~\ref{sec:ablation} possible at all. On the
affordance suite the danger literal is instantiated as a vision-language-answered
property predicate, either $\textsc{Hot}$ or $\textsc{Sharp}$, in place of
$\textsc{NearPath}$, and the property-gated form has no fallback to unmentioned
near-path objects: if nothing is property-positive, the election is empty and
nothing is guarded. This routing is declared per suite in advance rather than
selected per episode, and the value of abstention over force-electing is
measured in Table~\ref{tab:ablation}.

\subsection{Two interchangeable grounding backends}
\label{sec:ground}

The literals can be grounded in either of two ways, and the choice does not
change the rule. The first uses hand heuristics together with one-shot cached
vision-language priors over hazard properties. Those priors enter under a
fail-safe floor, in which semantics may raise concern about an entity but may
never remove protection from an unmentioned near-path object, so a
misjudged prior can only over-guard. The second backend uses \textit{VLM
predicate grounding}, in which a small vision-language model answers only the
local per-object questions $\textsc{Referenced}(x)$ and $\textsc{Protected}(x)$,
with 3 votes per question, cached per task, and no runtime API call inside the
control loop.

On all 59 scenes of the two benchmarks the two backends produce identical
elections. The vision-language backend additionally resolves references that no
string heuristic can, since the instruction ``deliver it to me'' makes the hand
$\textsc{Protected}$, and a plate held by a hand is correctly both
$\textsc{Referenced}$ and $\textsc{Protected}$. The design finding may be stated
in a single sentence: the vision-language model answers local predicates, and
the logic makes the decision.

\subsection{Three-regime routing}
\label{sec:routing}

Not every hazard is a keep-out region, and treating every hazard as one is a
failure mode that we measured. The same predicate layer routes each episode into
one of three regimes.
(1) \textbf{Avoid}, in which a geometric hazard lies off the task path and is
enforced by the barrier and search of Secs.~\ref{sec:barrier}
and~\ref{sec:enforce}.
(2) \textbf{Disengage}, in which the hazard is itself the destination, indicated
by $\textsc{Protected}(x)\wedge\textsc{Referenced}(x)$ or by the guard
overlapping the parsed destination within 0.15\,m. The geometric keep-out is
then disengaged for the episode and safety is obtained from routing instead.
This regime was required by the human-handover suites, in which any keep-out
region destroys the task (Sec.~\ref{sec:ls}).
(3) \textbf{Refuse}, in which the instruction itself is the hazard. A rule-based
semantic judge over triples of verb, patient, and hazard property refuses
execution, with offline F1 0.968, precision 0.938, and recall 1.000, measured
in-domain as discussed in Sec.~\ref{sec:limitations}.
Routing and identification read the same predicates, so this is one rule system
rather than three.

\subsection{Compiling the elected entity into a barrier}
\label{sec:barrier}

Election returns a named entity with a position and an extent, which is exactly
the object that a classical filter has always assumed it was given. Everything
downstream is standard.

Let $p_j$ denote the predicted end-effector position after the $j$-th step of a
decoded chunk, obtained by accumulating the chunk's unnormalized translational
commands from the measured current pose. The barrier is
\begin{equation}
B_j(p_j)=\min_{m,q}\big\lVert p_j+q-o_m\big\rVert - r_m(j),
\label{eq:barrier}
\end{equation}
a minimum over guarded obstacles $m$ and over companion offsets $q$, taken as
the end-effector origin together with a wrist point 8\,cm above it and a
fingertip point 6\,cm below it, so that the guard covers the gripper body rather
than a single frame origin. The effective radius grows along the chunk,
$r_m(j)=r_m+\lambda j$, which bounds obstacle motion within the chunk without
requiring a motion model. Safety of a prefix is the discrete control barrier
function chain
\begin{equation}
B_j \ \ge\ (1-\gamma)\,B_{j-1},\qquad \gamma=0.9,
\label{eq:dcbf}
\end{equation}
seeded at $B_0$ computed from the measured pose. The geometric decay condition
is standard~\cite{agrawal2017dcbf} and we claim no novelty for it.

One relaxation is load-bearing. A guard inflated to cover a graspable object
places the sanctioned grasp target inside the keep-out region, so an
unrelaxed chain vetoes precisely the chunks that complete the task. We therefore
apply a corridor scale to both the radius and the shape term of
Eq.~\eqref{eq:barrier}, relaxing the guard along the sanctioned reach corridor.
The relaxation reaches the shape term as well as the scalar margin, because
relaxing only the margin reproduces the veto for anisotropic guards.

\subsection{Enforcement by search over the noise preimage}
\label{sec:enforce}

Let $F(z,\mathrm{obs})$ denote the frozen 10-step Euler denoise map of the
flow policy and let $M(a_{1:L})$ denote the minimum barrier-chain margin of the
executed $L$-step prefix. All operators search the noise preimage
$S=\{z: M(F(z))\ge 0\}$, so that every emitted action is $F(z^\star)$ for some
sample $z^\star$ and therefore lies exactly on the policy manifold. No operator
edits a decoded action.

\textsc{Select} draws $K{=}8$ noise samples and decodes them under a single
shared vision-language KV cache, so that the prefix is encoded once and the $K$
suffix passes are the only additional cost. Candidates are scored by a
feasibility-dominant scalarization of the acceptance quantities of
Sec.~\ref{sec:cert}, weighting chain feasibility at $10^3$, prefix clearance
margin at $10$, and net prefix displacement at $0.1$, with two further
configurable terms for directed progress toward the parsed target and for the
worst clearance over the post-prefix horizon. The feasibility weight dominates
the achievable range of the remaining terms, so a feasible candidate is always
preferred to an infeasible one, while the margin, progress, and lookahead terms
trade against one another within each feasibility class. The displacement and
progress terms exist so that selection does not reward freezing, which is the
degenerate solution available to any margin-maximizing filter.

\textsc{Tilt} reweights and resamples the particle population at mid-denoising
steps under soft margin potentials, in the manner of a Feynman-Kac correction,
at essentially no additional cost over \textsc{Select}, since the population is
already being carried.

\textsc{Ascend} makes the search explicit. It scores the $K$ seeds by prefix
margin, takes the best, and then performs reverse-mode gradient ascent of the
prefix margin with respect to $z$, differentiating through all ten Euler steps
with the backbone parameters untouched, using normalized gradient steps that are
gated off once the margin exceeds a threshold, so that a candidate which is
already comfortably safe is not perturbed further. The cost is 57\,ms per
ascent step.

A soft annealed repulsor field on the velocity target is the natural soft
complement to these operators. It is the only mechanism we found that navigates
obstacle-on-grasp geometry, in which every hard projection locks up, and its
strength is annealed to zero as the gripper enters the grasp phase, since a
repulsor that remains active during the grasp fights the task.

\subsection{The certificate and its acceptance semantics}
\label{sec:cert}

Acceptance is evaluated on the executed prefix rather than on the full chunk,
because the receding-horizon client executes only the prefix before replanning.
The acceptance stage recomputes Eq.~\eqref{eq:barrier} and Eq.~\eqref{eq:dcbf}
over the first $L{=}5$ steps of the final decoded chunk, by explicit rollout of
the actuation model from the measured pose, and declares the candidate feasible
when the largest chain residual does not exceed a numerical tolerance of
$10^{-4}$.

\begin{proposition}[Checked prefix certificate]\label{prop:cert}
If the acceptance check passes, the executed prefix satisfies the chain
Eq.~\eqref{eq:dcbf} to the stated tolerance, regardless of how the chunk was
produced. If all $K$ candidates fail, the candidate with the largest margin
executes and its true margin is reported, so that the failure is visible rather
than silent.
\end{proposition}

Because soundness is a predicate of the emitted actions rather than of the
search that produced them, the certificate covers \textsc{Select},
\textsc{Tilt}, \textsc{Ascend}, and the repulsor identically, and it would cover
any further operator added later without modification. Correction-based
arguments, of the form that the repair enforces the chain, cannot establish
this, and prior in-denoising work does not re-verify its intermediate
corrections on the final chunk at all.

\subsection{The receding-horizon loop}
\label{sec:loop}

At each replan the client reads the current pose and the current object
estimates, re-evaluates $\textsc{Moving}$ over the settle window, re-checks that
each near-path guard still satisfies the disjunct under which it was elected,
using the 0.20\,m threshold of Sec.~\ref{sec:ident}, and issues a decode
request. The returned chunk's first 5 steps execute, and the loop repeats. The
replan cadence tightens automatically when the certified margin is small, so
that the system replans twice as often inside the tight zone.

Two further mechanisms handle the liveness failure that motivates this paper.
The first is an integral controller on the repulsor strength, which raises the
field whenever the certified margin of the selected chunk runs below its
setpoint and decays it whenever the margin exceeds it, bounded above so that the
field cannot dominate the policy. The second is a stall ladder. If the
end-effector fails to make progress over a fixed window, the system first issues
three replans carrying a small lateral detour bias applied to half of the $K$
seeds, leaving the remainder unbiased, and only if that fails does it execute a
scripted lift-and-retreat, at most twice per episode, after which normal
guidance resumes from the new pose. Both mechanisms act on the search, not on
the emitted action, so Proposition~\ref{prop:cert} continues to hold over them.

\subsection{Non-claims}
\label{sec:nonclaims}

Several limits of this construction should be stated where the construction is
given rather than only in Sec.~\ref{sec:limitations}. The rotation channels are
uncertified, since the barrier is a predicate over translational end-effector
geometry alone. Tracking error between the commanded displacement and the
realized motion is absorbed into the safety margin rather than proved for a
specific low-level controller. The chunk-horizon inflation $\lambda j$ bounds
obstacle motion at 8\,cm/s or less at 20\,Hz, which is below natural hand
speeds. We claim no formal perception-robustness radius, and quantify robustness
empirically instead (Sec.~\ref{sec:e3}). Finally, the certificate is a statement
about the executed prefix under the actuation model, not about the physical
simulator, and the two are reconciled only by the empirical collision criterion
of Sec.~\ref{sec:exp}.
```
