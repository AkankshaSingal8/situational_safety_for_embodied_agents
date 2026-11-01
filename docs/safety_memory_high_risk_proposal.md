# Discovering What a Robot Must Remember for Safety

## Research Thesis

The ambitious research target is a robot that **learns which distinctions in its history must survive in memory to predict and prevent future harm**. Its memory should acquire new state variables when its existing representation cannot distinguish safe from unsafe futures. Across deployments, it should retain the learned update mechanisms and reuse them with new objects, tasks, and combinations of hazards.

This is a higher-risk alternative to accumulating conditional safety contracts. The core contribution would be representation discovery driven by safety counterexamples, with persistent transfer and closed-loop VLA guidance. Logic can provide executable structure, but a fixed vocabulary of human-written safety predicates should not determine everything the system can learn.

Proposed paper title: *Discovering What to Remember: Continual Safety-State Invention for Vision-Language-Action Policies*.

The proposal below is a research hypothesis, not a demonstrated method or an established novelty claim. The literature assessment covers sources available through September 14, 2026. The broader [safety-memory review](safety_memory_research.md) covers constraint learning, VLA safety controllers, intervention feedback, and deployment details.

## The Distinguishing Example

Consider two visually identical scenes: a tool rests beside a clean object. In one episode, the tool previously contacted a harmful substance; in the other, it did not. Contacting the clean object is unsafe only in the first episode. An instantaneous scene graph cannot distinguish these situations.

A fixed memory system might already contain a human-designed `contaminated(tool)` variable. The proposed system initially has only observations, tracked entities, action histories, and outcome feedback. After discrepant outcomes, it discovers a persistent, object-bound state that changes following particular contacts, propagates through later contacts, and can be reset by an effective cleaning operation. A human-readable name is optional; its defining feature is that it predicts intervention outcomes.

The learning challenge is not merely to remember a contact frame. It is to infer a reusable state-update mechanism, including its uncertainty and exceptions. The system should subsequently handle a new tool and a new multi-object contact chain without storing an explicit rule for every pair.

Begin with simulated transfer of a harmless marker as a contamination analogue. Genuine biological or chemical contamination would require domain-specific sensing and validated dynamics; visual evidence alone cannot establish those physical properties. Also include safe contact cases and ineffective cleaning cases, so a blanket contact prohibition cannot solve the benchmark.

## The Novelty Boundary

Several attractive descriptions are already occupied by prior work:

| Prior work | Established contribution | Required distinction |
|---|---|---|
| SOMA, 2026 preprint | Contrastive successful/failed experience retrieval, attribution-driven orchestration, and consolidation for frozen VLA adaptation.[^1] | Neither contrastive memory nor failure attribution is sufficient novelty. Learn a new executable history representation, not just better intervention examples. |
| EventVLA, 2026 preprint | Event-driven visual evidence retention for long-horizon action policies.[^2] | Choosing useful past frames is insufficient. Demonstrate persistent discovery and reuse of an unprovided hazard state. |
| Predictive state representations | State can be represented by action-conditional predictions of future observations.[^3] | Predictive memory itself is foundational prior art. The research target is safety-driven discovery and continual compositional reuse. |
| FUSION, RA-L 2024 | Safety-aware causal representations and bisimulation-based learning for offline driving.[^4] | Safety-aware state abstraction is not new. Distinguish history aliasing, representation growth, and transfer across deployments. |
| Safe RL in Tensor RKHS, 2023 preprint | Applies predictive state representations to partially observable safe RL.[^5] | Combining PSRs and safety is explicitly not a first-of-its-kind contribution. |
| Predicate Invention for Bilevel Planning, AAAI 2023 | Learns predicates from demonstrations for robot planning.[^6] | Predicate invention itself is established. Target temporal safety state induced by outcome discrepancies. |
| InterPreT, RSS 2024; VisualPredicator, ICLR 2025 | Learn relational predicates from feedback or neuro-symbolic visual abstractions for planning.[^7][^8] | A VLM generating new predicates is insufficient. Require executable historical updates and intervention-based validation. |
| SkillWrapper, author project and manuscript | Actively collects robot experience and builds predicates from contrasting examples to model black-box skills.[^9] | Online contrastive predicate growth is also close prior art. Emphasize safety-predictive hidden state, not ordinary skill preconditions. |
| POMDP CEGAR and automata-learning CEGAR | Refine inadequate abstractions using counterexamples.[^10][^11] | Counterexample-driven refinement is an inherited algorithmic principle, not the novelty claim. |

The defensible candidate claim is therefore narrow: **continual discovery of object-bound, history-dependent safety state from intervention evidence, with compositional transfer to new hazard chains and use in continuous VLA execution**. The reviewed sources establish many ingredients separately; this review does not establish that no paper combines them. Before claiming priority, conduct a focused citation-chain review of predictive-state discovery, learned reward machines, and generative predicate invention.

## Formal Target

Let `h` denote observation-action history and let `Y_H` denote the vector of safety outcomes over horizon `H`. For a specified family of future intervention policies `Pi`, define:

```text
d_safe(h, h') = sup_{pi in Pi}
    TV( P(Y_H | h, do(pi)), P(Y_H | h', do(pi)) ).
```

This is a conceptual safety-predictive distance: two histories can share a memory state only when they imply sufficiently similar future safety outcomes under the interventions of interest. Use intervention policies that can react to observations, rather than silently assuming open-loop tests cover all feedback behavior. The outcome vector must include delayed harm; immediate collision cost alone is inadequate.

The learned memory `z = f_M(h)` aims to preserve distinctions with large `d_safe` while compressing irrelevant history. This is inspired by predictive-state and abstraction literature, not a new definition of state in general.[^3][^4] Finite data cannot evaluate the supremum over arbitrary interventions. Operationally, the system estimates disagreement on a bounded, expanding library of diagnostic tests and reports coverage limitations.

Separate two quantities that are often both called memory:

1. **Episode state `z_t`:** uncertain values of currently relevant latent variables, bound to tracked objects and updated after events.
2. **Persistent model `M`:** learned variable definitions, transition mechanisms, diagnostic tests, scope conditions, and evidence accumulated across episodes.

Resetting an episode may reset `z_t`; it must not erase `M`. If physical objects persist between episodes, their beliefs should persist too unless a reset has been observed. A changed environment can invalidate an old mechanism, so retention must include revision rather than an unconditional claim of monotonically improving safety.

## Encoding and Storage

Use a hybrid relational dynamical memory. Each learned mechanism contains:

```text
Mechanism:
  typed argument roles and permissible bindings
  latent state domain: Boolean, categorical, or continuous belief
  observation encoder and recurrent/event update function
  action-conditional hazard predictor
  diagnostic intervention tests
  applicability and out-of-distribution checks
  supporting, contradicting, and censored evidence IDs
  uncertainty, version, and predecessor versions
```

Boolean state is suitable for a persistent exposure flag; temperature and accumulated strain require continuous state and uncertainty. A small learned automaton can represent ordering-sensitive effects. The aim is to learn the appropriate representation within a declared model family, not claim unrestricted ontology discovery.

For the first algorithm, use typed object slots, neural event detectors, and a small grammar of state-update programs. Permit persistence, reset, bounded counting, time decay, and relational propagation. Learn which combinations matter and how they are grounded. This inductive bias must be disclosed: discovering an exposure state from generic persistence and contact operators is a narrower achievement than inventing all relevant physics from pixels.

Store raw clips and sensor traces in an immutable evidence archive; store mechanism metadata, versions, and event references in a structured database; store learned encoder parameters as versioned artifacts. A vector index can suggest similar evidence, but cannot decide whether a safety mechanism is applicable. Preserve the observation uncertainty and tracking provenance, since a confidently wrong object identity can corrupt an otherwise correct mechanism.

Keep a compact persistent regression set of diagnostic history pairs. Retain both sides of each important distinction, rather than only representative failure episodes. Deletion or consolidation is acceptable only after checking that the compressed model still separates the relevant future risks.

## Learning Algorithm

The proposed learning loop has six steps:

1. **Predict and record.** Before execution, record proposed actions, predicted outcome distributions, current memory state, and retrieved mechanisms. After execution, distinguish actual outcomes from assessor judgments and from actions blocked by the controller.
2. **Detect aliasing.** Search for histories mapped to similar memory states whose outcome distributions differ for comparable interventions. Repeated evidence or uncertainty-aware tests are needed: one stochastic outcome difference does not prove a missing variable.
3. **Propose a state refinement.** Use the discrepant histories to generate candidate event detectors and recurrent update programs. A VLM can suggest candidates, but an attractive verbal explanation is not validation.
4. **Discriminate mechanisms.** Select simulator interventions that separate candidate explanations: change the object appearance, remove a contact, vary the delay, insert a cleaning operation, or alter the order of events. Paired simulator branches should share exogenous randomness where possible and use dynamically valid rollouts; arbitrary image edits do not establish physical counterfactuals.
5. **Fit and validate.** Learn grounding and transition parameters; evaluate held-out intervention likelihood, risk discrimination, and complexity. Promote a mechanism only after it improves held-out prediction and preserves earlier distinctions. Keep competing explanations when evidence is insufficient.
6. **Transfer and revise.** Bind the mechanism to new objects by roles, test uncertainty, and update evidence. Split an overly broad mechanism when a genuine exception appears; merge states only when diagnostic outcomes support equivalence.

An illustrative selection objective is:

```text
score(M) = held-out intervention prediction loss
         + lambda * representation complexity
         + beta * regression loss on retained diagnostic pairs.
```

The important experiment is whether structural growth is necessary. Compare against the same architecture with a fixed number of latent states and against a larger recurrent model with the same evidence and compute budget.

A safety intervention censors the outcome of the rejected action. Do not label it as a demonstrated failure or label the successful replacement as evidence for the rejected action. Simulator branches, verified demonstrations, and informative human feedback can help, but their provenance must remain distinct. If there is no observable cue and no distinguishing feedback, the missing state is unidentifiable; the appropriate result is uncertainty, not an invented causal explanation.

## Retrieval and VLA Guidance

Retrieve against **the candidate action and its predicted interaction chain**. A movement toward a clean object should activate mechanisms associated with the moving tool's prior contacts and possible transfer paths, even if the original episode looked different. Use typed role matching and event dependencies to select candidate mechanisms, then score relevance through their predicted effect on action risk.

Top-k semantic retrieval alone is inappropriate for hard safety constraints. Always evaluate known applicable hard monitors; use approximate retrieval for uncertain, expensive mechanism hypotheses. Dependency closure matters: a relevant tool state may depend on a prior contact with another object, whose state depends on an earlier event. Bound this expansion explicitly and measure missed-hazard recall.

Guide the VLA at two timescales:

**Before an action chunk:** sample candidate chunks and evaluate them using the learned safety state and action-conditional hazard model. Include validated recovery or inspection skills as candidates. Choose among candidates satisfying a risk criterion, favoring task progress and low deviation from the nominal policy. If no candidate is acceptable, replan or invoke a context-appropriate fallback. Stopping motion is not universally safe, for example when holding an unstable load.

**During execution:** retain a fast geometric controller for collisions and known motion constraints. A CBF does not enforce contamination or delayed thermal safety merely because those hazards are stored in memory. Those mechanisms need their own grounded action checks, prediction horizons, and recovery logic.

For a higher-risk second stage, train a small memory-conditioned adapter or risk-guidance head inside the VLA action generator. Compare it against candidate reranking before assuming internal guidance is better. Freeze the base VLA during the central causal experiment so memory improvements can be isolated; freezing need not be a final architectural constraint.

Natural-language reminders are a useful baseline and explanation interface. They should not be the sole path from learned hazard state to action selection. The recommendation is explicit action-risk evaluation because it makes the memory's effect measurable and does not depend entirely on instruction compliance.

## Decisive Evaluation

SafeLIBERO can provide familiar manipulation and control infrastructure, but obstacle avoidance alone will not establish the proposed contribution. Add a history-dependent safety suite with paired current observations and independently instrumented simulator outcomes. Do not expose simulator hazard state to the agent except in an explicitly labeled oracle condition.

| Axis | Required experiment |
|---|---|
| Necessity of history | Match current observations while varying a relevant earlier event. |
| State invention | Introduce a hazard mechanism absent from the initial memory vocabulary. |
| Cross-episode accumulation | Retain learned mechanisms while resetting ordinary episode context. |
| Relational transfer | Change identities, appearances, layouts, and task instructions. |
| Compositional transfer | Train short propagation chains; test longer chains and combinations of mechanisms. |
| Correctable conservatism | Include effective resets, harmless lookalikes, and safe exceptions. |
| Continual retention | Revisit earlier mechanisms after learning new ones; include genuine mechanism changes. |
| Action consequences | Measure physical safety outcomes independently of the system's own predictions. |

Use three mechanism families: persistent exposure and transfer, heating/cooling with delayed handling risk, and cumulative load or repeated-contact damage. Begin with one family to establish feasibility, then test whether the method learns more than one hand-designed special case.

Baselines should include current-observation control, full-history recurrent memory, event-keyframe memory, textual/episodic retrieval, fixed-vocabulary temporal rules, a fixed-capacity predictive-state model, a predicate-invention adaptation, and an oracle hazard-state model. Match observations, feedback, simulator query counts, VLA backbone, and recovery access. Distinguish reproduced published systems from adapted baselines.

Report cumulative severity-weighted harm, safe task completion, repeat-hazard rate, unnecessary intervention rate, risk calibration, simulator queries per discovered mechanism, memory size, latency, and performance on past hazards. Plot safety against retained experience and against completed tasks, so refusing everything cannot appear to be successful accumulation.

The key ablations remove structural growth, diagnostic interventions, relational binding, or diagnostic-pair retention. A complementary ablation substitutes oracle perception: if the full system fails but oracle grounding works, the bottleneck is perception rather than the memory principle.

## Theory and Failure Criteria

A tractable theory target is a restricted finite relational system with a realizable update-program grammar, identifiable diagnostic interventions, bounded noise, and a known outcome horizon. Ask whether a counterexample-refinement procedure identifies an adequate safety abstraction with a bounded number of tests. General guarantees for arbitrary VLA histories, open-world hazards, and learned simulator dynamics are not justified.

For a fixed policy family, a uniform total-variation bound of `epsilon` on predicted outcome distributions implies an `epsilon` bound on each represented harm-event probability. This is a conditional consequence of the definition, not the substantial theorem. The substantial work would be proving when finite diagnostic evidence controls that uniform error, including after representation updates and adaptive action selection.

The idea fails as a strong paper if a large recurrent baseline matches it under equal data, if the safety predicates were effectively supplied in the prompt, or if improvements disappear on new relational compositions. It also fails if extra simulator queries explain the entire gain. These are useful early rejection tests, not details to postpone until final evaluation.

The highest-value early experiment is small: can the system discover one previously absent persistent state and its reset, then transfer it to unseen object identities and longer contact chains? If that fails under clean perception and a reliable simulator, scaling to a larger VLA will probably obscure the issue rather than solve it.

## Recommendation

Make **safety-state invention** the central research question. Treat evidence retrieval, logic, world models, and VLA steering as supporting mechanisms with strong baselines. This has greater scientific upside than another safety RAG system because success would demonstrate that accumulated experience changes what the robot knows it must remember, not merely how many warnings it can retrieve.

The practical fallback remains the conditional-contract method in the broader review. Both directions can share an evidence archive and intervention instrumentation, but they should not be presented as one undifferentiated contribution.

## Sources

[^1]: Zhuoran Li, Zhiyang Li, Kaijun Zhou, and Jinyu Gu. [SOMA: Strategic Orchestration and Memory-Augmented System for Vision-Language-Action Model Robustness via In-Context Adaptation](https://arxiv.org/html/2603.24060v1). March 2026 preprint. Full-text method consulted.
[^2]: Ganlin Yang et al. [EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies](https://arxiv.org/abs/2606.20092). June 2026 preprint. Abstract consulted; no independent replication assessed.
[^3]: Michael L. Littman, Richard S. Sutton, and Satinder Singh. [Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). NeurIPS 2001. Primary paper and indexed text consulted.
[^4]: Haohong Lin et al. [Safety-aware Causal Representation for Trustworthy Offline Reinforcement Learning in Autonomous Driving](https://hhlin.info/publication/fusion/). RA-L 2024; author page dated November 2023. Author metadata and [indexed manuscript](https://par.nsf.gov/servlets/purl/10587874) consulted; direct manuscript fetch was unavailable.
[^5]: Xiaoyuan Cheng, Boli Chen, Liz Varga, and Yukun Hu. [Safe Reinforcement Learning in Tensor Reproducing Kernel Hilbert Space](https://arxiv.org/abs/2312.00727). December 2023 preprint. Abstract consulted; its theorem assumptions were not independently audited.
[^6]: [Predicate Invention for Bilevel Planning](https://ojs.aaai.org/index.php/AAAI/article/view/26429). AAAI 2023; preprint March 2022. Publisher abstract and indexed paper consulted.
[^7]: [InterPreT: Interactive Predicate Learning from Language Feedback for Generalizable Task Planning](https://www.roboticsproceedings.org/rss20/p034.pdf). RSS 2024. Proceedings text and [author project](https://interpret-robot.github.io/) consulted.
[^8]: [VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/975db59bfa6eba2175a410f6afcecd99-Abstract-Conference.html). ICLR 2025. Proceedings abstract consulted.
[^9]: Ziyi Yang et al. [SkillWrapper: Generative Predicate Invention for Task-level Planning](https://yzylmc.github.io/skill-wrapper/). Author project and linked manuscript; publication date not established from the inspected page. Method overview consulted; formal guarantees not independently audited.
[^10]: [Counterexample-guided Abstraction Refinement for POMDPs](https://arxiv.org/abs/1701.06209). January 2017 preprint. Abstract consulted.
[^11]: [Automata Learning Through Counterexample-Guided Abstraction Refinement](https://sws.cs.ru.nl/publications/papers/fvaan/CEGAR12/FM.pdf). FM 2012. Primary publication page and indexed paper consulted.
