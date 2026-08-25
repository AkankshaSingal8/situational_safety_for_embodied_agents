# Plan: rewrite the Introduction in the style of OGPO + Bajcsy

Deliverable of this document: the plan only. `main.tex` is **not** edited this turn.

## 0. What the reference papers actually do (verified from LaTeX source, not renderings)

Both sources pulled via `arxiv.org/e-print` and read directly, because the HTML/summarizer
path destroys exactly the features at issue (paragraph boundaries, `\textbf{}` leads).

| | OGPO (2605.03065, `arxiv_version/intro_new.tex`) | UNISafe (2505.00779, `_I.Introduction/1.introduction.tex`) | **ours (now)** |
|---|---|---|---|
| intro words | 676 | 459 | **1443** |
| `\textbf{}` / `\paragraph{}` leads | **0** | **1** — and it is the method *name*, not a lead | **7** |
| contributions bullet list | none | none | 5 items |
| numbers in intro | **none** | **none** | ~14 |
| running example | none | Jenga, introduced ¶1, reused ¶3 and in Fig. 1 caption | banana pair, used once |
| explicit key-idea sentence | "there exists a *sample-cost asymmetry*" | "Our key idea is to use the world model's epistemic uncertainty as a proxy for identifying unseen potential hazards." | **absent** |

The advisor flagged the bold-lead recommendation as unsupported by a paraphrase. It is now
supported: `grep -c '\\textbf{\|\\paragraph{'` returns 0 and 1 respectively.

### The move both papers share

Both turn on **one asymmetry sentence** that makes the method feel inevitable rather than assembled:

- OGPO: environment samples are expensive, denoising samples are free → *decouple* the two MDPs.
- UNISafe: OOD detection tells you that you are uncertain but not what to do → *fold uncertainty into the state* so reachability can act on it.

Everything downstream reads as a consequence. Our intro currently has no such sentence, so the
method arrives as a pipeline description (¶7, 146 words) rather than as a consequence of a fact.

---

## 1. Target structure — 6 flowing paragraphs, ~800 words

Cut ~600 words. Drop all 7 bold leads. Keep the contributions list (IROS reviewers skim it and
neither reference is an IROS submission), but tighten to ~90 words.

**¶1 — VLAs are deploying, and they are unauditable.** Keep roughly as-is; it already matches
OGPO's opening register (bare problem statement, no hedging). Trim 113 → ~90 words.

**¶2 — Safety is a property of the scene, and here is the pair that proves it.** *This is our
Jenga.* Merge current ¶2 and ¶3 (244 + 212 = 456 words) down to ~140. Lead with the banana
minimal pair — same object, same grasp, same geometry, hazard in one task and the goal in the
other — and state the consequence in one line: no geometric predicate and no amount of
collision-avoidance training separates them; only the instruction does. Cut the alignment
footnote's argument down to a clause; the 19.6k-demo/4.7–5.3% evidence moves to §III where it
is already made properly.

**¶3 — Classical filters assume the answer to the hard question.** Trim 218 → ~130. Keep the
knife/24-mesh-geom example: it is the second-best concrete thing in the intro and it earns its
words. Keep the Hsu 4-module decomposition framing. **Weaken the universal claim** — see §3.

**¶4 — Why not just ask the policy nicely?** Trim 194 → ~110. State the null and the mechanism;
keep exactly **one** number (AUC 0.998 at the action interface, because it is the surprising
one). "115 cells", "seven rewriters", "0% on ten controls" move to §III. Land "knowing is not
avoiding" as the paragraph's last sentence — it is the title and it should close a paragraph,
not sit mid-sentence in the middle of ¶5.

**¶5 — The asymmetry, then the method.** Rewrite ¶6+¶7 (183 + 146 = 329) to ~150, built around
the key-idea sentence (§2). **Delete the learned-critic negative result from the intro** — it is
a 3-line digression here and it is already §V-E's job. The three-family survey compresses to one
sentence and moves its weight to §II.

**¶6 — Contributions.** 5 items → 5 items, ~90 words. Fold the current items 1 and 4 hint of
overlap; keep the benchmark-defect item, which is a genuine draw for this workshop.

---

## 2. The key-idea sentence — the actual motivation fix

This is the half of the ask that is *motivation*, not style. Target sentence, to be placed at
the head of ¶5:

> A VLM asked *"what in this scene is dangerous?"* answers correctly 47–84% of the time; a VLM
> asked *"is this object referred to by the instruction?"* answers correctly essentially always.
> Our key idea is to exploit that asymmetry: let the VLM answer only local, per-object,
> checkable questions, and move the composition into a fixed three-literal rule small enough to
> read in one line.

Why this is the right sentence:

- It is OGPO's rhetorical shape (an asymmetry that licenses a decoupling) with our content.
- It answers the first question a neuro-symbolic reviewer asks — *why a fixed FOL rule and not
  just a structured VLM output schema?* — before they ask it. The answer is that a schema still
  lets the VLM compose, and §V-E measures composition failing **unsafe**.
- It licenses the two things the paper already spends its experiments defending: the rule being
  *fixed* (§V-E) and every literal being *load-bearing* (§V-D).
- The numbers are already in Fig. 4, so the intro is pointing at a figure rather than
  pre-spending a result.

Follow it with the one-line consequence that connects to §IV: because the rule's output is a
*named entity with a position and an extent*, it is exactly the object the classical filter
always assumed it had been given — so everything downstream is textbook discrete-CBF machinery.
That sentence is what makes the paper feel like one idea instead of two.

---

## 3. AnySafe — a real threat, handled in §II (out-of-intro edit)

`AnySafe` (Agrawal, Seo, Nakamura, Tian, **Bajcsy**, arXiv:2509.19555) is the one paper that
could be read as having already solved our stated gap: it builds latent safety filters that
"adapt to user-specified safety constraints at runtime," which is close enough to our pitch that
a Bajcsy-adjacent reviewer will raise it.

Our distinction is real and defensible from its own abstract: AnySafe adapts **which**
user-supplied constraint *image* is active. A human still names the hazard. We elect the hazard
autonomously from instruction + scene, with no human in the loop at deployment. Runtime
*adaptation* to a given constraint, versus runtime *derivation* of the constraint.

Edits:
- **§II**, 2 sentences next to `nakamura2025latent`/`nakamura2025latentcbf`, plus a `references.bib` entry.
- **§I ¶3**, one clause: family (i) currently says analytic filters "must be handed a safe set"
  as if universal. Soften to note that recent latent filters parameterize the constraint at
  runtime but still require a human to supply it. Costs ~15 words, removes an easy rebuttal.

Also worth citing but **not** in the intro: `2510.06492` (Kim, Nakamura, Bajcsy) names
*estimation gaps* — safety-critical information absent from the latent state — which is our
representation–action gap reached from the world-model direction. It is free corroboration for
**§III**, one sentence. Do not spend intro words on it.

---

## 4. What does NOT change

- The title, abstract, and all section structure past §II.
- All numbers, tables, and figures. This is a rhetoric edit; no claim changes value.
- The `\textbf{}` leads *elsewhere* in the paper (§II, §V). They are conventional in related-work
  and results sections and both references use them there. Only the intro drops them.

## 5. Regression checks after the rewrite (~600 words removed will reflow everything)

1. `fig_teaser` still leads page 2 — it floated to page 3 once already and had to be forced.
2. Table V does not overflow the column — it overflowed once already.
3. Page count still 8 content + 1 references (the 8+N format).
4. All four `\includegraphics` targets still resolve.
5. Rebuild `resai_v2_submission.zip`, re-commit, push to `resai_workshop_paper`.

## 6. Order of work

1. ¶5 first — the key-idea sentence. If it does not land, nothing else matters.
2. ¶2 (banana pair compression), since it is the largest single cut.
3. ¶1, ¶3, ¶4 trims; delete bold leads.
4. §II AnySafe + §III estimation-gap sentences; `references.bib`.
5. Contributions tighten.
6. Regression checks §5.
