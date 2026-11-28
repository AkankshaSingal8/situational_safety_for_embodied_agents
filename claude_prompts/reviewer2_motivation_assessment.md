# Is the SafeLIBERO residual method worth publishing? A Reviewer 2 assessment

This is an honest review of the learned safety residual on frozen pi0.5: what problem it addresses, what the literature already does, how we differ, and whether our current results support a paper. Numbers for our method come from `safelibero_residual_full_report.md` in this folder. Literature numbers come from the cited papers.

## The four questions, answered briefly

**1. What is the problem?**
VLAs such as pi0.5 finish tabletop tasks but drive the arm into obstacles in their path. On SafeLIBERO, pi0.5 avoids collisions in about 19% of episodes ([VLSA/AEGIS](https://arxiv.org/html/2512.11891v1)), and about 3-29% per row in our own Table 1. The problem is real and well posed, and it has a public benchmark.

**2. How has current research tried to solve it?**
Three lines:
- Training-free safety filters on a frozen VLA. AEGIS combines a VLM, GroundingDINO, depth and a control barrier function (CBF) filter, and reports CAR 18.7 → 77.9 and TSR 50.9 → 68.1 with pi0.5 ([AEGIS](https://arxiv.org/html/2512.11891v1)). An attention-guided CBF filter reads the target object from the VLA's own attention heads ([arXiv 2606.09749](https://arxiv.org/abs/2606.09749)).
- Constrained RL fine-tuning of the VLA itself ([SafeVLA, NeurIPS 2025](https://arxiv.org/html/2503.03480v3)).
- Learned residual or recovery modules on frozen VLAs. These target task failure, not safety ([ReCoVLA](https://arxiv.org/pdf/2606.09630), [FIND](https://arxiv.org/pdf/2609.32069)).

**3. How are we different?**
We distill a simulator-lookahead safety shield, which uses privileged simulator state, into a learned, language-conditioned residual on top of frozen pi0.5. At run time it needs only the two RGB images and a sentence naming the obstacle: no depth, no detector, no VLM call and no planner.

**4. Is it useful? Is it better than other methods?**
Not with the current evidence.
- On the same benchmark and base policy, AEGIS reports a larger gain: CAR +59 vs our about +33 on Table 1, and TSR +17 vs our about +4. AEGIS also needs no training.
- Our gains shrink where they matter most for a generalization claim:
  - Novel objects (Table 2): mean SR gain about +5, with 3-5 of 8 rows failing.
  - New obstacle positions (Table 3): Object I and II and Goal I fail.
- The decisive test is the hard novel objects (Table 2b), 10 test-split objects that pi0.5 hits most of the time. Our residuals raise CAR by only about 5-15 points there, against 30-60 on Table 1. The residual avoids obstacles it was trained on and mostly fails to steer around unfamiliar ones.

## What Reviewer 2 would write

1. **No head-to-head baseline.** The paper never runs AEGIS or any CBF filter on our own tables, even though `vlsa-aegis` sits in this repo. The published numbers suggest the baseline wins on SafeLIBERO.
2. **Privileged labels.** The labels come from a shield that sees simulator state, so the method is imitation of an oracle. The question becomes why we wouldn't run a perception-based version of that filter at test time instead.
3. **The obstacle is named for us.** The "true sentence" arm is told which obstacle to avoid. AEGIS finds it with a VLM. The method either has to infer the obstacle itself, or count the VLM that names it in its cost.
4. **Weak transfer.** Table 2b shows the gain comes mostly from familiar obstacles. More data and rescaled training objects did not fix this (ex12 did worse than ex11 on Table 2).
5. **Task success is fragile.** On rows where pi0.5 already succeeds, the residual causes stalls, and task success drops by about as much as collisions improve (Table 3 Object II: TSR 97 → 88.5). A second rollout often succeeds, so the policy is unstable near obstacles.
6. **Metric choice.** We report safe success (SR); AEGIS reports CAR and TSR separately. That's fair to us only if we also measure AEGIS's SR on the same episodes.

## Can the method be motivated honestly?

Only on cost and sensing, and only if new results back it up. The one angle the literature leaves open is a single learned forward pass at control rate, with no depth, no detector and no VLM. AEGIS's VLM query is slow enough that it runs once per episode, which also leaves it unable to track moving obstacles ([arXiv 2606.09749](https://arxiv.org/abs/2606.09749)). To turn that angle into a claim we would need:

1. **AEGIS on our exact Tables 1-3,** reporting TSR, CAR and SR on the same episodes.
2. **Latency and compute** per control step for both methods.
3. **A setting where geometric filters break** and ours does not. Candidates: no depth, heavy clutter or occlusion, obstacles that are hard to segment, or moving obstacles.
4. **No oracle obstacle name.** Either the residual infers what to avoid, or a VLM supplies the sentence and its cost is counted.

If we match AEGIS's safety at a fraction of its cost, or beat it where perception-based filters fail, there is a paper. If not, the honest framing is a negative result: a learned residual distilled from a privileged shield does not transfer to unfamiliar obstacles, and Table 2b quantifies that.

## Bottom line

As it stands, the method is dominated on SafeLIBERO by a training-free baseline, and its own results show weak transfer to unfamiliar objects. The open questions are runtime cost and robustness where geometric filters fail. Neither has been measured yet.

## Sources

- [VLSA/AEGIS and the SafeLIBERO benchmark (arXiv 2512.11891)](https://arxiv.org/html/2512.11891v1)
- [SafeLIBERO dataset](https://huggingface.co/datasets/THURCSCT/SafeLIBERO/blob/main/README.md)
- [Attention-guided safety filter for VLAs (arXiv 2606.09749)](https://arxiv.org/abs/2606.09749)
- [SafeVLA (NeurIPS 2025)](https://arxiv.org/html/2503.03480v3)
- [ReCoVLA](https://arxiv.org/pdf/2606.09630)
- [FIND: real-world RL for self-improving VLAs](https://arxiv.org/pdf/2609.32069)
- [Neuro-symbolic constrained flow matching (arXiv 2607.01378)](https://arxiv.org/pdf/2607.01378)
- [LIBERO-Safety (arXiv 2606.23686)](https://arxiv.org/pdf/2606.23686)
