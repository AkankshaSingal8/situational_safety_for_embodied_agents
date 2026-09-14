# Language-guided continual safety proposal

Upload `main.tex` to a blank Overleaf project and select **pdfLaTeX**. The document is self-contained: references and the system diagram are embedded, with no external images, bibliography files, or shell-escape requirement.

The accompanying ZIP contains this source and README. A compiled PDF is provided separately.

To compile locally, run twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The proposal recommends SmolVLA-450M, custom ManiSkill3/Panda tasks, and a local Qwen3-VL-4B-Instruct proposer. It specifies an early competence test, alternative policies, training and evaluation boundaries, sequential updates, and estimated experiment budgets. The recommendation has not been validated by running the models. Budgets are planning assumptions.

Primary sources were checked on September 13, 2026. Record exact model and dependency revisions before implementation; the cited documentation can change.

The prose was reviewed with the no-ai-slop and humanizer skills. The editing pass removed unsupported novelty language, separated model facts from proposed settings, and kept the limitations beside the relevant design choices.
