# Safety memory design — 11 October 2026

Import `../overleaf_safety_memory_2026_10_11.zip` using **Overleaf → New Project → Upload Project**. Select `main.tex` as the main document and pdfLaTeX as the compiler.

The compiled proposal is `main.pdf`. It contains the recommendation, architecture, current-code integration, literature comparison, predicate/rule learning procedure, worked temporal-memory example, failure analysis and revised design, PSC capacity/budget, and falsifiable experiments. This is a design proposal; no runtime implementation or research evaluation was performed.

- `figures/system.tex`: editable TikZ architecture.
- `figure.tex` / `figure.pdf`: standalone figure source and compiled export.
- `references.tex`: linked bibliography, without BibTeX/Biber dependencies.
- `sources.json`: primary-source metadata and access dates.
- `environment_notes.md`: inspected repository and read-only PSC checks.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error figure.tex
```

No shell escape or external figure downloads are required. The document was compiled locally and the architecture render was visually checked. Bibliography references resolve, and the final build has no overfull boxes or LaTeX warnings. Ordinary underfull-box notices can occur in narrow table columns.

The September proposal in `../overleaf_safety_memory/` is preserved. This dated revision is an alternative simplified design, not evidence that the prior proposal has been implemented.

No Overleaf connector was available; the plugin directory search returned no Overleaf integration. This package has not been uploaded to an Overleaf account.

Revision: added offline learning and runtime pseudocode, predicate motivation, restricted first-order schema motivation, and a direct response to the predicate-only objection using Brunke et al. (RA-L 2025). The matched comparison permits direct relational mappings and a shared pair risk head. FOL necessity is not claimed. Applied the humanizer and no-ai-slop editing guidance.

Added Section 1.1, a ten-step natural-language pipeline covering camera inputs, RGB-D geometry, object tracking, predicates and memory, VLA actions, rule checks, execution, evidence collection, offline learning, and reuse.
