# Overleaf Project

Import `overleaf_safety_memory.zip` as a new Overleaf project. Set `main.tex` as the main document and select pdfLaTeX. The figures are editable TikZ sources; no external images, shell escape, or bibliography service is required.

The document contains the architecture, learning loop, image/instruction workflow, concrete memory records, action-selection arithmetic, research boundaries, and references. The camera panels are explicitly schematic, not recorded robot images.

Local build: run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice in this directory.

Example check: run `python3 example_replay.py`. This standard-library-only replay tests supplied memory and risk calculations. It does not implement perception, state invention, VLA inference, or physical safety guarantees.

The package is ready to import but has not been uploaded to an Overleaf account.
