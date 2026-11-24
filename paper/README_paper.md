# Paper build

All numbers in the paper are produced from the experiment outputs in `results/`.

1. Run the experiments (training, evaluation, pedagogy) so that `results/` is filled.
2. Run the figure script so that the result figures exist (`kt_figures/` or `~/Downloads/kt_figures/`).
3. Build the paper:

```
.venv/bin/python paper/build_paper.py
```

The script regenerates every table in `paper/tables/`, recomputes every result number quoted in the
text of `main.tex` and prints PASS/FAIL for each, copies the result figures, and compiles `main.pdf`
(`pdflatex` + `bibtex`). References are in `paper/refs.bib`.
