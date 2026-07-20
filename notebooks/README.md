# Research notebooks

## `where_we_are`

`where_we_are.ipynb` is the canonical claim-by-claim visual guide to the research
programme. Its standalone, code-hidden rendering is `where_we_are.html`.

The guide is also the GitHub Pages homepage. It includes the definition decision
procedure, rejected shortcuts, philosophy-to-model mapping, experiment genealogy,
stage-by-stage setups, and the interpretation of the mechanism ablations for virtue.
Display equations use `$$...$$` delimiters so Markdown preserves the TeX source for
the static-math export step.

The notebook is generated from `build_where_we_are.py`, then executed so every table,
graph, and animation is embedded in both artifacts. Numerical evidence is read from
the current committed benchmark ledgers. Historical, archived, and
non-acceptance-ready results are omitted from the public guide. Pedagogical animations
are explicitly labelled and must not be cited as additional evidence.

Rebuild from the repository root:

```powershell
python -B notebooks/build_where_we_are.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/where_we_are.ipynb --ExecutePreprocessor.timeout=300
python -m jupyter nbconvert --to html notebooks/where_we_are.ipynb --output where_we_are.html --output-dir notebooks --no-input
python -B notebooks/render_static_math.py notebooks/where_we_are.html
```

The final step converts every display and table equation to literal inline SVG markup. The
notebook retains editable TeX, while the standalone HTML renders mathematics without
MathJax, data-URI images, a CDN, JavaScript execution, or network access.

`RESEARCH_DASHBOARD.ipynb` and its HTML rendering are the shorter dashboard view.
