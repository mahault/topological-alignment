# Research notebooks

## `where_we_are`

`where_we_are.ipynb` is the canonical claim-by-claim visual guide to the research
programme. Its standalone, code-hidden rendering is `where_we_are.html`.

The notebook is generated from `build_where_we_are.py`, then executed so every table,
graph, and animation is embedded in both artifacts. Numerical evidence is read from
the committed benchmark ledgers; pedagogical animations are explicitly labelled and
must not be cited as additional evidence.

Rebuild from the repository root:

```powershell
python -B notebooks/build_where_we_are.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/where_we_are.ipynb --ExecutePreprocessor.timeout=300
python -m jupyter nbconvert --to html notebooks/where_we_are.ipynb --output where_we_are.html --output-dir notebooks --no-input
```

`RESEARCH_DASHBOARD.ipynb` and its HTML rendering are the shorter dashboard view.
