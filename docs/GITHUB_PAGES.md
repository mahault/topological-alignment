# Public research dashboard

The public GitHub Pages site is available at:

- `https://mahault.github.io/topological-alignment/` — canonical detailed,
  claim-by-claim visual guide;
- `https://mahault.github.io/topological-alignment/where-we-are.html` — stable alias
  for the same detailed guide;
- `https://mahault.github.io/topological-alignment/summary.html` — shorter
  presentation-oriented research dashboard.

## Deployment source

The workflow at `.github/workflows/pages.yml` creates a deliberately narrow Pages
artifact:

| Repository source | Public path |
|---|---|
| `notebooks/where_we_are.html` | `/index.html` and `/where-we-are.html` |
| `notebooks/RESEARCH_DASHBOARD.html` | `/summary.html` |

No datasets, source code, local settings, or other repository files are copied into
the web artifact. Both HTML files are self-contained exports with embedded plots;
the detailed guide also embeds offline-playable pedagogical animations. Equations are
pre-rendered as embedded SVGs, so they display without MathJax or network access.

The deployment runs when either HTML export or the workflow changes on `main` or
`virtue-pragmatics-active-inference`. It can also be started manually from the GitHub
Actions interface.

## Updating the site

1. Edit `notebooks/build_where_we_are.py` for changes to the canonical detailed guide.
2. Rebuild and execute the notebook.
3. Export it to `notebooks/where_we_are.html` with inputs hidden.
4. Run `notebooks/render_static_math.py` on the HTML export.
5. Commit and push the builder, notebook, and HTML export.
6. Confirm the `Deploy research dashboard to GitHub Pages` workflow succeeds.

The public URL is stable across deployments, so collaborator links do not need to be
changed when the analysis is updated.
