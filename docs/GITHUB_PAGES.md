# Public research dashboard

The visual research dashboard is published with GitHub Pages at:

- `https://mahault.github.io/topological-alignment/` — presentation-oriented
  research dashboard;
- `https://mahault.github.io/topological-alignment/where-we-are.html` — detailed
  executable visual guide.

## Deployment source

The workflow at `.github/workflows/pages.yml` creates a deliberately narrow Pages
artifact:

| Repository source | Public path |
|---|---|
| `notebooks/RESEARCH_DASHBOARD.html` | `/index.html` |
| `notebooks/where_we_are.html` | `/where-we-are.html` (claim-by-claim guide with embedded animations) |

No datasets, source code, local settings, or other repository files are copied into
the web artifact. Both HTML files are self-contained exports with embedded plots;
the detailed guide also embeds two offline-playable pedagogical animations.

The deployment runs when either HTML export or the workflow changes on `main` or
`virtue-pragmatics-active-inference`. It can also be started manually from the GitHub
Actions interface.

## Updating the site

1. Execute the relevant notebook.
2. Export it to its existing HTML path in `notebooks/`.
3. Commit and push the notebook and HTML export.
4. Confirm the `Deploy research dashboard to GitHub Pages` workflow succeeds.

The public URL is stable across deployments, so collaborator links do not need to be
changed when the analysis is updated.
