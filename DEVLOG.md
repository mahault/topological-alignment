# Devlog

## 2026-06-26
- Added an extended-abstract draft (uncommitted, work in progress)

## 2026-06-27
- (uncommitted, work in progress) Continued drafting the extended abstract

## 2026-06-29
- Added an extended abstract (tex). (uncommitted)

## 2026-06-30
- Drafted an extended abstract (tex/extended_abstract.tex). (uncommitted, work in progress)

## 2026-07-10
- Call with Jamie Duell: use the MIG (ICML 2024, Appendix A) energy-minimization
  geodesic scheme with the metric tensor G replaced by the (empirical) Fisher
  information; compare agents' posterior belief manifolds with Gromov-Wasserstein
  (no global manifold correspondence needed, endpoint correspondence only).
- Implemented `experiments/geodesics.py`: closed-form + empirical Fisher metrics,
  MIG-style geodesic solver, self-contained entropic Gromov-Wasserstein
  (validated against POT, ~13x faster at our cloud sizes), per-point GW
  misalignment profile for localizing structural disagreement.
- Added `experiments/exp4_geodesic_alignment.py` (H-G1..H-G4: GW separation in
  private frames vs KL baseline, Fisher geodesics bowing through uncertainty +
  rigid/flexible transition-cost asymmetry, divergence localization along
  matched geodesics, sliding-window GW convergence tracking) and wired
  `--exp4` into scripts/run_all.py.

## 2026-07-19
- Kicked off a virtue-ethics x active-inference research framework (11 commits, ~780
  lines): moral predicates grounded in shared semantics, with semantics and cooperation
  derived from free-energy dynamics rather than assumed; uncertainty modeled in
  standing and capability floors.
- Built an adversarial moral proxy benchmark with finite proxy systems, plus a finite
  probabilistic active-inference counterexample.
- Adversarial audit passes: unsupported moral-inference constructs replaced, free-energy
  semantics and cooperation claims audited, "goodness" reframed and the research
  pipelines audited alongside.
- Added a V0 felt-goodness measurement pilot (faebe6e).

## 2026-07-20
- Reproduced the V2/V3 selective-stability signature across three environment seeds
  for 41 of 48 sampled parameter configurations.
- Added mechanism ablations. Slow-centre learning, evidence-precision gating,
  context-sensitive semantics, and goodness-grounding were necessary in the tested
  construction; the explicit attraction term and EFE policy mapping were not.
- Documented how `shared-protention-alignment`, `externalization-horizon`,
  `empathy-prisonner-dilemma`, `group-formation`, `aif-meta-cogames`, and
  `Active_Inference_Social_Lock_In` feed the proposed V4 experiment.
- Added executed notebook and standalone HTML dashboards explaining the theory,
  evidence ladder, robustness results, claim boundaries, and next experiment.
- Added a GitHub Pages deployment that publishes only the two self-contained visual
  dashboards for stable collaborator-facing links.
