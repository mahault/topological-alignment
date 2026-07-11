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
