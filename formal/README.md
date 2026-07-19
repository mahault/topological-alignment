# Formal verification workspace

This directory contains the machine-checked kernel for the topological-alignment
project. It begins with finite definitions and deliberately small results.

## Installed stack

- Elan 4.2.3
- Lean 4.32.0 selected by Elan through `lean-toolchain`
- Z3 4.16.0 through the Python `z3-solver` package

The initial verified kernel uses Lean's core library. Mathlib is required for the later
metric, measure-theoretic, topological, and probabilistic formalizations. Mathlib was
not installed: two attempted repository checkouts failed with incomplete Git object
transfers on 2026-07-19. The current manifest therefore has no package dependencies,
and the core proofs intentionally avoid importing Mathlib until the checkout can be
retried for those richer definitions.

The commands below passed on 2026-07-19. The Lean source tree also passed a scan for
`sorry` and `admit`.

## Checks

```powershell
cd formal
# A newly opened shell should find `lake`; in the installation shell use:
C:\Users\mahau\.elan\bin\lake.exe build
python counterexamples\empowerment_not_reachability.py
```

## Proof status

| Obligation | Status |
|---|---|
| Relational capability dominance is reflexive | Lean checked |
| Relational capability dominance is transitive | Lean checked |
| Mutual dominance equates the capability coordinates | Lean checked |
| Policy closure implies finite-horizon phenotype viability | Lean checked |
| Recovery plus closure implies post-recovery viability | Lean checked |
| Policy closure alone does not imply perturbation recovery | Lean counterexample theorem |
| Componentwise functional dominance is reflexive and transitive | Lean checked |
| One-environment superiority does not imply robust dominance | Lean counterexample theorem |
| Robust focal benefit does not imply an affected-agent capability floor | Lean counterexample theorem |
| Finite relative moral-goodness predicate and projection lemmas | Lean checked |
| Action influence does not imply arbitrary target reachability | Z3 finite counterexample |
| Misalignment quantity is a metric or pseudometric | Not started |
| Finite phenotype viability theorem | Initial deterministic kernel complete |
| Decorated alignment-map composition | Not started |

Exact assumptions, boundary cases, and interpretation limits are recorded in
[`THEOREM_LEDGER.md`](THEOREM_LEDGER.md).

No declaration containing `sorry` counts as complete. Formal verification certifies
that results follow from their encoded assumptions; it does not certify that those
assumptions accurately model a phenotype, virtue, or moral good.
