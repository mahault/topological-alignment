# Mathematical Proof Validation

## Policy, tooling, and acceptance gates

Mathematical claims in this project must be validated at three distinct levels.
Numerical evidence, symbolic manipulation, and formal proof have different roles and
must not be presented as interchangeable.

## 1. Tool responsibilities

| Tool | Accepted use | What it cannot establish |
|---|---|---|
| NumPy, SciPy, JAX, PyTorch | simulation, optimization, sensitivity analysis, counterexample search | a universal theorem |
| SymPy | symbolic identities, derivatives, algebraic simplification, small exact cases | correctness of a general proof without independently checked assumptions |
| Z3 | finite and bounded logical constraints, satisfiability, counterexample search | general results in stochastic analysis, topology, or differential geometry |
| Lean 4 + Mathlib | machine-checked definitions, lemmas, and theorems | that formal assumptions accurately describe the empirical world |

## 2. Required theorem record

Every theorem candidate must have a ledger entry containing:

1. exact definitions;
2. typed assumptions;
3. theorem statement;
4. conventional proof or derivation;
5. cited external results;
6. boundary and degenerate cases;
7. attempted counterexamples;
8. numerical or finite-model tests where applicable;
9. Lean verification status;
10. empirical evidence required to justify the assumptions; and
11. the weakest honest label: definition, assumption, hypothesis, conjecture,
    proposition, or theorem.

## 3. Acceptance gate

A claim may be called a **theorem** in the manuscript only when:

- every object in the statement is well-defined;
- existence and finiteness assumptions are explicit;
- the conventional derivation has survived independent review;
- adversarial finite and numerical tests have found no counterexample;
- all imported results are cited with their hypotheses checked;
- the result has a machine-checked Lean proof where the libraries make this
  reasonably feasible; and
- the manuscript distinguishes the theorem from any empirical claim that its
  assumptions hold.

Claims that do not pass this gate must be labeled more weakly.

## 4. Adversarial checklist

For every proposed result ask:

- Is the state space complete, compact, measurable, or separable where the proof
  requires it?
- Does the dynamical system possess the claimed solution, invariant measure, or
  attractor?
- Are stochastic and deterministic flows being conflated?
- Is every supremum, infimum, integral, and norm finite?
- Is a claimed metric actually a metric rather than a pseudometric or divergence?
- Does identity of indiscernibles hold only up to an equivalence relation?
- Are mappings defined on shared spaces, or is an unstated correspondence required?
- Does mutual information imply only channel capacity, or is an unjustified geometric
  reachability claim being added?
- Are empirical estimators being confused with the mathematical object estimated?
- Does the result survive zero-noise, singular, disconnected, nonstationary, and
  degenerate cases?
- Can the theorem be made trivially true by definitions chosen from the desired
  conclusion?

## 5. Validation workflow

```text
informal claim
    -> definitions and typed assumptions
        -> finite examples and edge cases
            -> Z3/SymPy/numerical counterexample search
                -> handwritten proof
                    -> Lean formalization
                        -> independent review
                            -> manuscript theorem
```

Failure at any step sends the claim back for revision or relabeling.

## 6. Initial formalization priorities

### P0. Determine whether misalignment is a metric

Check non-negativity, symmetry, identity of indiscernibles, and triangle inequality.
If equality identifies only observational equivalence classes, define the object as a
pseudometric and construct the quotient explicitly.

### P1. Finite phenotype viability

For a finite transition system, state and prove conditions under which a policy keeps
a phenotype inside a viability set for a finite horizon with a specified probability.

### P2. Empowerment counterexamples

Machine-check finite systems showing that high individual empowerment does not imply
corrigibility, non-domination, or reachability of an arbitrarily selected target.

### P3. Relational-empowerment order properties

Define the partial order induced by self-, other-, and joint capability floors. Prove
basic reflexivity, transitivity, and Pareto-dominance results without forcing the
profile into an unjustified scalar.

### P4. Decorated alignment maps

Define identity and composition for maps preserving selected semantic labels,
transition direction, relational effects, and uncertainty. Prove closure and identify
which properties fail under approximate maps.

### P5. Robust phenotype dominance

For a finite declared environment class, prove conditional dominance results without
claiming that the selected flourishing criteria are thereby morally justified.

## 7. Repository layout

```text
formal/
  lakefile.toml
  lean-toolchain
  README.md
  TopologicalAlignment.lean
  TopologicalAlignment/
    Definitions.lean
    Misalignment.lean
    Viability.lean
    Empowerment.lean
    DecoratedAlignment.lean
```

The first formal milestone is not a proof of the full paradigm. It is a small verified
kernel of definitions, counterexamples, and finite theorems on which later continuous
and stochastic work can safely build.

## 8. Reproducibility

The repository pins the Lean toolchain in `formal/lean-toolchain` and Mathlib in the
Lake configuration. Formal checks should run with:

```powershell
cd formal
lake build
```

Z3 and numerical counterexample tests should be exposed through ordinary test scripts
and added to continuous integration once their first proof obligation is implemented.

Tool versions, formal coverage, admitted axioms, and unfinished proofs must be visible
in `formal/README.md`. No use of `sorry` may be counted as a completed proof.
