# Theorem Ledger

## V1 -- Policy closure implies finite-horizon viability

**Lean declaration:** `viableFor_of_closedUnderPolicy`
**Status:** machine-checked theorem
**Source:** `TopologicalAlignment/Viability.lean`

### Definitions

- `PolicySystem`: a deterministic transition function and state-dependent policy.
- `Phenotype.viable`: a declared predicate identifying phenotype-relative viable
  states.
- `ViableFor P S h x`: the initial state and each of the next `h` closed-loop states
  satisfy `P.viable`.
- `ClosedUnderPolicy P S`: every viable state has a viable closed-loop successor.

### Exact claim

For arbitrary state and action types, if a phenotype's viability predicate is closed
under a policy-controlled transition system, every initially viable state remains
viable for every finite horizon.

### Assumptions

1. Dynamics and policy are deterministic and total.
2. The state contains everything required for the Markov transition.
3. The viability predicate is fixed during the horizon.
4. Closure holds for every viable state, not merely observed or sampled states.
5. The initial state is viable.

### Proof and checking

Induction on the horizon. The zero case is initial viability. In the successor case,
closure gives viability of the next state and the induction hypothesis supplies the
remaining horizon. Checked by Lean 4.32.0 without `sorry`, `admit`, or additional
axioms declared by this project.

### Boundary and degenerate cases

- Horizon zero asserts only current viability.
- An empty viability region makes closure vacuously true, but the theorem still
  requires an initially viable state and therefore yields no instance.
- A universal viability predicate makes the result trivial; empirical work must rule
  out an uninformative phenotype model.
- The result does not cover stochastic dynamics, policy uncertainty, perturbations,
  model error, recovery after leaving the region, or changing phenotypes.

### Counterexample obligation

If closure is weakened to hold only on sampled states, an unobserved viable state may
transition outside the region. If closure is removed, a one-step transition from a
viable to a non-viable state refutes the conclusion at horizon one.

### Honest interpretation

This is a conditional theorem about functional persistence relative to a declared
phenotype. It does not show that the predicate is empirically valid, that persistence
is flourishing, that the policy is flexible or virtuous, or that its effects on other
agents are morally admissible.

### Empirical requirements

- independent operationalization and validation of the phenotype's viability region;
- tests of state sufficiency and transition-model error;
- perturbations beyond the environments used to construct the predicate; and
- measurement of costs and capability losses imposed on affected others.

## V2 -- Recovery plus closure implies continued viability

**Lean declaration:** `viableAfterRecovery`
**Status:** machine-checked theorem
**Source:** `TopologicalAlignment/Viability.lean`

If the closed-loop trajectory is viable at a declared recovery time and the viability
region is closed under the policy, it remains viable for every declared finite future
horizon. The proof composes the recovery certificate with V1.

This theorem does not establish that recovery occurs, that the recovery time is
optimal, or that the intervening non-viable trajectory is harmless. Exact-time
recovery is used at this stage; stochastic and interval-censored recovery remain
future extensions.

## V3 -- Policy closure does not imply perturbation recovery

**Lean declaration:** `closure_does_not_imply_recovery`
**Status:** machine-checked existential counterexample
**Source:** `TopologicalAlignment/Viability.lean`

The witness has a viable absorbing state and a non-viable absorbing state. The policy
preserves every viable state, but a declared perturbation moves the system into the
non-viable state, from which it never recovers at any finite time.

This establishes a strict conceptual separation between ordinary invariance and
resilience. Consequently, evidence of stable functioning in an unperturbed setting
cannot by itself support the stronger claim that a phenotype or virtue regime is
adaptively metastable.

## D1 -- Componentwise functional dominance is a preorder

**Lean declarations:** `outcomeDominates_refl`, `outcomeDominates_trans`,
`robustlyDominates_refl`, `robustlyDominates_trans`
**Status:** machine-checked theorems
**Source:** `TopologicalAlignment/RobustDominance.lean`

The outcome vector contains viability margin, available options, epistemic fit, and
recovery cost. Dominance requires every benefit coordinate to be no lower and
recovery cost to be no higher. No scalar weights or compensation rates are assumed.
The relation is reflexive and transitive both within one environment and pointwise
across an environment family.

The choice and measurement of coordinates remain empirical and normative questions.
The result does not establish completeness: two regimes may be incomparable.

## D2 -- Local superiority does not imply robust dominance

**Lean declaration:** `one_environment_dominance_not_robust`
**Status:** machine-checked existential counterexample
**Source:** `TopologicalAlignment/RobustDominance.lean`

Two regimes reverse their ordering across two environments. A regime dominates in
the selected environment but fails pointwise robust dominance. This justifies
declaring the environment class in advance and testing out of context.

## R1 -- Focal robust benefit does not imply non-externalization

**Lean declaration:** `focal_robust_dominance_does_not_imply_nonexternalization`
**Status:** machine-checked existential counterexample
**Source:** `TopologicalAlignment/RelationalViability.lean`

A regime componentwise dominates its alternative for the focal phenotype in every
declared environment while reducing an affected other's remaining capability below a
hard floor. Therefore robust phenotype-relative functional goodness does not entail
relational admissibility or moral goodness.

`AdmissiblyDominates` records the stronger conditional requirement: focal robust
dominance plus an affected-agent capability floor. Calling that requirement morally
authoritative still requires the Phase 2 bridge argument.

## M1 -- Finite relative moral-goodness predicate

**Lean definition:** `MorallyGoodRelativeTo`
**Lean declarations:** `morallyGood_implies_admissible`,
`morallyGood_implies_plural_improvement`,
`morallyGood_implies_procedural_conditions`
**Status:** definition plus machine-checked projection theorems
**Source:** `TopologicalAlignment/MoralGoodness.lean`

Relative to a declared outcome interpretation, environment family, capability floor,
normative procedure, candidate regime, and baseline, positive moral goodness requires:

1. robust focal dominance and the affected-agent floor;
2. no reduction in the affected capability in any declared environment;
3. at least one strict functional or relational improvement;
4. justifiability;
5. contestability;
6. epistemic responsiveness; and
7. repairability.

Lean verifies that the conjunction entails each grouped condition. This proves that
the encoded definition is internally transparent; it does not prove that the outcome
coordinates, floor, baseline, affected-agent representation, or procedural predicates
are morally correct. The current kernel represents one focal phenotype and one
affected-capability coordinate. The multi-agent probabilistic generalization is the
next formal obligation.
