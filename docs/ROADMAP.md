# Research and Verification Roadmap

## Purpose

This is the executable workboard for the paradigm. The detailed questions live in
`PARADIGM_PROOF_OBLIGATIONS.md`; this file orders the work, names its artifacts, and
defines when each milestone is complete. A milestone passing its formal gate does not
establish its empirical assumptions or its normative bridge principles.

## Status key

- **Complete**: artifact exists and its stated checks pass.
- **In progress**: currently being implemented.
- **Planned**: scoped but not started.
- **Blocked**: cannot proceed without a named dependency or decision.

## Phase 0 -- Foundations and claim discipline

| Milestone | Status | Exit criterion | Artifact |
|---|---|---|---|
| Literature and novelty audit | Complete | hypotheses compared with adjacent fields and weakened where needed | `VIRTUE_ACTIVE_INFERENCE_LITERATURE_REVIEW.md` |
| Proof-obligation map | Complete | conceptual, normative, mathematical, and empirical questions separated | `PARADIGM_PROOF_OBLIGATIONS.md` |
| Proof-validation gate | Complete | theorem ledger and machine-check policy specified | `MATHEMATICAL_PROOF_VALIDATION.md` |
| Formal toolchain | Complete | `lake build` passes; Z3 countermodel runs | `formal/` |

## Phase 1 -- Finite functional-good kernel

**Objective:** establish exactly what can be concluded about phenotype-relative
functional goodness before attempting the bridge to moral goodness.

| Milestone | Status | Exit criterion | Artifact |
|---|---|---|---|
| 1.1 Finite viability definitions | Complete | phenotype viability set, policy dynamics, and finite-horizon viability are typed | `formal/TopologicalAlignment/Viability.lean` |
| 1.2 Invariance theorem | Complete | Lean proves closure under a policy implies viability for every finite horizon | same |
| 1.3 Perturbation and recovery | Complete | exact recovery bound, post-recovery viability, and failure of closure to imply recovery are Lean-checked | same plus theorem ledger |
| 1.4 Robust dominance | Complete | componentwise dominance is proved reflexive and transitive; one-environment success is shown insufficient | `RobustDominance.lean` |
| 1.5 Non-externalization | Complete | Lean counterexample separates robust focal benefit from affected-agent capability floors | `RelationalViability.lean` |

**Gate 1: passed for the finite deterministic kernel.** We may say “conditionally good for phenotype P” only when phenotype,
environment class, horizon, viability constraints, perturbations, and effects on
others are explicit. We may not yet say “morally good.”

## Phase 2 -- Normative bridge

**Objective:** defend, rather than mathematically smuggle in, the passage from
functional goods to moral standing and admissibility.

| Milestone | Status | Exit criterion |
|---|---|---|
| 2.1 Standing argument | In progress | P1-P6 vulnerability-and-justifiability argument drafted; comparative and adversarial review remains | `NORMATIVE_BRIDGE.md` |
| 2.2 Admissibility constraints | In progress | capability-floor layers, voice, contestability, repair, and non-domination specified; justification remains open | same |
| 2.3 Conflict procedure | In progress | ordered non-scalar procedure drafted; comparative defense remains | same |
| 2.4 Counterexample dossier | In progress | eight hard-case families analyzed; scenario construction and adversarial review remain | same |

**Gate 2:** moral-goodness claims require an explicit bridge principle plus public
justification; neither simulation nor Lean can establish that principle by itself.

## Phase 3 -- Virtue as context-transformed regulation

**Objective:** test whether an abstract virtue is realized by different situated
policies that preserve a deeper regulatory organization.

| Milestone | Status | Exit criterion |
|---|---|---|
| 3.1 Construct validation | Planned | scenarios preserve virtue identity while varying appropriate action |
| 3.2 Pilot and parameter recovery | Planned | competing models are identifiable on simulated and pilot data |
| 3.3 Preregistered Experiment 1 | Planned | protocol, exclusions, outcomes, and model comparison frozen in advance |
| 3.4 Held-out context test | Planned | transformed-schema model beats trait-only and situation-only baselines out of domain |

Primary protocol: `FIRST_EXPERIMENT_REGULATORY_INVARIANCE.md`.

## Phase 4 -- Active-inference mechanism

**Objective:** determine whether active inference adds explanatory and predictive
content rather than redescribing flexible behaviour.

1. **In progress:** specify mappings from virtue realization to salience, precision,
   temporal depth, stakeholder representation, information seeking, and policy
   selection. The initial mathematical specification is in
   `ACTIVE_INFERENCE_NORMATIVE_FORMALIZATION.md`.
2. **Finite predicate complete:** `formal/TopologicalAlignment/MoralGoodness.lean`
   encodes relative moral goodness as admissibility, non-externalized strict plural
   improvement, and procedural legitimacy. The probabilistic multi-agent semantics
   remain planned.
3. **Translation ledger complete:** each moral category now has a normative meaning,
   candidate active-inference object, operational test, non-equivalence warning, and
   epistemic status in `MORAL_GOODNESS_ACTIVE_INFERENCE_TRANSLATION.md`.
4. Establish parameter recovery and compare against simpler reinforcement-learning,
   trait-by-situation, and heuristic models.
5. Test practical wisdom as model and precision governance in expert/novice tasks.

**Gate 4:** retain active inference only if its parameters are recoverable and it
improves held-out prediction or intervention response.

## Phase 5 -- Decorated topological alignment

**Objective:** test whether topology contributes after semantic, causal, relational,
and uncertainty information is retained.

1. Determine whether the proposed misalignment object is a metric, pseudometric, or
   merely a divergence.
2. Formalize identity and composition for decorated alignment maps.
3. Build adversarial pairs: same topology/opposed meaning and different
   topology/equivalent function.
4. Test cross-agent and human-AI generalization.

**Gate 5:** topology remains in the paradigm only if decorated representations beat
semantic and ordinary dynamical baselines on held-out adversarial cases.

## Immediate work queue

1. Draft the moral-standing bridge argument; do not encode it as though it were a
   mathematical consequence.
2. Assemble hard cases involving domination, adaptive preference, paternalism,
   sacrifice, emergencies, and conflicts among genuine goods.
3. Specify which capability floors are universal, phenotype-relative, or
   institutionally contestable.
4. Extend the finite kernel to stochastic transitions only after the normative
   distinctions are stable.
5. Prepare construct-validation materials for Experiment 0.

## Reproducible checks

```powershell
cd formal
lake build
python counterexamples\empowerment_not_reachability.py
```

Every completed formal milestone must build without `sorry` or `admit`. Every
empirical milestone must state a falsification criterion and a simpler comparison
model. Every normative milestone must expose its bridge principle and hard cases.
