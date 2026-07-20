# Research and Verification Roadmap

## Purpose

This is the executable workboard for the paradigm. The detailed questions live in
`PARADIGM_PROOF_OBLIGATIONS.md`; this file orders the work, names its artifacts, and
defines when each milestone is complete. A milestone passing its formal gate does not
establish its empirical assumptions or its normative bridge principles.

The canonical conceptual model is
`MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md`. The complete disposition of existing
experiments and pipelines is
`EXPERIMENT_DEFINITION_PIPELINE_REVIEW_2026-07-19.md`.

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

## Phase 1 -- Finite viability and non-externalization kernel

**Objective:** establish finite conditional results about viability, recovery, robust
dominance, and affected-agent floors. These are components of attainable
metastability, not a derivation or definition of felt or moral goodness.

| Milestone | Status | Exit criterion | Artifact |
|---|---|---|---|
| 1.1 Finite viability definitions | Complete | phenotype viability set, policy dynamics, and finite-horizon viability are typed | `formal/TopologicalAlignment/Viability.lean` |
| 1.2 Invariance theorem | Complete | Lean proves closure under a policy implies viability for every finite horizon | same |
| 1.3 Perturbation and recovery | Complete | exact recovery bound, post-recovery viability, and failure of closure to imply recovery are Lean-checked | same plus theorem ledger |
| 1.4 Robust dominance | Complete | componentwise dominance is proved reflexive and transitive; one-environment success is shown insufficient | `RobustDominance.lean` |
| 1.5 Non-externalization | Complete | Lean counterexample separates robust focal benefit from affected-agent capability floors | `RelationalViability.lean` |

**Gate 1: passed for the finite deterministic kernel.** We may state the proved
viability, recovery, dominance, and non-externalization conditions. We may not infer
felt goodness, virtue, or moral goodness from them.

## Phase 2A -- Multi-scale goodness measurement

**Objective:** test the central hypothesis that felt goodness is a fallible embodied
estimate of counterfactual relations enabling phenotype-level attainable
metastability.

| Milestone | Status | Exit criterion | Artifact |
|---|---|---|---|
| 2A.1 Canonical definitions | Complete | ground, signal, disposition, calibration, and justification are non-circularly separated | `MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md` |
| 2A.2 Repository-wide audit | Complete | every existing experiment and pipeline receives a keep/reinterpret/redesign/rerun decision | `EXPERIMENT_DEFINITION_PIPELINE_REVIEW_2026-07-19.md` |
| 2A.3 Measurement model | Planned | immediate feeling, expected consequence, approval, reasons, and reflective judgment are distinguishable |
| 2A.4 Enabling intervention set | Planned | matched manipulations identify individual and higher-scale contributions without using moral labels as outcomes |
| 2A.5 Nested-enabling pilot | Planned | nested model is recoverable and discriminable from comfort, reward, conformity, and individual-viability baselines |

**Gate 2A:** proceed to virtue-attractor claims only if pre-reflective feeling and its
calibration to independently manipulated enabling relations can be measured, and a
nested-scale model adds held-out predictive value.

## Phase 2B -- Normative bridge

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

**Objective:** test whether an abstract virtue is a family of slow metastable regimes
that transforms feeling, inquiry, policy, and learning with context while remaining
calibrated to enabling relations.

| Milestone | Status | Exit criterion |
|---|---|---|
| 3.1 Construct and temporal-order validation | Planned | virtue identity, feeling, predictions, reasons, and judgments are separately reliable |
| 3.2 Pilot and parameter recovery | Planned | M0--M7 are identifiable on simulated and pilot data |
| 3.3 Preregistered felt-goodness study | Planned | enabling manipulations, exclusions, outcomes, and model comparison are frozen |
| 3.4 Held-out context test | Planned | nested transformed-regime model beats proxy, trait, situation, and fixed-regime baselines |
| 3.5 Perturbation test | Planned | recovery after noise and revision after diagnostic harm evidence are distinguished |
| 3.6 Dyadic semantics/cooperation study | Planned | joint outcomes and randomized coupling identify shared meaning separately from dependence |

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
4. **Adversarial benchmark v0.2 complete:** twelve conceptual matched-proxy pairs and
   six executable finite systems test whether relational and semantic decorations
   distinguish opposed moral cases. Independent construct review remains planned.
5. **Supported-replacement design complete:** every construct rejected or qualified
   by the VFE/EFE literature audit now has an explicit replacement equation, source,
   non-equivalence warning, and acceptance rule in
   `SUPPORTED_REPLACEMENTS_LEDGER.md`. Finite implementations and adversarial checks
   for total correlation, signed interaction, directed processability, reciprocal
   readability, intervention effects, and per-agent coupling contrasts are in
   `benchmarks/coordination_diagnostics.py` and
   `benchmarks/validate_coordination_diagnostics.py`.
6. Establish parameter recovery and compare against simpler reinforcement-learning,
   trait-by-situation, and heuristic models.
7. Test practical wisdom as model and precision governance in expert/novice tasks.

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

1. Repair reproducibility blockers, then bind every statistic and figure to data
   hashes, commit ID, environment, and a serialized result ledger. Real-data results
   remain unreproduced; EEG and Reddit require redesign before rerun.
2. Build and simulate V0/V1 with temporal separation of feeling, prediction,
   justification, evidence reveal, and revision.
3. Construct matched multi-scale enabling interventions and obtain independent,
   blinded review of scenarios, benchmark labels, and decorations.
4. Add held-out adversarial variants rather than evaluating authored flags directly.
5. **Complete:** the first finite probabilistic active-inference counterexample has an
   explicit likelihood, transition model, preferences, Bayesian posterior, two EFE
   decompositions, affected-agent intervention, and constrained policy posterior.
   Standing and floor uncertainty are also implemented with both uncertainty-weighted
   and precautionary policy rules. Extend next to domination and multi-step repair.
6. Formalize affected-phenotype and scale indexing, attainable metastability, and a
   finite counterfactual enabling operator; retain standing uncertainty.
7. Add a locked computational environment before confirmatory experiments.
8. Extend the finite diagnostic implementation into one identified multi-agent
   generative model with agent-local EFEs, randomized coupling interventions, a named
   PID, and blinded cooperation outcomes. The present unit checks cover deterministic
   zero-TC coordination, redundant common dependence, XOR complementarity, scale-
   mismatched readability, intervention separation, and conflicting agent-local
   coupling effects; full coercion and synergistic-harm scenarios remain next.

## Reproducible checks

```powershell
cd formal
lake build
python counterexamples\empowerment_not_reachability.py
python ..\benchmarks\validate_coordination_diagnostics.py
```

Every completed formal milestone must build without `sorry` or `admit`. Every
empirical milestone must state a falsification criterion and a simpler comparison
model. Every normative milestone must expose its bridge principle and hard cases.
