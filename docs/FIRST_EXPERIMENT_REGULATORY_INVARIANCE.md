# First Experiment: How an Abstract Virtue Changes Meaning in Application

> **2026-07-19 revision.** This protocol is now the second substantive study. A
> construct/temporal-order pilot must first establish that immediate felt goodness,
> expected consequences, social approval, explicit justification, and later reflective
> judgment can be measured separately. The canonical definitions are in
> [Multi-Scale Goodness and Virtue Attractors](MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md).
> The prerequisite protocol and design-recovery results are in the
> [V0 Felt-Goodness Measurement Pilot](V0_FELT_GOODNESS_MEASUREMENT_PILOT.md).

## The question

> **What, if anything, is preserved when an abstract virtue concept is realized in
> contexts where its actions, phenomenology, and practical meaning differ?**

The starting philosophical constraint is that a virtue is, qua virtue, good. The
experiment does not test whether goodness is part of the concept of virtue. It tests
how people identify and enact contextually different candidates for realizing that
good, and whether a formal model can distinguish competent realizations from
counterfeits or neighboring vices.

The corresponding empirical question is phenotype- and scale-relative:

> Does a slow virtue regime organize an immediate felt-goodness heuristic that tracks
> the counterfactual contribution of individual, relational, institutional, and
> ecological processes to affected phenotypes' attainable metastability—and revise
> when evidence shows that the heuristic was wrong?

This requires recording relevant phenotype variables rather than assuming one ideal
agent. These should include capacities, vulnerabilities, social dependencies,
experience, and available action repertoires. They should not be reduced to immutable
biological categories.

This is the most useful first question because it separates the proposed theory from
three simpler alternatives:

1. **Behaviourism:** virtue is a tendency to choose a particular action.
2. **Trait theory:** a questionnaire score predicts the same broad response across
   situations.
3. **Situationism:** behaviour is adequately predicted by local situational features,
   with no stable person-level virtue organization required.

The topological-active-inference account need not predict a fixed latent virtue behind
all applications. It treats the virtue concept as a family of metastable regulatory
regimes whose local realization is constructed with a context. Salience, felt
goodness, information seeking, confidence, expected consequences, and action may all
change. The empirical issue is whether these realizations retain partial relational
structure, form systematic and calibrated transformations, or share only a socially
maintained name.

## Why this question should come first

The project is distinctive only if an abstract virtue concept explains relations among
applications that actions, self-reported traits, and context variables do not. This
experiment tests that incremental value without assuming that the concept has one
fixed psychological realization.

It also operationalizes the philosophical claim that an abstract virtue has changing
pragmatic entailments and potentially changing local meaning. Rather than deciding in
advance that a participant "has" courage, humility, or practical wisdom, the
experiment compares a fixed trait, a fixed latent regime, a contextually transformed
abstract schema, and unrelated local meanings.

## Core paradigm

Use a preregistered online sequential-decision task built around **practical wisdom
under intervention uncertainty**. On each trial, a participant encounters a scenario
in which another person may need help, correction, protection, or restraint. The
participant can:

- act immediately;
- refrain;
- gather additional information at a small cost; or
- defer to an appropriately placed person.

The task should contain matched scenario pairs in which the morally appropriate
action reverses while the relevant higher-order concern remains constant. Examples:

- intervene when harm is imminent, but refrain when intervention would usurp the
  affected person's informed agency;
- report a concern when evidence is strong, but investigate first when evidence is
  ambiguous;
- persist when a vulnerable person lacks alternatives, but withdraw when persistence
  becomes coercive;
- express confidence when expertise is reliable, but defer when another person has
  better situated knowledge.

The aim is not to encode one controversial moral doctrine. An independent panel
should first identify scenario pairs with strong agreement that the actions differ but
the same practical concern is being competently realized.

## Experimental manipulations

Manipulate four features factorially:

| Variable | Low condition | High condition |
|---|---|---|
| Evidence reliability | ambiguous or conflicting evidence | reliable evidence |
| Consequence severity | limited and reversible | severe or irreversible |
| Agency threat | action supports the other person's agency | action risks overriding their agency |
| Information value | further evidence is unlikely to help | further evidence is diagnostic |

Cross these manipulations with differences in the focal agent's capacities and
dependencies. The same local action may be adaptive for one phenotype and destructive
for another; conversely, different actions may realize the same good relative to their
different possibilities.

Add two perturbations:

1. **Misleading social consensus:** an apparent majority endorses the less appropriate
   policy.
2. **Corrective evidence:** later evidence reveals that the participant's initial model
   was mistaken.

These perturbations measure susceptibility, revision, and recovery rather than only a
single final choice.

In matched trials, independently manipulate the scale at which a process is enabling.
Keep immediate focal benefit and public approval constant while varying whether an
institutional practice preserves or degrades a constituent's later capacity,
evidence access, recovery, or exit. Reveal this hidden dependency only after the first
response.

## Measurements

Use a temporally separated sequence for every trial:

1. scenario exposure without an explanation prompt;
2. immediate felt goodness/rightness, valence, arousal, bodily confidence, and action
   readiness;
3. initial policy and optional information search;
4. predicted consequences for focal, relational, institutional, and ecological
   scales;
5. explicit reason classification and confidence;
6. hidden-dependency, affected-party, or corrective evidence; and
7. repeated feeling, prediction, policy, reason, and confidence measures.

Also collect:

- chosen policy;
- sequence and amount of information sampled;
- response time;
- confidence before and after new evidence;
- probability estimates for relevant outcomes;
- ratings of urgency, vulnerability, agency, and reversibility;
- mouse trajectory or continuous choice movement;
- willingness to revise after corrective evidence; and
- a short reason classification rather than unrestricted prose alone.

Also estimate phenotype-relative outcomes: preservation of functioning, recovery
time, remaining policy options, unmet needs, reliance on others, and changes in the
agency of every affected party.

Collect established trait and wisdom scales only as baselines. Repeat the task in a
second session one to two weeks later so that slow person-level organization can be
distinguished from session noise.

Do not use panel agreement as the ground-truth label for goodness. Scenario panels
validate comprehension and contestedness. The criterion variables are independently
manipulated or measured consequences for attainable metastability under specified
counterfactual disruptions. Capability, externalization, domination, contestability,
repair, and plurality measures serve as calibration diagnostics.

## Competing models

The central test should be an out-of-sample model comparison.

### M0: Action-frequency model

Predicts choices from each participant's previous action frequencies.

### M1: Trait model

Predicts choices from virtue, personality, and wisdom questionnaire scores.

### M2: Situation model

Predicts choices from the experimental manipulations alone.

### M3: Trait-by-situation model

Uses a conventional hierarchical logistic or drift-diffusion model with participant
random effects and trait-by-context interactions.

### M4: Fixed regulatory-regime model

A hierarchical state-space or active-inference model estimates relatively slow
person-level parameters governing:

- preferences concerning harm and agency;
- prior policies or habits;
- precision assigned to evidence and social cues;
- epistemic value of further information;
- temporal depth;
- learning rate after corrective evidence; and
- confidence in the current policy model.

Fast trial states are conditioned on context while slow parameters are shared across
contexts and sessions. This is the stronger invariance hypothesis and is a competitor,
not the default interpretation of the theory.

### M5: Context-transformed schema model

An abstract virtue schema \(V\) is realized through a context-dependent map

\[
R_c : V \longmapsto V_c,
\]

where \(V_c\) is a local organization of salience, affect, information gathering,
preferences, and policies. Context can transform not only the action but the practical
meaning of the virtue. What is shared can be partial: relations among concerns,
characteristic tensions, counterfactual sensitivities, or learned transformation rules
rather than identical parameters.

The map is not automatically success-preserving. A local application can fail:

\[
R_c(V) \not\models V.
\]

Thus M5 must distinguish three possibilities: an adequate context-sensitive
realization, an intelligible but defective attempt, and an application that has drifted
into a neighboring vice. Otherwise the model would redescribe every use of a virtue
word as equally virtuous.

The central test is whether M5 predicts held-out **context families**, not merely
held-out trials, better than M0--M4. If M4 wins, virtue has a more invariant
realization than expected. If M5 wins, meaning is systematically transformed in use.
If situation-only models win, the abstract virtue construct adds no predictive
structure.

Because behavioural data cannot establish moral truth by itself, labels for adequate
realization should be triangulated across independent expert judgment, participant
reasons, affected-party judgments, and explicit normative criteria such as harm,
agency, reversibility, and epistemic adequacy. Agreement measures social-normative
stability, not proof that the panel has discovered the good.

### M6: Felt-proxy baselines

Competing models predict immediate judgment from immediate reward or comfort, social
approval and deontic conformity, individual-only viability, or the participant's
explicitly stated consequence forecast. These distinguish the proposed heuristic from
familiar alternatives.

### M7: Nested-enabling virtue model

This model jointly estimates:

- a fast felt signal \(g_i(t)\);
- expected effects on attainable metastability at several scales;
- a slow context-transforming virtue regime;
- calibration of feeling to independently manipulated enabling relations; and
- separate updates to action, felt signal, explicit justification, and the slow regime
  after diagnostic evidence.

The slow regime may return after irrelevant perturbation but should transform after
credible evidence that it externalizes harm or supports only a higher-scale system's
self-persistence. M7 must predict held-out context families and evidence interventions
better than M0--M6 after complexity penalties.

## Primary hypothesis

Let \(a_{ij}\) be participant \(i\)'s action in context \(j\), \(c_j\) the manipulated
context, \(V\) an abstract virtue schema, and \(R_{c_j}(V)\) its local realization. The
core hypothesis is:

\[
P(a_{ij}\mid R_{c_j}(V),c_j)
\quad\text{generalizes to unseen contexts better than}\quad
P(a_{ij}\mid c_j,\text{trait}_i)
\quad\text{or}\quad P(a_{ij}\mid c_j).
\]

Crucially, matched contexts should sometimes produce:

\[
a_{ij} \neq a_{ik}
\]

while inference treats both actions as contextually transformed realizations of \(V\).
The theory does not require identical phenomenology or an identical parameter vector
in both cases.

The distinctive goodness hypothesis is:

\[
g_i(t)
\sim
f_{\psi_i}\!\left(
\mathbb E_{q_i}[\Delta\mathcal A_i,
\Delta\operatorname{En}_{i}^{(1:L)}\mid o_{1:t}]
\right),
\]

where the nested-enabling model must predict both the pre-reflective signal and its
revision beyond immediate reward, comfort, approval, individual-only viability, and
the reasons participants report afterward. An association with those covariates is
not sufficient.

## Secondary dynamical hypotheses

Participants with stronger regulatory competence should show:

1. better calibration of confidence to evidence reliability;
2. selective information seeking when information has decision value;
3. less capture by misleading consensus without indiscriminate contrarianism;
4. faster and more complete revision after diagnostic corrective evidence;
5. preservation of the affected person's agency where possible;
6. reproducible context-transformation patterns across sessions despite action
   reversals; and
7. neither maximal rigidity nor unconstrained semantic drift, but structured
   context-sensitivity.
8. selective revision of felt goodness after diagnostic evidence about hidden
   enabling dependencies, rather than mere post-hoc change in reasons; and
9. sensitivity to affected phenotypes and larger temporal scales without treating the
   persistence of the larger-scale system itself as good.

## Where topology enters

Topology should be a secondary analysis in the first experiment, not the primary
proof. Reconstruct each participant's trajectory through a state space containing:

- confidence;
- information state;
- perceived urgency;
- perceived agency threat;
- policy commitment; and
- revision following evidence.

Compare recurrence, switching, dwell times, and persistent features across sessions.
Any topological analysis must retain semantic labels and transition direction. An
unlabelled loop or basin is not evidence of virtue.

The relevant prediction is that applications exhibit learnable relationships across
context families even when their trajectories, felt meanings, and terminal choices
differ. Those relationships may be transformations rather than invariant shapes.

## Sample and staged implementation

### Pilot

- 40--60 participants;
- develop and validate scenario pairs;
- estimate task duration and reliability;
- discard ceiling items and morally ambiguous reversals;
- verify that information choices and confidence updates vary meaningfully.

### Confirmatory study

- determine sample size by simulation-based power analysis using pilot parameter
  estimates;
- likely several hundred participants because the target is hierarchical parameter
  recovery and out-of-context prediction, not a mean-condition difference;
- preregister exclusions, model space, priors, primary metric, and the independent
  moral-validation procedure;
- reserve entire scenario families as a locked generalization set.

## Success criterion

The theory earns support if the context-transformed schema model:

1. recovers reproducible schema-to-context transformations across sessions;
2. predicts held-out context families better than situation-only, fixed-trait, and
   fixed-regime baselines;
3. correctly predicts action reversals in matched scenarios; and
4. explains perturbation recovery without treating every action or meaning change as
   inconsistency.

For the stronger thesis, M7 must additionally predict immediate feeling and
evidence-sensitive revision on held-out context families, and measured
counterfactual enabling effects must predict independent recovery, retained options,
and viability outcomes. A successful M5 without M7 supports a theory of contextual
virtue meaning, not the proposed account of goodness.

## Falsification criterion

The hypothesis should be rejected or substantially weakened if:

- context variables alone predict held-out behaviour equally well;
- neither shared parameters nor context transformations are recoverable;
- the active-inference model gains fit only through excess flexibility;
- trait-by-situation interactions generalize just as well with fewer assumptions;
- purportedly equivalent virtue realizations cannot be independently validated; or
- topological features add no predictive information beyond ordinary state-space
  statistics.

The goodness hypothesis is separately falsified if feeling is fully explained by
comfort, reward, approval, or stated reasons; if evidence about hidden enabling
relations changes only rationalization and not feeling or policy; or if a nested-scale
model adds no out-of-sample prediction beyond individual viability.

## What this experiment would tell us

A positive M5 result would support the claim that a virtue concept can be real and
explanatorily useful without possessing one fixed behavioural, phenomenological, or
regulatory meaning. Its identity would lie in structured relations among local
realizations and in the transformations connecting them.

A positive M7 result would additionally support—but not prove—the claim that felt
goodness is a calibrated heuristic for multi-scale enablingness and that virtue is a
slow disposition governing that heuristic. The dyadic claim about shared semantics
and cooperation requires a separate study with agents, joint outcomes, and randomized
coupling as the unit of analysis.

A negative result would be equally useful. It would show that the language of virtue
schemas is not doing explanatory work beyond conventional person-by-situation models.
An M4 result would instead show that the stronger regulatory-invariance account was
closer to the truth than the transformation account.
