# Multi-Scale Goodness and Virtue Attractors

## Status and scope

This document is the canonical conceptual specification for the project's current
account of goodness and virtue. It supersedes formulations that identify goodness
with stability, a scalar welfare functional, social agreement, or a checklist of
moral constraints. Those older constructs remain useful as measurements or
error-correcting audits, but none is the definition of goodness.

The proposal has four layers that must not be collapsed:

1. **Ground:** counterfactual contribution to phenotype-enabling metastability.
2. **Signal:** felt goodness as an embodied, fallible estimate of that contribution.
3. **Disposition:** virtue as a slow, context-transforming metastable regime.
4. **Reflection:** practical wisdom and justification as calibration and explanation.

This is a research hypothesis, not an established reduction of moral goodness to
active inference.

## 1. Relational bearer and phenotype

The bearer of the dynamics is not an isolated belief. At scale \(\ell\), let

\[
Z_t^{(\ell)}=(X_t^{(\ell)},E_t^{(\ell)},N_t^{(\ell)},R_t^{(\ell)})
\]

contain embodied states, environmental conditions, normative niche, and relations.
A phenotype \(P_i\) is an embodied organization of viability bounds, capacities,
needs, horizons, and constitutive dependencies. It is indexed explicitly because the
same environment can enable different phenotypes differently.

The relevant target is not mere persistence. Define the **attainable metastability**
\(\mathcal A_i\) of phenotype \(P_i\) as a vector or partial order over:

- remaining within or recoverably returning to viable states;
- maintaining an action repertoire under perturbation;
- retaining sensitivity to relevant evidence;
- shifting regimes when circumstances genuinely change; and
- preserving the social and ecological dependencies that make those capacities
  attainable.

It should not be reduced to a weighted scalar before the ordering, floors, and
trade-offs have been independently justified.

## 2. The ground of goodness: enabling relations

A larger-scale process is not good merely because it persists. Its candidate enabling
contribution for phenotype \(P_i\) is counterfactual:

\[
\operatorname{En}_{i}^{(\ell)}
=
\mathcal A_i
-
\mathcal A_i^{\operatorname{do}(X^{(\ell)}\ \mathrm{removed\ or\ scrambled})}.
\]

This asks whether a relationship, practice, institution, ecosystem, or other
higher-scale organization makes the phenotype's viable, recoverable, revisable modes
of life attainable. The intervention must be specified carefully: removing a
constitutive dependency may change the identity of the system, while a crude
intervention may introduce unrelated damage. Where literal removal is incoherent,
matched replacement or component-wise disruption is required.

The quantity is:

- **phenotype-relative:** \(\operatorname{En}_{i}^{(\ell)}\) may differ across agents;
- **scale-indexed:** individual, relational, institutional, and ecological effects may
  diverge;
- **temporally extended:** an immediate benefit may destroy later enabling conditions;
- **signed and multidimensional:** a process can help one capacity while undermining
  another; and
- **counterfactual:** observed coexistence, dependence, or stability is insufficient.

Thus an oppressive institution may be highly metastable while having negative
enabling effects for its constituents. Higher-scale self-maintenance is evidence of
goodness only when it counterfactually sustains, rather than consumes, the attainable
metastability of affected phenotypes.

## 3. Felt goodness as a heuristic

Agents normally do not calculate \(\operatorname{En}\). They encounter goodness as a
felt orientation: ease, rightness, fit, vitality, trust, relief, meaningfulness, or
the sense that a way of living can continue. Let

\[
g_i(t)
\approx
f_{\psi_i}\!\left(
\mathbb E_{q_i}
\left[
\Delta\mathcal A_i,
\Delta\operatorname{En}_{i}^{(1:L)}
\mid o_{1:t}
\right]
\right).
\]

Here \(g_i(t)\) is the felt-goodness signal and \(f_{\psi_i}\) is a learned,
embodied compression. It is formed through interoception, affect, action readiness,
social learning, and anticipated consequences under the agent's generative model.

The signal is useful precisely because the real counterfactual problem is too large
to solve online. It is also fallible. It may track:

- immediate comfort instead of long-horizon viability;
- social approval instead of actual enabling relations;
- focal benefit while hiding displaced costs;
- familiar controllability instead of truth;
- the persistence of an institution rather than that of its constituents; or
- a historically adaptive pattern in a changed environment.

This separates **phenomenological goodness** (what feels good or right) from
**tracking success** (what the signal accurately indicates). Their association is an
empirical question, not an identity.

## 4. Meaning, expected consequences, and shared semantics

The practical meaning of a virtue is not a dictionary definition. For agent \(i\),
context \(c\), and virtue sign \(V\), define a semantic-pragmatic profile

\[
M_i(V,c)=
\bigl(
q_i(o_{t:T}\mid V,c),
q_i(\pi\mid V,c),
A_i,B_i,C_i,D_i,
\gamma_i,
\text{protentions},
\text{felt profile}
\bigr).
\]

Meaning is therefore partly a function of expected consequences: which futures,
policies, affordances, and feelings the sign makes likely. It is also relational.
Shared semantics requires neither identical representations nor identical actions.
It requires sufficiently reciprocal processability of these profiles across relevant
contexts, including enough agreement about how evidence and consequences would
change application.

Consequently, two agents can share a virtue concept while realizing it differently,
and they can use the same word while lacking shared meaning. Coordination or total
correlation alone does not establish either semantic sharing or goodness.

## 5. Virtue as an attractor family

Classical virtue is a stable disposition organizing perception, feeling, desire,
judgment, and action. The dynamical translation is a metastable regime in a richer
state:

\[
x_i=(b_i,\gamma_i,a_i,\pi_i,h_i,m_i,g_i),
\qquad
\dot x_i=f(x_i,c,P_i,N;r_V)+\omega_i.
\]

The coordinates may include beliefs, precision, affect, policy tendencies, temporal
horizon, semantic model, and felt goodness. The virtue is not one fixed point or one
behaviour. It is a slow regulatory regime \(r_V\) that shapes the distribution of
attention, affect, inquiry, policy, and learning.

Because application changes with context and phenotype, the appropriate object is a
family of local attractor regimes:

\[
\mathfrak A_V=\{A_{V,c,P}\},
\qquad
R_{c\rightarrow c'}:A_{V,c,P}\rightarrow A_{V,c',P}.
\]

The transformation \(R\) may alter both action and felt meaning while preserving a
recognizable regulatory role. This makes courage compatible with advancing,
retreating, asking for help, or refusing a reckless demand.

Attractorhood is morally neutral. Dogmatism, servility, cruelty, and compulsive
self-protection can also be stable, confident, socially shared regimes. A candidate
virtue realization therefore requires all three relational properties:

\[
\operatorname{VirtueRealization}(V,c,P)
\Rightarrow
\operatorname{AttractorRegime}(A_{V,c,P})
\land
\operatorname{Calibrated}(g_P,\operatorname{En}_P)
\land
\operatorname{EnablingRealization}(A_{V,c,P}).
\]

This is a programme for operationalization, not yet a theorem. “Calibrated” and
“enabling” need measurement under perturbation and counterfactual intervention.

## 6. Practical wisdom and justification

**Practical wisdom** is meta-regulation of the whole inference-to-action process. It
calibrates the felt heuristic by governing:

- which scales and affected phenotypes enter the model;
- whether hidden dependencies and externalized costs are sought;
- the precision assigned to affect, testimony, social norms, and prediction errors;
- the temporal and counterfactual depth of evaluation;
- when to exploit a learned regime and when to inquire or revise it; and
- whether a local failure requires a new action, a transformed realization, or a
  change to the slow virtue regime itself.

Explicit moral justification is normally downstream of the felt orientation:

\[
q_i(H\mid g_i,o_i,\text{social narratives})
\propto
p_i(g_i,o_i\mid H)p_i(H).
\]

Reasons can reveal genuine structure, coordinate shared revision, and make conduct
contestable. They can also rationalize a captured heuristic. Experiments must
therefore measure feeling before eliciting reasons, then test how each changes when
hidden consequences or affected-party evidence are revealed.

## 7. Active-inference interpretation

VFE supplies the perceptual and learning dynamics through which the agent infers its
own state, context, dependencies, and likely consequences. EFE supplies a
policy-conditioned family of anticipated outcomes and information gains. Neither
quantity is intrinsically moral.

The proposed correspondences are:

| Project construct | Active-inference role | Non-equivalence warning |
|---|---|---|
| attainable metastability | predicted and observed viability, recovery, adaptability, and reachable-policy profile across scales | not negative VFE or mere survival |
| enabling contribution | counterfactual difference between full and disrupted multi-scale generative processes | not statistical dependence |
| felt goodness | learned affective/interoceptive estimate of expected enablingness | not prior preference itself and not guaranteed accurate |
| virtue regime | slow hierarchical parameters and states organizing perception, policy inference, affect, and learning | not one policy, reward, or precision scalar |
| practical wisdom | structure, scale, precision, horizon, stakeholder, and model-revision governance | not precision alone |
| justification | posterior inference over socially communicable explanations of feeling and consequence | not privileged access to the causes of judgment |

Agent-local EFEs remain a vector until an aggregation or bargaining rule is justified.
The theory therefore does not obtain moral goodness merely by summing expected free
energies or installing a preferred-outcome distribution.

## 8. Relation to virtue ethics and consequentialism

The account remains a virtue theory because its primary object is the organization of
the person across perception, feeling, inquiry, action, and learning. Consequences
matter constitutively to practical meaning and calibration, but an isolated act with
a favorable outcome is not thereby virtuous.

Consequentialism makes outcomes the direct currency of evaluation but assumes an
ordering of them. This account instead studies how embodied agents learn a felt proxy
for the enabling conditions of living, how stable dispositions govern its use, and
how those dispositions are corrected through consequence, testimony, and shared
inquiry.

The strongest form of the thesis is therefore:

> A virtue is a context-transforming metastable disposition whose felt orientation
> is sufficiently calibrated to the multi-scale relations that actually enable the
> attainable metastability of affected phenotypes, and whose realization tends to
> sustain those relations under relevant perturbations.

The words “sufficiently,” “affected,” “actually,” and “relevant” name open proof and
measurement obligations. They must not be hidden by a scalar score.

## 9. Moral diagnostics as error correction

Capability floors, non-externalization, non-domination, contestability, repair, and
plural improvement remain essential. Their role has changed. They are diagnostic
decompositions and institutional error-correction mechanisms that test whether a
felt heuristic or stable regime is tracking enabling relations across agents and
scales.

They detect characteristic failures:

- a capability floor detects a purported benefit bought through another's collapse;
- non-externalization expands the modeled affected set;
- non-domination tests whether apparent coordination depends on asymmetric control;
- contestability supplies error signals that conformity may suppress;
- repair tests whether harms and mistaken models can be reversed; and
- plural improvement tests whether higher-scale order enables multiple constituent
  forms of life rather than optimizing one focal phenotype.

These tests do not independently manufacture moral goodness. They discipline the
inference from felt goodness and social stability to actual enablingness.

## 10. Nested attractors

The theory predicts at least three interacting attractor scales:

1. **Phenotype scale:** viable and recoverable embodied organization.
2. **Character scale:** slow dispositions that regulate feeling, inference, and
   policy across situations.
3. **Social-ecological scale:** practices and institutions that structure available
   evidence, affordances, dependencies, and forms of life.

The relation is recursive. Social attractors shape the conditions under which
individual virtues form; virtuous and vicious actions reproduce or transform social
attractors. No scale is automatically normatively privileged. The empirical target
is the signed counterfactual relation between scales.

## 11. Central falsifiers

The proposal must be revised or rejected if:

1. felt goodness has no prospective relation to independently measured enablingness
   beyond comfort, reward, approval, and explicit outcome forecasts;
2. the relation cannot be calibrated by evidence about hidden dependencies or harms;
3. nested enabling models do not outperform individual-only viability models;
4. slow virtue-regime models add no held-out predictive value beyond traits and
   situations;
5. purported virtues do not generalize through systematic context transformations;
6. multi-scale counterfactual enablingness is not identifiable even in controlled
   systems; or
7. the account cannot distinguish resilient enabling systems from resilient
   exploitative ones without adding the desired answer by hand.
