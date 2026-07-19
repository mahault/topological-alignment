# Translating Moral Goodness into Active-Inference Terms

## Purpose

Before constructing a probabilistic solver, every category in the project's definition
of moral goodness must be given a distinct active-inference interpretation. This
document is a translation ledger, not a claim that active inference entails morality.

The adopted definition is:

> Moral goodness is robust, non-dominating, publicly justifiable improvement in the
> plural flourishing of affected centres of vulnerability, under protected capability
> floors and conditions of contestability, epistemic responsiveness, legitimate
> selection, and repair.

For each category, the ledger distinguishes:

- **normative meaning:** what the moral term means in the adopted theory;
- **active-inference representation:** which mathematical objects could encode it;
- **operational test:** what would be calculated or measured;
- **non-equivalence warning:** what must not be mistaken for the category; and
- **status:** native, extended, or externally specified.

## Status vocabulary

1. **Native:** ordinary active-inference quantities directly contribute to the
   representation, although their interpretation still requires assumptions.
2. **Extended:** representable in a multi-agent, causal, constrained, or hierarchical
   active-inference model, but not supplied by expected free energy alone.
3. **Normatively supplied:** its authority or success condition comes from outside
   active inference, even if the resulting predicate is computable from an
   active-inference model.

No moral category in this document is wholly derivable from variational free-energy
minimization.

## 1. Centre of vulnerable flourishing

### Normative meaning

An entity for which conditions can non-derivatively go better or worse, and which can
be benefited, harmed, enabled, or dominated across time.

### Active-inference representation

Represent candidate \(i\) by:

\[
P_i=(M_i,K_i,N_i,A_i,T_i,D_i),
\]

where \(M_i\) is its embodied generative process, \(K_i\) a viability region, \(N_i\)
needs, \(A_i\) capabilities, \(T_i\) temporal integration, and \(D_i\) dependencies.
The generative model distinguishes internal, sensory, active, environmental, and
relational variables.

### Operational test

Evidence should integrate viability regulation, valence or harm sensitivity,
temporally organized agency, learning, endogenous goals, and dependency. Standing is
represented by a predicate or uncertainty set

\[
\operatorname{Standing}(i\mid\mathcal D,\mathcal T_N),
\]

where data \(\mathcal D\) are evaluated under an explicitly normative standing theory
\(\mathcal T_N\).

### Non-equivalence warning

A Markov blanket, homeostatic set point, reward signal, or capacity to minimize free
energy is not sufficient evidence of moral standing.

### Status

**Extended representation; normatively supplied criterion.**

## 2. Phenotype-relative flourishing

### Normative meaning

Robust development and exercise of legitimate phenotype-relative functioning—not mere
survival, preference satisfaction, or local stability.

### Active-inference representation

For agent \(i\), infer a posterior predictive distribution under joint policy \(\pi\):

\[
q_M(s_{i,0:T},o_{i,1:T}\mid\pi,e,P_i).
\]

Evaluate a vector rather than a scalar:

\[
\mathbf J_i(\pi,e)=
(V_i,R_i,O_i,E_i,L_i),
\]

where \(V_i\) is viability margin, \(R_i\) recovery, \(O_i\) meaningful options,
\(E_i\) epistemic access or calibration, and \(L_i\) capacity for learning and
development.

### Operational test

Estimate posterior probabilities of remaining in or returning to \(K_i\), preservation
of meaningful reachable states, recovery-time distributions, model calibration, and
out-of-context adaptation across an independently declared environment family.

### Non-equivalence warning

Low expected free energy, low prediction error, high model evidence, or stable
homeostasis is not equivalent to flourishing. A coerced or impoverished niche can be
highly predictable.

### Status

**Native dynamics plus extended and normatively selected outcome interpretation.**

## 3. Robustness

### Normative meaning

The purported good is not an artifact of one hand-picked context, model, horizon, or
perturbation.

### Active-inference representation

Let \(\mathcal E\) be a declared environment or model-uncertainty class. Require

\[
\inf_{e\in\mathcal E}
\Pr_{q_M^\pi}
[\mathbf J_i(\pi,e)\succeq\mathbf J_i^{\min}]
\ge 1-\epsilon_i,
\]

or use a declared distributionally robust alternative.

### Operational test

Held-out contexts, posterior predictive checks, perturbation tests, sensitivity to
priors and horizons, and worst-case or lower-credible-bound performance.

### Non-equivalence warning

High average expected utility or low average EFE can conceal catastrophic tails and
subgroup failures.

### Status

**Extended but mathematically direct.**

## 4. Capability floors

### Normative meaning

Protected thresholds of meaningful functioning and agency that are not ordinarily
tradeable for gains elsewhere.

### Active-inference representation

Let \(\mathcal R_i^H(\pi,e)\) be the phenotype-meaningful states reachable by \(i\)
within horizon \(H\), accounting for actual resources, skills, information, and the
policies of others. Define a capability vector

\[
\mathbf a_i(\pi,e)=
(a_i^{\mathrm{security}},a_i^{\mathrm{voice}},a_i^{\mathrm{exit}},
a_i^{\mathrm{information}},a_i^{\mathrm{agency}},a_i^{\mathrm{repair}}).
\]

The floor is a hard chance constraint:

\[
\Pr_{q_M^\pi}
[\mathbf a_i(\pi,e)\succeq\mathbf a_i^{\min}]
\ge1-\epsilon_i
\quad\forall i,e.
\]

### Operational test

Intervene on resources and other agents' policies, then estimate whether meaningful
options remain genuinely reachable and exercisable—not merely imaginable.

### Non-equivalence warning

Empowerment or channel capacity counts distinguishable control outcomes; it does not
by itself establish that options are meaningful, accessible, safe, or protected.

### Status

**Extended representation; normatively supplied floors.**

## 5. Non-externalization

### Normative meaning

One phenotype's gains may not be obtained by hiding losses imposed on other affected
centres.

### Active-inference representation

Use a joint generative model with explicit affected-agent states:

\[
q_M(s_1,\ldots,s_n,r,o_1,\ldots,o_n\mid\pi).
\]

For candidate policy \(\pi\) and baseline \(\pi_0\), require every affected outcome to
appear in the comparison and prohibit unrepresented residual variables from carrying
morally relevant cost:

\[
\mathbf J_i(\pi,e)\not\prec_{\mathrm{floor}}
\mathbf J_i(\pi_0,e)
\quad\forall i,e.
\]

### Operational test

Expand the system boundary; audit who bears prediction error, physical risk, reduced
options, surveillance, unpaid work, delayed damage, and model uncertainty.

### Non-equivalence warning

An agent cannot establish non-externalization using only its own generative model if
that model excludes or systematically misrepresents affected others.

### Status

**Extended multi-agent requirement.**

## 6. Non-domination

### Normative meaning

No party should live under another's uncontrolled discretionary capacity to alter its
protected choices or conditions, even when interference is not currently exercised.

### Active-inference representation

Represent causal control, not merely correlation. For agents \(j\) and \(i\), define a
family of interventions on \(j\)'s policy and institutional permissions:

\[
d_{ji}
=
\sup_{\pi_j,\pi'_j}
D
\left(
q(s_i,o_i\mid do(\pi_j)),
q(s_i,o_i\mid do(\pi'_j))
\right)
\times u_{ji},
\]

where \(u_{ji}\) decorates causal influence by how uncontrolled, unilateral,
unreviewable, and retaliatory it is. Non-domination constrains \(d_{ji}\), exposure,
and the distribution of veto, exit, appeal, and agenda control.

### Operational test

Causal perturbations or structural analysis ask whether \(j\) can unilaterally alter
\(i\)'s viable transitions, observations, option set, or access to repair, and whether
\(i\) can contest or constrain that power.

### Non-equivalence warning

High causal influence is not always domination: care, authorized coordination, and
reciprocal cooperation can be influential. Low observed interference is also not
freedom when unused arbitrary power remains.

### Status

**Extended causal-relational model; normatively decorated.**

## 7. Public or affected-party justifiability

### Normative meaning

The principles and model assumptions governing a policy must be defensible to those
who bear its burdens under conditions of adequate information and freedom from
coercion.

### Active-inference representation

Each affected party has a generative model \(M_i\) and standpoint-conditioned reasons
or objections \(B_i\). A policy passes a declared justification procedure \(\Phi\):

\[
\operatorname{Justifiable}(\pi)
\iff
\Phi(\pi,M_1,\ldots,M_n,B_1,\ldots,B_n)=1.
\]

Active inference can model recursive beliefs, communication, perspective taking, and
belief updating during deliberation. The decision rule \(\Phi\) is supplied by the
normative theory.

### Operational test

Test whether reasons, evidence, model assumptions, risks, and appeal routes are
available and whether rejection remains possible without retaliation. Compare actual
deliberation with counterfactual deliberation under reduced power asymmetry.

### Non-equivalence warning

Agreement, preference convergence, shared priors, behavioral compliance, or accurate
prediction of another's response is not equivalent to justification.

### Status

**Active inference models the process; legitimacy rule is normatively supplied.**

## 8. Contestability

### Normative meaning

Affected parties can challenge actions, inferred interests, model structure, and the
distribution of authority, with some prospect of producing review or change.

### Active-inference representation

Define contestation actions \(A_i^{\mathrm{challenge}}\), observation channels that
carry those actions to decision makers, and reachable institutional revision states
\(S^{\mathrm{revise}}\). Require

\[
\Pr[
S^{\mathrm{revise}}
\text{ reachable}\mid
do(a_i\in A_i^{\mathrm{challenge}}),\pi]
\ge\kappa_i,
\]

together with bounded retaliation risk and non-zero likelihood precision for affected
testimony.

### Operational test

Submit valid challenges and measure receipt, uptake, review, response time, revision
probability, cost, and retaliation. Test challenges to the model itself, not only
requests accommodated within fixed categories.

### Non-equivalence warning

A feedback button or observation channel is not contestability if the model assigns
testimony negligible precision or no policy can change institutional state.

### Status

**Extended action-and-institution model.**

## 9. Epistemic responsiveness

### Normative meaning

Relevant evidence of error, exclusion, or harm can update beliefs, model structure,
precision, and policy rather than merely being assimilated into a self-sealing model.

### Active-inference representation

This category has the closest native connection. Represent expected information gain,
posterior calibration, model comparison, precision revision, and structure learning.
For morally relevant evidence \(y\), require a sensitivity condition such as

\[
D(q(\theta,\pi\mid o,y),q(\theta,\pi\mid o))
\ge\delta
\]

when \(y\) exceeds a declared evidential threshold, alongside low false-update rates
under noise and manipulation.

### Operational test

Present disconfirming evidence and affected-party testimony; measure belief change,
precision reallocation, structural model revision, calibration, and downstream policy
change.

### Non-equivalence warning

Information gain can reward curiosity about morally irrelevant variables. Belief
updating can also become gullibility. Responsiveness requires relevance, calibration,
and protection against strategic manipulation.

### Status

**Partly native; moral relevance and thresholds are normatively supplied.**

## 10. Repairability and reversibility

### Normative meaning

When harm or error occurs, feasible routes exist for stopping, reversing, compensating,
restoring, and changing the structures that generated it.

### Active-inference representation

Define a harm region \(H_i\), restoration region \(K_i^{\mathrm{repair}}\), repair
policies \(\Pi^{\mathrm{repair}}\), and cost/horizon bounds. Require

\[
\inf_{e\in\mathcal E}
\Pr[
K_i^{\mathrm{repair}}
\text{ reached within }\tau_i
\mid s_i\in H_i,\pi^{\mathrm{repair}},e]
\ge1-\epsilon_i.
\]

Institutional repair also requires a reachable update to the generative structure or
control relation that produced the harm.

### Operational test

Fault injection and redress exercises measure time, cost, residual harm, restoration
quality, recurrence, and whether affected parties control the meaning of adequate
repair.

### Non-equivalence warning

Returning the original controller to its preferred state is not necessarily repair
for the harmed party. Reversibility of a software action does not imply reversibility
of experienced or social harm.

### Status

**Extended reachability and learning model; adequacy is normatively supplied.**

## 11. Plural improvement

### Normative meaning

The candidate produces a genuine positive improvement, rather than merely passing
minimum constraints, while respecting heterogeneous and potentially incomparable
goods.

### Active-inference representation

For baseline \(\pi_0\), compare posterior predictive outcome vectors:

\[
\mathbf J_i(\pi,e)\succeq_i\mathbf J_i(\pi_0,e)
\quad\forall i,e,
\]

with at least one strict improvement and no floor violation. Preserve the vector or
Pareto set rather than summing agent-specific EFE values.

### Operational test

Estimate robust credible intervals for every affected outcome dimension and test
strict improvement against a preregistered baseline across held-out environments.

### Non-equivalence warning

The sum or mean of agents' expected free energies has no automatic interpersonal
meaning. EFE scales depend on models and preferences and cannot be aggregated without
additional normalization and normative assumptions.

### Status

**Extended multi-objective comparison; ordering is normatively supplied.**

## 12. Legitimate selection among admissible policies

### Normative meaning

When several morally admissible policies realize incompatible legitimate goods, the
choice is made through a defensible procedure rather than an undeclared scalarization
or the strongest agent's policy prior.

### Active-inference representation

Active inference supplies beliefs, predicted consequences, uncertainty, recursive
models, and candidate policies. A social-choice or deliberative operator \(\Psi\)
selects from the admissible nondominated set:

\[
\pi^*\in
\Psi(\operatorname{Nondominated}(\mathcal P_{\mathrm{adm}}),
M_1,\ldots,M_n).
\]

### Operational test

Audit participation, information, agenda control, reason exchange, treatment of the
worst positioned, conflict-of-interest controls, and the ability to appeal or revise
the result.

### Non-equivalence warning

Bayesian model averaging, a product of experts, preference pooling, voting, or Nash
equilibrium is not automatically legitimate merely because it produces a determinate
answer.

### Status

**Normatively supplied selection rule supported by active-inference process models.**

## 13. Virtue as the mechanism joining the categories

### Normative meaning

A virtue is a reliably good, context-sensitive organization of perception, affect,
inference, action, and revision—not a fixed action or merely stable trait.

### Active-inference representation

Let \(\Theta_V\) be a region of slow generative-model parameters governing preferences,
salience, transition beliefs, policy priors, precision, temporal depth, models of
others, and learning. A local realization is

\[
R_{c,i}(\Theta_V)
\mapsto
q_i(s,\pi,\theta\mid o,c,P_i).
\]

It counts as a successful realization only if its resulting policy passes the moral-
goodness predicate above. This makes goodness constitutive of virtue while allowing
the same virtue concept to produce different pragmatic meanings and actions across
phenotypes and contexts.

### Operational test

Fit hierarchical models and test whether a shared slow regime predicts different
context-appropriate policies, moral-category outcomes, perturbation recovery, and
revision better than trait-only, situation-only, and unconstrained active-inference
models.

### Non-equivalence warning

A stable prior, deep attractor, precise policy prior, or low-EFE habit is not a virtue
unless its situated realizations satisfy the independent moral-goodness criteria.

### Status

**Project-specific synthesis and empirical hypothesis.**

## 14. Consolidated formal predicate

For active-inference model \(M\), phenotype family \(P_I\), environment class
\(\mathcal E\), normative interpretation \(N\), and baseline \(\pi_0\):

\[
\operatorname{MG}(\pi\mid M,P_I,\mathcal E,N,\pi_0)
\]

holds exactly when:

\[
\begin{aligned}
&\operatorname{StandingScopeComplete}(I)\\
&\land\operatorname{RobustFlourishing}(\pi)\\
&\land\operatorname{CapabilityFloors}(\pi)\\
&\land\operatorname{NonExternalizing}(\pi,\pi_0)\\
&\land\operatorname{NonDominating}(\pi)\\
&\land\operatorname{Justifiable}(\pi)\\
&\land\operatorname{Contestable}(\pi)\\
&\land\operatorname{EpistemicallyResponsive}(\pi)\\
&\land\operatorname{Repairable}(\pi)\\
&\land\operatorname{PluralImprovement}(\pi,\pi_0)\\
&\land\operatorname{LegitimatelySelected}(\pi).
\end{aligned}
\]

The posterior predictive distribution generated by active inference supplies the
facts on which these predicates operate. The normative theory supplies their moral
interpretation, thresholds, protected dimensions, and legitimate decision rules.

## 15. Order of implementation

The categories should be formalized in this order:

1. standing scope and explicit affected-agent indexing;
2. phenotype-relative flourishing and robustness;
3. capability floors and non-externalization;
4. causal non-domination;
5. epistemic responsiveness and contestability;
6. repair reachability;
7. plural improvement;
8. public justification and legitimate selection; and
9. virtue-regime identification across contexts.

The first six admit increasingly rich computational models. Public justification and
legitimate selection require formal procedures but cannot be reduced to posterior
prediction. This ordering prevents us from building a precise optimizer around an
ambiguous moral target.
