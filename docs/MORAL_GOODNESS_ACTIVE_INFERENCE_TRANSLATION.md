# Translating Moral Goodness into Active-Inference Terms

## Purpose

> **2026-07-19 reinterpretation:** this is now a translation ledger for reflective
> moral admissibility and error correction, not the canonical definition of felt
> goodness. The upstream ground/signal/disposition model is
> [Multi-Scale Goodness and Virtue Attractors](MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md).

Before constructing a probabilistic solver, every category in the project's moral
audit must be given a distinct active-inference interpretation. This document is a
translation ledger, not a claim that active inference entails morality.

The adopted audit criterion is:

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

## Semantic and joint-distribution precondition

The moral predicates below must not be applied directly to action labels such as
`cooperate`, `courageous`, or `humble`. Those labels acquire content through predicted
consequences and their interpretation by affected agents. Let a joint policy be

\[
\boldsymbol\pi=(\pi_1,\ldots,\pi_n)
\]

and let the social generative model induce the joint posterior predictive distribution

\[
Q^{\boldsymbol\pi}_e
=q_M(s_{1:n,0:T},o_{1:n,1:T},a_{1:n,0:T-1},r_{0:T}
\mid\boldsymbol\pi,e).
\]

Each agent constructs a phenotype- and context-relative semantic-pragmatic profile

\[
\mathfrak M_i(v,c)=
\left(
Q_i(s,o\mid\pi,z_v,c),
Q_i(\pi\mid z_v,c),
P_i(\pi\mid o^d,c),
Q_i(\theta_i\mid z_v,c),
\gamma_i(c)
\right).
\]

The shared sign \(z_v\) and local parameters can be inferred through a naming game,
federated inference, or a declared joint variational model. The profile includes
anticipated consequences, induced policies, deontic cue mappings, parameter beliefs,
and precision. It operationalizes inferential and pragmatic role without claiming
that predicted consequences exhaust the full meaning of a thick moral concept.

For these profiles to be shared, they need not be identical. First require posterior
evidence for a common sign. Then fit context-sensitive alignment maps on held-out
overlapping content and report directed processability. The stronger sheaf diagnostic
from the sibling `shared-protention-alignment` project asks whether compatible local
anticipations glue to a global section. Write the combined diagnostic as

\[
\operatorname{SemCompat}_{\mathcal I}
=\left(
q(z_v\mid o_{1:n},u_{1:n}),
\operatorname{Proc}_{i\to j},
\|\delta x\|
\right).
\]

where \(\mathcal I\) declares what must survive translation. If overlap observations
and restriction maps are part of the generative model, their consistency error appears
in the VFE accuracy term and gluing can be computed from learned posterior-predictive
maps. Otherwise gluing is an external diagnostic; ordinary VFE does not automatically
learn a sheaf. This distinguishes shared semantics from agreement, preference
convergence, or identical world models.

Cooperative organization must be investigated through joint predictions and
interventions, not read from one agent's action label. Use the diagnostic vector

\[
\mathbf C(\boldsymbol\pi)=
\left(
Q(Y\in Y^*\mid\boldsymbol\pi),
\operatorname{Proc}_{\leftrightarrow},
\operatorname{Read}_{\leftrightarrow},
\operatorname{Syn}_Y,
\operatorname{CIF}_{1:n\to Y},
\boldsymbol\Delta^{\mathrm{cpl}}
\right),
\]

where \(\Delta_i^{\mathrm{cpl}}=G_i^{do(C=0)}-G_i\) is retained separately for every
agent. Total correlation is recorded only as a dependence baseline. The vector
separates joint achievement, semantic processability, reciprocal readability,
outcome-relevant interaction, causal contribution, and agent-local EFE effects. It is
still insufficient for moral cooperation: pragmatic benefit, epistemic benefit, and
the independently justified relational constraints must be tested separately.

## Semantic lifting of every category

The whole definition must be evaluated only after the same three-stage construction:

\[
\boldsymbol\pi
\xmapsto{M,e} Q^{\boldsymbol\pi}_e
\xmapsto{\mathrm{infer}} (z_v,\mathfrak M_{1,c},\ldots,\mathfrak M_{n,c})
\xmapsto{\operatorname{SemCompat}_{\mathcal I}}
(\operatorname{Proc}_{\leftrightarrow},\|\delta x\|).
\]

The first arrow is generated by EFE-conditioned policy dynamics, the second by
Bayesian or VFE-based sign and parameter learning, and the third comprises held-out
diagnostics computed from the learned maps. None is an analyst-attached action label.
A normative interpretation \(N\) then evaluates that endogenous content.
For compactness in the predicates below, write \(m_i:=\mathfrak M_i(v,c)\), and write
\(m^{\mathrm{sh}}\) only for a fused overlap projection whose processability and
sheaf-consistency diagnostics have passed. This notation does not imply identical
private beliefs.
Accordingly, every term in the adopted audit criterion is a derived predicate:

1. **Affected centre of vulnerability:**
   \(\operatorname{Affected}_i(Q,m,N)\) holds when intervention on the joint policy
   changes consequences that, under a warranted interpretation of phenotype \(i\),
   can alter its flourishing or protected functioning. Causal exposure alone is not
   standing; exclusion from the shared vocabulary cannot erase actual effects.

2. **Phenotype-relative flourishing:**
   \(\operatorname{Flourish}_i(Q,m_i,N)=F_i(\mathfrak M_i(v,c;Q),P_i,N)\). Viability,
   recovery, learning, relationship, and option trajectories become flourishing only
   through a phenotype-sensitive account of what they enable. The same physical
   consequence can therefore have different pragmatic meanings without making every
   interpretation equally warranted.

3. **Robustness:**
   \(\operatorname{Robust}(\boldsymbol\pi)=\inf_{e,M,I\in\mathcal U}
   \operatorname{Adequate}_{N}(Q^{\boldsymbol\pi}_{e,M},m_I)\). Robustness ranges
   over uncertainty in dynamics *and* interpretation. Success under one convenient
   semantic coding of harm, agency, or achievement is not robust success.

4. **Capability floors:**
   \(\operatorname{Floor}_i(Q,m_i,N)=
   \mathbf 1\{C_i(\mathfrak M_i(v,c;Q),P_i,N)\succeq c_i^{\min}\}\). Reachable states count
   as capabilities only if they are meaningful, exercisable, and socially available.
   An option the agent cannot understand, afford, safely choose, or have recognized is
   not yet a capability.

5. **Non-externalization:** \(\operatorname{NonExt}(Q,m,N)\) holds only when the
   joint model includes every causally affected standpoint and no loss is hidden by
   marginalization or failed semantic translation. The comparison is between full
   joint distributions interpreted across affected phenotypes, not one focal agent's
   expected values.

6. **Non-domination:**
   \(\operatorname{NonDom}(Q,m^{\mathrm{sh}},N)=
   D_N(\{Q^{do(\pi_j)}\}_j,m^{\mathrm{sh}})\). Intervention distributions identify
   asymmetric control; shared semantics identifies which changes concern authority,
   voice, exit, dependence, and arbitrariness. Influence becomes domination only under
   this relational interpretation.

7. **Public or affected-party justifiability:**
   \(\operatorname{Just}(Q,m_{1:n},m^{\mathrm{sh}},N)\) holds when reasons about
   expected consequences are translatable across affected standpoints, relevant
   objections survive gluing, and the declared procedure licenses the result. Shared
   semantics enables justification but does not guarantee it.

8. **Contestability:** \(\operatorname{Contest}(Q,m,N)\) is the probability that a
   semantically recognized challenge produces an institutional review or revision
   state. The challenger and institution must coordinate on what the challenge means,
   and the joint distribution must contain feasible revision paths with bounded cost
   and retaliation.

9. **Epistemic responsiveness:**
   \(\operatorname{Resp}(Q,m,N)=R_N(m^{\mathrm{sh}}_{t+1}-m^{\mathrm{sh}}_t,
   Q_{t+1}-Q_t)\). Responsiveness requires appropriate revision of both empirical
   beliefs and shared interpretations when consequences disconfirm them. Updating
   probabilities inside a fixed but exclusionary vocabulary is insufficient.

10. **Repairability and reversibility:** \(\operatorname{Repair}_i(Q,m_i,
    m^{\mathrm{sh}},N)\) holds when feasible joint trajectories reach conditions the
    harmed party can recognize as adequate repair, under a mutually processable
    account of the harm and future relation. Restoring only the controller's preferred
    state does not qualify.

11. **Plural improvement:** \(\operatorname{Improve}(Q^{\boldsymbol\pi},
    Q^{\boldsymbol\pi_0},m_{1:n},N)\) holds when interpreted consequence vectors
    improve under each warranted phenotype-relative ordering, subject to floors and
    without forcing incomparable meanings into an unlicensed scalar. Shared language
    enables comparison; it does not erase plural goods.

12. **Legitimate selection:** \(\operatorname{LegitSelect}(\boldsymbol\pi,Q,
    m_{1:n},m^{\mathrm{sh}},N)\) holds when selection is from the admissible joint set
    through a procedure whose roles, reasons, and consequences are mutually
    processable and whose allocation of voice and authority satisfies \(N\).
    Coordination or consensus alone is not legitimacy.

Virtue is tested as a slow hierarchical latent regime over this construction. It
changes distributions over preferences, likelihood and transition beliefs, policy
priors, precision, and temporal depth. Calling it good asserts that its realizations
repeatedly satisfy the semantically lifted moral predicate; goodness is not inferred
from metastability, evidence, deontic conformity, or shared uptake.

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

For agent \(i\), marginalize the joint posterior predictive distribution under joint
policy \(\boldsymbol\pi\):

\[
Q^{\boldsymbol\pi}_{i,e}
=q_M(s_{i,0:T},o_{i,1:T}\mid\boldsymbol\pi,e,P_i).
\]

Evaluate a vector rather than a scalar:

\[
\mathbf J_i(\boldsymbol\pi,e)=
(V_i,R_i,\operatorname{Emp}_i,E_i,L_i),
\]

where \(V_i\) is intervention-tested viability margin, \(R_i\) allostatic recovery,
\(\operatorname{Emp}_i\) viability-constrained empowerment over meaningful options,
\(E_i\) epistemic access or calibration, and \(L_i\) capacity for learning and
development. These quantities are non-equivalent and remain separately reported.

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

Let \(\mathcal U\) be a declared joint class of environments, model structures,
parameters, and semantic mappings. Require

\[
\inf_{(e,M,\theta,I)\in\mathcal U}
\Pr_{q_M^\pi}
[\mathbf J_i(\pi,e)\succeq\mathbf J_i^{\min}]
\ge 1-\epsilon_i,
\]

or use a declared distributionally robust free-energy objective. EFE's likelihood
entropy or “ambiguity” term is not uncertainty over alternative generative models.

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

A virtue is a reliably good function of socially processable meaning: a
context-sensitive organization of perception, affect, inference, action, and revision
whose identity depends on the expected relational consequences it makes salient and
on the shared semantic practices through which those consequences are interpreted.
It is not a fixed action, a merely stable trait, or an uninterpreted parameter region.

### Active-inference representation

Let \(\mathcal V_t\) be a historically stabilized but revisable social-semantic schema,
and let \(\Theta_{i,V}\) be the corresponding region of slow generative-model
parameters in agent \(i\), governing preferences, salience, transition beliefs, policy
priors, precision, temporal depth, models of others, and learning. A local realization
is a mapping from shared schema, context, phenotype, and expected joint consequences:

\[
R_{c,i}\!:
(\mathcal V_t,\Theta_{i,V},P_i,Q^{\boldsymbol\pi}_e)
\longmapsto
m_{i,c}^{V}(\boldsymbol\pi).
\]

Candidate local meanings must be mutually processable on morally relevant overlaps,
but they need not be identical. A candidate counts as a successful realization of the
virtue only if (i) it is licensed by this shared semantic structure and (ii) the joint
policy it organizes passes the moral-goodness predicate below. This makes goodness
constitutive of virtue while allowing one virtue concept to generate different
pragmatic meanings and actions across phenotypes, relationships, and contexts.

### Operational test

Fit hierarchical multi-agent models and test whether the combination of (a) expected
joint consequences, (b) learned semantic alignment maps, and (c) a shared slow schema
predicts context-appropriate policy profiles, participants' virtue classifications,
moral-category outcomes, perturbation recovery, and revision better than trait-only,
situation-only, private-consequence-only, and unconstrained active-inference models.

### Non-equivalence warning

A stable prior, deep attractor, precise policy prior, low-EFE habit, or culturally
shared label is not a virtue unless its situated joint realizations satisfy the
independent moral-goodness criteria. Nor does semantic agreement establish goodness:
a community can share a coherent interpretation of an exploitative practice.

### Status

**Project-specific synthesis and empirical hypothesis.**

## 14. Consolidated formal predicate

For active-inference model \(M\), learned semantic parameters \(\phi_{1:n}\),
phenotype family \(P_I\), environment class \(\mathcal E\), shared invariant
specification \(\mathcal I\), normative interpretation \(N\), and baseline joint
policy \(\boldsymbol\pi_0\):

\[
\operatorname{MG}(\boldsymbol\pi\mid
M,\phi_{1:n},P_I,\mathcal E,\mathcal I,N,\boldsymbol\pi_0)
\]

holds exactly when:

\[
\begin{aligned}
&\operatorname{StandingScopeComplete}(Q,m,N)\\
&\land\operatorname{SemanticCompatibility}(z_v,m_{1:n},\mathcal I)\\
&\land\operatorname{RobustFlourishing}(Q,m,N)\\
&\land\operatorname{CapabilityFloors}(Q,m,N)\\
&\land\operatorname{NonExternalizing}(Q,Q_0,m,N)\\
&\land\operatorname{NonDominating}(Q,m^{\mathrm{sh}},N)\\
&\land\operatorname{Justifiable}(Q,m_{1:n},m^{\mathrm{sh}},N)\\
&\land\operatorname{Contestable}(Q,m,N)\\
&\land\operatorname{EpistemicallyResponsive}(Q,m,N)\\
&\land\operatorname{Repairable}(Q,m,N)\\
&\land\operatorname{PluralImprovement}(Q,Q_0,m,N)\\
&\land\operatorname{LegitimatelySelected}(\boldsymbol\pi,Q,m,N),
\end{aligned}
\]

where \(Q=Q_e^{\boldsymbol\pi}\),
\(Q_0=Q_e^{\boldsymbol\pi_0}\),
\(m_i=\mathfrak M_i(v,c;Q,\phi_i)\), and \(m^{\mathrm{sh}}\) is a fused overlap
projection used only when shared-sign evidence, held-out processability, and sheaf
consistency pass their declared criteria. The semantic-compatibility conjunct means
that an interpersonal claim is empirically translatable on the tested overlaps; it is
not itself evidence that the claim is morally good.

The joint posterior predictive distribution generated by active inference supplies
the expected relational consequences. Learned shared signs, local semantic-pragmatic
profiles, and held-out compatibility diagnostics supply evidence of shared semantic
content. The normative theory supplies the moral
invariants, thresholds, protected dimensions, and legitimate decision rules. None of
these three layers can be silently substituted for another.

## 15. Order of implementation

The categories should be formalized in this order:

1. explicit joint policies, joint trajectory distributions, and affected-agent scope;
2. learned shared signs and agent-relative semantic-pragmatic profiles;
3. held-out processability and sheaf consistency on declared overlaps;
4. phenotype-relative flourishing and robustness;
5. capability floors and non-externalization;
6. causal non-domination;
7. epistemic responsiveness and contestability;
8. repair reachability;
9. plural improvement;
10. public justification and legitimate selection; and
11. virtue-schema identification across contexts.

The first six admit increasingly rich computational models. Public justification and
legitimate selection require formal procedures but cannot be reduced to posterior
prediction. This ordering prevents us from building a precise optimizer around an
ambiguous moral target.
