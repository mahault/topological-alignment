# A Normatively Constrained Active-Inference Interpretation

## Status

**2026-07-19 scope correction:** this document formalizes a downstream admissibility
and audit layer. It does not define the phenomenological origin of goodness. The
upstream hypothesis—felt goodness as a fallible estimate of multi-scale enablingness,
and virtue as the slow regime that governs its calibration—is specified in
[Multi-Scale Goodness and Virtue Attractors](MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md).
The predicates below should be read as tests for captured or truncated heuristics,
not as labels bolted onto VFE/EFE and renamed feelings.

This document gives a formal interpretation of the paradigm relative to active
inference. It distinguishes three claims:

1. active inference can model what an agent expects and prefers;
2. it can model how an abstract virtue regime changes situated inference and action;
3. it cannot derive moral standing or the authority of moral constraints from free
   energy minimization alone.

The normative bridge supplies an admissibility proposal that becomes a formal predicate over
policies and their predicted trajectory distributions. Active inference supplies the
distribution to which that predicate is applied and a process theory of how agents
act, learn, and revise within—or violate—the proposed morally admissible region.

The correct claim is therefore not merely that morality can externally “filter”
active inference. Given the adopted audit criteria, moral admissibility is formally
interpretable relative to an active-inference model. What active inference does not do
is establish that those criteria are uniquely correct or that an agent's felt
goodness is calibrated to them.

## 1. Multi-agent generative model

Let there be affected agents \(i\in I\), latent relational state

\[
s_t=(x_{1,t},\ldots,x_{n,t},e_t,r_t,n_t),
\]

and observations \(o_{i,t}\) available to each agent. Here \(x_i\) is embodied agent
state, \(e\) environment state, \(r\) the distribution of dependency and control, and
\(n\) the normative and institutional niche.

Agent \(i\)'s generative model is

\[
p_i(o_{1:T},s_{1:T},\pi_i,\theta_i)
=p_i(\theta_i)p_i(\pi_i\mid\theta_i)
\prod_{t=1}^{T}
p_i(o_{i,t}\mid s_t,\theta_i)
p_i(s_t\mid s_{t-1},\pi_i,\pi_{-i},\theta_i).
\]

The inclusion of \(\pi_{-i}\) and \(r_t\) is essential. Treating other agents as mere
environmental noise would hide strategic dependence, coercion, assistance, and
externalized costs.

Approximate inference minimizes variational free energy

\[
F_i[q_i]
=
\mathbb E_{q_i(s,\theta)}
[\log q_i(s,\theta)-\log p_i(o,s,\theta)].
\]

This is an epistemic objective: it scores an approximate posterior relative to a
declared generative model. It is not a moral-value function.

### Endogenous semantics and joint organization

Semantic content and cooperation cannot enter this model as labels attached after
inference. Extend the latent state with a shared but uncertain semantic variable
\(z_t\), communicative acts \(u_{i,t}\), and learnable agent-specific interpretation
parameters \(\phi_i\):

\[
\begin{aligned}
p(o,u,s,z,\boldsymbol\pi,\theta,\phi)
={}&p(\theta,\phi)p(\boldsymbol\pi\mid\theta)
\prod_t p(s_t,z_t\mid s_{t-1},z_{t-1},\boldsymbol\pi,\theta)\\
&\times\prod_i
p_i(o_{i,t},u_{-i,t}\mid s_t,z_t,\phi_i).
\end{aligned}
\]

Here \(z_t\) is not a predefined word meaning. It is a learned shared sign whose
posterior helps predict agents' observations, signals, and anticipated consequences.
Each \(\phi_i\) specifies how that sign appears in agent \(i\)'s sensorium and action
repertoire. Naming-game and federated-inference results support this narrower claim;
they do not establish that one latent variable exhausts the meaning of a thick moral
concept.

The coupled variational objective is the ordinary free energy of this joint model:

\[
F[q,\phi]
=\mathbb E_q[log q(s,z,\theta,\phi)-log p(o,u,s,z,\theta,\phi)].
\]

Under a structured mean-field approximation, this objective contains likelihood and
complexity terms for each local interpretation together with dependencies induced by
their common latent sign. Gradient flow

\[
\dot q=-\Gamma_q\nabla_qF,
\qquad
\dot\phi=-\Gamma_\phi\nabla_\phi F
\]

therefore performs perceptual inference and parameter learning within the declared
model class. It does not guarantee a socially or morally correct ontology. Candidate
structures are compared by approximate evidence and held-out predictive calibration:

\[
q(m\mid D)\propto p(m)\exp[-F_m^*],
\qquad F_m^*=\min_{q_m}F[q_m;m].
\]

For a sign \(v\) and context \(c\), the operational semantic-pragmatic profile is

\[
\mathfrak M_i(v,c)=
\left(
Q_i(s_{1:H},o_{1:H}\mid\pi,z_v,c),
Q_i(\pi\mid z_v,c),
P_i(\pi\mid o^d,c),
Q_i(\theta_i\mid z_v,c),
\gamma_i(c)
\right).
\]

It includes prospective consequences, induced policies, learned deontic cue mappings,
parameter beliefs, and precision. This is an operational profile, not a claim to have
reduced all meaning to consequences.

Translation and sheaf objects are second-stage diagnostics. Fit a fixed channel
\(T_{ij}\) between held-out protention streams and report normalized directed
processability; then use \(\|\delta x\|\) to test compatibility of local sections.
Unless the restriction maps occur as parameters in an explicit overlap likelihood,
the sheaf residual is not a term automatically learned by VFE. The equations and
implementations are indexed in `SUPPORTED_REPLACEMENTS_LEDGER.md`.

Joint policies are evaluated from a fixed root expected-free-energy definition:

\[
G(\boldsymbol\pi)
=\mathbb E_{Q(o,s,z\mid\boldsymbol\pi)}
\left[
\log Q(s,z\mid\boldsymbol\pi)-
\log P(o,s,z)
\right].
\]

For a declared factorization this can expose, without inventing new labels, the
extrinsic-value minus epistemic-value decomposition:

\[
G(\boldsymbol\pi)
=\underbrace{\mathbb E_Q[-\log P_C(o)]}_{\text{pragmatic cost}}
-\underbrace{I_Q(s,z;o\mid\boldsymbol\pi)}_{\text{epistemic value}}.
\]

Under the corresponding state-preference factorization, the same fixed root may admit
the risk-plus-ambiguity decomposition

\[
G(\boldsymbol\pi)
=\underbrace{D_{\mathrm{KL}}[Q(s,z\mid\boldsymbol\pi)\|P_C(s,z)]}_{\text{risk}}
+\underbrace{\mathbb E_{Q(s,z\mid\boldsymbol\pi)}
[H(P(o\mid s,z))]}_{\text{ambiguity}}.
\]

These displays are not unconditional identities across all EFE formulations. The root
definition, preference representation, factorization, and approximations must be
fixed. Any approximation residual must be derived rather than introduced as a
canonical term. The shared latent \(z\) makes social epistemic value explicit: actions
can be valuable because they disambiguate intentions and consequences for more than
one agent.

For a coupling intervention defined *inside the same model*, with identical variables,
horizon, base measure, and agent-specific preference representation, retain separate
agent-local EFE contrasts:

\[
\Delta_i^{\mathrm{cpl}}(\boldsymbol\pi)
=G_i^{do(C=0)}(\boldsymbol\pi)-G_i(\boldsymbol\pi),
\qquad
\boldsymbol\Delta^{\mathrm{cpl}}
=(\Delta_1^{\mathrm{cpl}},\ldots,\Delta_n^{\mathrm{cpl}}).
\]

Positive \(\Delta_i^{\mathrm{cpl}}\) means only that retaining the declared coupling
reduces agent \(i\)'s EFE relative to that intervention. The vector is not summed
unless a common trajectory space and justified social functional make the components
commensurable. Posterior dependence can additionally be described by total
correlation

\[
\mathcal T_Q(\boldsymbol\pi)
=D_{\mathrm{KL}}\!\left(
Q(\pi_1,\ldots,\pi_n)\;\middle\|\;
\prod_i Q(\pi_i)
\right)
\]

which measures departure from factorization, not cooperation or synergy. It is neither
necessary nor sufficient: deterministic coordination can have \(\mathcal T_Q=0\),
while common causes, coercion, or redundant imitation can make it positive.

Candidate cooperative coordination is therefore represented by a diagnostic vector,
not a replacement scalar:

\[
\mathbf C(\boldsymbol\pi)=
\left(
Q(Y\in Y^*\mid\boldsymbol\pi),
\operatorname{Proc}_{\leftrightarrow},
\operatorname{Read}_{\leftrightarrow},
\operatorname{Syn}_Y,
\operatorname{CIF}_{1:n\to Y},
\boldsymbol\Delta^{\mathrm{cpl}}
\right).
\]

These entries distinguish joint achievement, cross-frame semantic processability,
reciprocal model-based readability, outcome-relevant PID or interaction structure,
intervention-based causal contribution, and per-agent changes in EFE. Each is computed
from the generative or causal model, but none alone defines cooperation. A blinded
external validation criterion determines whether the vector tracks cooperative
coordination. Non-domination, capability-floor, and plural-improvement predicates then
determine whether that coordination is morally admissible.

Thus the framework has two sharply separated claims:

1. shared signs should be learned through explicit Bayesian or variational dynamics,
   while semantic compatibility and candidate cooperative organization are evaluated
   by distinct held-out diagnostics derived from those dynamics; and
2. whether the resulting organization is morally good remains a constrained
   normative judgment over its joint posterior-predictive consequences.

## 2. Phenotype-relative practical good

For each agent or moral patient \(i\), declare a phenotype model

\[
P_i=(K_i,N_i,A_i,T_i,D_i),
\]

where \(K_i\) is a viability region, \(N_i\) needs, \(A_i\) capabilities, \(T_i\)
relevant time horizons, and \(D_i\) dependencies.

The phenotype induces evidence-based but revisable preference structure \(C_i\), not
merely a report of current desire. Preferred-outcome probabilities can be represented
as

\[
p_i^C(o_{i,t}\mid P_i,\theta_i).
\]

Expected free energy for policy \(\pi_i\) may then be decomposed schematically as

\[
G_i(\pi_i)
=
\underbrace{
\mathbb E_q[-\log p_i^C(o_{i,\tau})]
}_{\text{expected negative log preference}}
-
\underbrace{
I_q(s_\tau;o_{i,\tau}\mid\pi_i)
}_{\text{epistemic value}},
\]

This schematic is the pragmatic-value minus epistemic-value form. It is not
interchangeable without qualification with the risk-plus-ambiguity form: the exact
equalities depend on the root EFE definition, generative-model factorization,
preference representation, and approximation assumptions. Those choices and the sign
convention must be fixed before implementation.

This gives a process interpretation of **good for phenotype \(i\)**: policies expected
to protect independently validated viability and capability dimensions while
maintaining epistemic access to relevant uncertainty. The empirical assessment should
keep three quantities distinct: intervention-tested viability value of information,
allostatic recovery after perturbation, and constrained empowerment or reachable
control. Robust claims must additionally range over a posterior model ensemble or a
declared distributional ambiguity set. None of these quantities yet gives moral
goodness.

## 3. Virtue as a slow hierarchical latent regime

An abstract virtue \(V\) is not one preferred observation, one action, or a new EFE
component. Represent the candidate construct as a latent regime \(r_t\) with a slower
transition timescale than situated state inference:

\[
p(r_{1:T},\theta_{1:T},s_{1:T},o_{1:T})
=p(r_1)\prod_t
p(r_t\mid r_{t-1})
p(\theta_t\mid r_t)
p(s_t,o_t\mid s_{t-1},\theta_t,c_t).
\]

The regime's support is a constrained region of model parameters:

\[
\Theta_V\subseteq
\Theta_C\times\Theta_A\times\Theta_B\times\Theta_D
\times\Theta_{\gamma}\times\Theta_H,
\]

covering:

- outcome preferences \(C\);
- likelihood and salience model \(A\);
- transition beliefs \(B\);
- policy priors \(D\);
- precision allocation \(\gamma\); and
- temporal depth and model structure \(H\).

Context \(c\) and phenotype \(P_i\) generate a local realization through

\[
R_{c,i}:\Theta_V\longrightarrow
q_i(s,\pi_i,\theta_i\mid o_i,c,P_i).
\]

Two agents can therefore instantiate the same abstract virtue while selecting
different actions. The empirical identity claim is not action equality but preservation
of a regulatory organization across context transformations.

Practical wisdom is the governance operation that updates model structure, precision,
stakeholder inclusion, temporal horizon, and policy evaluation when ordinary
first-order inference is inadequate:

\[
\mathcal W_i:
(q_i,c,P_i,\mathcal U_i)
\mapsto
(q'_i,\gamma'_i,H'_i,\Pi'_i).
\]

This remains a testable model family, not an established identification of
*phronesis* with precision control.

The virtue-regime interpretation is retained only if its approximate evidence and
held-out predictions beat trait-only, situation-only, and action-frequency models:

\[
\log BF_{V,k}\approx F_k^*-F_V^*.
\]

It must predict when action changes while the relevant semantic-pragmatic organization
persists, and it must fail the virtue label when the resulting policy violates the
moral-goodness predicate. Stability or predictive fit alone cannot make a regime
virtuous.

## 4. Moral admissibility as constraints on policy inference

Let \(\mathcal P\) be the joint-policy space. Define the morally admissible subset

\[
\mathcal P_{\mathrm{adm}}
=
\bigcap_{i\in I}
\left(
\mathcal V_i\cap\mathcal C_i
\right)
\cap\mathcal N\cap\mathcal J\cap\mathcal X\cap\mathcal R.
\]

The components are:

### Viability and recovery

\[
\mathcal V_i
=
\left\{
\pi:
\inf_{e\in\mathcal E}
\Pr_{q^\pi_e}
[s_{i,0:T}\in K_i\ \text{or recovers within }\tau_i]
\ge 1-\epsilon_i
\right\}.
\]

### Capability floors

For a vector of protected capabilities \(a_i(\pi,e)\),

\[
\mathcal C_i
=
\{\pi:a_i(\pi,e)\succeq a_i^{\min}
\quad\forall e\in\mathcal E\}.
\]

This is componentwise. One capability gain does not automatically compensate for
crossing another capability's floor.

### Non-domination

Let \(d_{ji}(\pi,e)\) measure the uncontrolled capacity of \(j\) to alter \(i\)'s
protected options, observations, or transition structure. Then

\[
\mathcal N
=
\{\pi:d_{ji}(\pi,e)\le d_{ji}^{\max}
\quad\forall i,j,e\}.
\]

This quantity cannot be identified with mutual information or empowerment alone. It
must distinguish control that is reciprocal, authorized, contestable, or institutionally
constrained from unilateral discretionary control.

### Justification, contestability, and repair

- \(\mathcal J\): relevant affected-agent models are included and decisions satisfy a
  declared public-justification procedure;
- \(\mathcal X\): voice, refusal, appeal, and model-challenge channels remain above
  declared thresholds; and
- \(\mathcal R\): the policy preserves feasible paths for reversal, compensation,
  restoration, or institutional update.

These sets contain normative judgments supplied by the bridge argument. Encoding them
does not prove them morally correct.

## 5. Policy selection

The model should not hide morality inside one agent's preference distribution. Use a
two-stage construction:

\[
\text{Stage 1:}\qquad
\Pi_{\mathrm{eligible}}
=\Pi\cap\mathcal P_{\mathrm{adm}},
\]

\[
\text{Stage 2:}\qquad
\pi^*\in
\operatorname{Nondominated}_{\pi\in\Pi_{\mathrm{eligible}}}
\left(
G_1(\pi),\ldots,G_n(\pi),
-\operatorname{Emp}_1(\pi),\ldots,-\operatorname{Emp}_n(\pi),
-Q(Y\in Y^*\mid\pi),
\rho(\pi)
\right),
\]

where each empowerment term is constrained by that affected party's viability and
capability floors, \(Y^*\) is a semantically validated joint achievement, and \(\rho\)
is residual risk or irreversibility. This avoids hiding a disputed common good inside
one “joint empowerment” scalar. A legitimate conflict procedure chooses among
nondominated eligible policies; the mathematics need not force a complete ordering.

This yields the formal distinction:

\[
\mathrm{FunctionallyGood}_i(\pi)
\not\Rightarrow
\mathrm{Admissible}(\pi),
\]

and

\[
\mathrm{MorallyGood}(\pi)
:=
\mathrm{Admissible}(\pi)
\land
\mathrm{PluralImprovement}(\pi)
\land
\mathrm{LegitimatelySelected}(\pi).
\]

The first conjunct protects standing; the second identifies positive improvement
rather than mere permissibility; the third addresses unresolved conflict among
incomparable legitimate goods.

### Direct policy predicate

For generative model \(M\), phenotype family \(P_I\), environment class
\(\mathcal E\), normative specification \(N\), and comparison baseline \(\pi_0\), define

\[
\operatorname{MG}(\pi\mid M,P_I,\mathcal E,N,\pi_0)=1
\]

exactly when the posterior predictive trajectory distribution under \(\pi\):

1. meets robust viability and recovery requirements for every standing-bearing
   affected phenotype;
2. preserves every protected capability floor;
3. contains no prohibited domination relation;
4. weakly improves the represented legitimate goods relative to \(\pi_0\), with at
   least one strict improvement and no represented affected party made worse;
5. satisfies the declared tests of justification, contestability, epistemic
   responsiveness, and repairability; and
6. is selected through the declared legitimate conflict procedure when eligible
   policies remain incomparable.

Moral evaluation is therefore conditional but computable:

\[
q_M(s,o\mid\pi)
\longmapsto
\operatorname{MG}(\pi\mid M,P_I,\mathcal E,N,\pi_0).
\]

Hard-constrained active-inference policy selection can then be written as

\[
q(\pi)\propto
\mathbf 1[\operatorname{Adm}(\pi)=1]
\exp[-\gamma G(\pi)],
\]

followed by legitimate selection among admissible, non-dominated policies. For
positive moral goodness rather than permissibility alone, require

\(\operatorname{MG}(\pi)=1\).

Equivalently, inadmissible policies may receive infinite normative cost, but the
indicator form better displays that protected floors are not ordinary preferences
that sufficiently large benefits may outweigh.

## 6. Active-inference interpretation of moral failure

The framework generates distinct failure modes:

| Failure | Active-inference interpretation | Why EFE alone misses it |
|---|---|---|
| Self-serving adaptation | accurate inference under preferences excluding others | preference satisfaction is agent-relative |
| Adaptive preference | learned preferences fit a coercive niche | low surprise can reflect oppression |
| Domination | one agent controls another's transitions or observations | aggregate predictability can increase |
| Epistemic exclusion | affected-party evidence is absent or down-weighted | the model can be internally coherent but wrong |
| Rigid virtue | excessive policy or model precision blocks revision | stable grip can look successful locally |
| Moral overload | no joint policy satisfies all protected floors | optimization cannot erase tragic conflict |
| Value lock-in | slow parameters cannot be contested or revised | convergence is not moral correctness |

This is the strongest reason to use active inference here: it can represent how a
purported virtue becomes pragmatically competent, rigid, self-sealing, dominating, or
corrigible. Its value is diagnostic and mechanistic, not foundationally moral.

## 7. Formal verification programme

### Finite deterministic kernel -- current

Already Lean-checked:

- policy closure implies finite-horizon viability;
- closure does not imply perturbation recovery;
- recovery plus closure implies post-recovery viability;
- one-environment superiority does not imply robust dominance; and
- robust focal benefit does not imply an affected-agent capability floor.

### Finite probabilistic model -- next

1. Represent finite state, observation, action, policy, and affected-agent spaces.
2. Define exact rational transition probabilities.
3. Define finite-horizon chance-constrained viability.
4. Prove monotonicity in risk tolerance and environment-set inclusion.
5. Construct countermodels in which minimizing one agent's EFE violates another's
   floor.
6. Prove that filtering by admissibility prevents those finite countermodels by
   construction, while clearly labeling the normative assumptions.

### Empirical model -- after identifiability checks

Fit and compare:

- action-frequency and trait baselines;
- situation-only and trait-by-situation models;
- unconstrained active inference;
- virtue-regime active inference; and
- normatively constrained multi-agent active inference.

The key evidence is held-out context prediction and recovery under perturbation, not
post-hoc fit.

## 8. What this formalization establishes

It establishes a coherent interpretation in which:

1. phenotype-relative good supplies agent-indexed pragmatic content;
2. virtue is a slow, abstract regime with context-dependent local realizations;
3. practical wisdom governs inference and model revision;
4. moral principles constrain the joint policy space rather than appearing magically
   from EFE minimization; and
5. moral goodness is admissible plural improvement under a legitimate conflict
   procedure.

It does not establish that the selected phenotype models, standing criterion,
capability floors, domination measure, or legitimacy procedure are correct. Those
remain independent empirical and normative obligations.
