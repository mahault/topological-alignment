# Supported Replacements for Unsupported Constructs

## Purpose

This ledger replaces the constructs that the literature audit classified as
unsupported, project-specific, or rejected. A replacement is accepted only when it
has:

1. an explicit object in a generative or causal model;
2. an equation or executable estimator;
3. either primary-literature support or an identified implementation in a sibling
   project; and
4. a stated non-equivalence: what the quantity does **not** establish.

The central correction is that there is no single free-energy scalar for meaning,
cooperation, virtue, or moral goodness. The defensible model is a typed collection of
inferential, pragmatic, causal, and normative objects.

## Replacement summary

| Unsupported or rejected construct | Replacement | Status |
|---|---|---|
| VFE discovers the socially or morally correct ontology | Bayesian structure/model selection plus held-out calibration | established inference method; correctness remains external |
| The full meaning of a virtue is one shared latent variable | an agent-indexed semantic-pragmatic profile, connected through a learned shared sign | literature-backed components; composite is a project operationalization |
| VFE automatically learns a sheaf | shared-latent or federated learning first; processability and sheaf consistency second | established learning plus project diagnostics |
| An unspecified EFE residual | one fixed root EFE and only decompositions derived from it | established correction |
| One canonical joint EFE defines cooperation | agent-local EFE vector, free-energy equilibrium, and a declared joint counterfactual only when commensurable | literature-backed replacement |
| Total correlation defines cooperation | outcome achievement, directed processability, reciprocal readability, causal contribution, and PID/interaction diagnostics | established ingredients; no scalar sufficiency claim |
| A scalar joint-EFE “surplus” | per-agent matched-coupling contrasts; a social gap only under a declared common objective | project adaptation with explicit admissibility conditions |
| Virtue is literally an EFE term or an untyped “higher-order regime” | a slow hierarchical latent regime over model parameters, compared against simpler models by evidence | testable project model grounded in deep active inference |
| Viability-relevant semantic information supplies moral goodness | causal viability value plus allostatic recovery and constrained empowerment | established descriptive quantities; moral standing remains normative |

## 1. Ontology: replace “VFE finds the right categories” with model selection

For candidate model structures \(m\in\mathcal M\), optimize each variational
posterior and compare approximate model evidence:

\[
F_m^*=\min_{q_m}F[q_m;m],
\qquad
q(m\mid D)\propto p(m)\exp(-F_m^*).
\]

The selected structure must also pass held-out posterior-predictive checks:

\[
S_{\mathrm{pred}}(m)
=-\sum_{(o,u)\in D_{\mathrm{test}}}
\log p_m(o,u\mid D_{\mathrm{train}}),
\]

together with calibration and intervention tests. This is supported by active-
inference work on [structure learning and concept formation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9662737/)
and by [federated inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/).

**What this replaces:** the claim that unconstrained VFE discovers the correct
social or moral ontology.

**What it does not prove:** low free energy establishes predictive adequacy relative
to a model class, not truth outside that class and not moral correctness. Competing
models should remain in a posterior ensemble when evidence does not identify one.

## 2. Meaning: replace one latent “essence” with a semantic-pragmatic profile

Let \(v\) be a public sign such as *courage*, \(z_v\) a learned shared sign variable,
and \(c\) a context. Define the operational meaning available to agent \(i\) as

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

The entries respectively encode anticipated consequences, policies made available,
learned deontic cue-to-policy mappings, parameter beliefs, and precision. Expected
consequences are therefore constitutive of the sign's pragmatic role, but are not
declared to exhaust meaning.

This replacement combines three literature-supported objects:

- decentralized posterior inference over a shared latent sign in the
  [Recursive Metropolis-Hastings Naming Game](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1229127/full);
- shared forward models and narratives in
  [active-inference accounts of language](https://pmc.ncbi.nlm.nih.gov/articles/PMC7758713/); and
- the learned likelihood \(P(\pi\mid o^d,c)\) for deontic cues in the
  [DEEP model of social expectations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6452780/).

Two uses count as pragmatically equivalent only relative to a declared test family
\(\mathcal T\):

\[
\mathfrak M_i(v,c)\simeq_{\mathcal T}\mathfrak M_j(v,c')
\quad\Longleftrightarrow\quad
d_{\mathcal T}
\bigl(\mathfrak M_i(v,c),\mathfrak M_j(v,c')\bigr)\le\varepsilon.
\]

This is a defeasible operational equivalence, not a metaphysical identity criterion
for meaning.

## 3. Shared semantics: learn signs first, diagnose compatibility second

The primary learning mechanism is a shared-latent or federated model:

\[
p(z,\theta_{1:n},o_{1:n},u_{1:n})
=p(z)\prod_i p(\theta_i)p_i(o_i,u_i\mid z,\theta_i).
\]

Agents can approximate \(p(z\mid o_{1:n},u_{1:n})\) by a naming game, federated
message passing, or centralized variational inference. This supplies a learned sign
and agent-specific likelihood mappings without requiring identical private states.

After learning, fit one row-stochastic channel \(T_{ij}\) across the whole predictive
horizon:

\[
r_{ij}
=\min_{T_{ij}}
\frac1H\sum_{t=1}^{H}
D_{\mathrm{KL}}
\left(\Pi_{j,t}\middle\|\Pi_{i,t}T_{ij}\right),
\]

\[
I_{\mathrm{time}}(\Pi_j)
=\frac1H\sum_t
D_{\mathrm{KL}}(\Pi_{j,t}\|\bar\Pi_j),
\qquad
\operatorname{Proc}_{i\to j}
=\max\left(0,1-\frac{r_{ij}}{I_{\mathrm{time}}(\Pi_j)}\right).
\]

This is implemented in the sibling project at
`../shared-protention-alignment/core/morphism.py`. Its reliability flag rejects
nearly constant target streams for which reconstruction is vacuous. Mutual
processability is the minimum of the reliable directed scores, not their correlation.

When multiple local semantic spaces must be composed, use the cellular-sheaf
coboundary only as a compatibility diagnostic:

\[
(\delta x)_e=R_{ve}x_v-R_{ue}x_u,
\qquad
\rho_{\mathrm{sheaf}}(x)=\|\delta x\|,
\qquad
H^0=\ker\delta.
\]

The implementation is at
`../shared-protention-alignment/sheaf/cellular_sheaf.py`. Unless \(R_{ue}\) and
\(R_{ve}\) are parameters of an explicit overlap likelihood, this sheaf score is a
post-inference diagnostic and not a term learned automatically by ordinary VFE.

## 4. EFE: fix the root before naming components

Use one declared root functional, factorization, preference representation, and sign
convention. For example:

\[
G_i(\pi)
=\mathbb E_{Q_i(o,s\mid\pi)}
\left[\log Q_i(s\mid\pi)-\log P_i(o,s)\right].
\]

Only then derive a pragmatic-minus-epistemic or risk-plus-ambiguity decomposition
under its required assumptions. No generic residual is permitted. This follows the
analysis in [Millidge, Tschantz, and Buckley](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy)
and the [discrete active-inference synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/).

## 5. Cooperation: replace a scalar label with a diagnostic vector

For a declared joint outcome variable \(Y\), construct

\[
\mathbf C(\boldsymbol\pi)=
\left(
A_Y,
\operatorname{Proc}_{\leftrightarrow},
\operatorname{Read}_{\leftrightarrow},
\operatorname{Syn}_Y,
\operatorname{CIF}_{1:n\to Y},
\Delta^{\mathrm{cpl}}_{1:n}
\right).
\]

The components have different jobs:

- \(A_Y=Q(Y\in Y^*\mid\boldsymbol\pi)\) is predicted joint-task attainment, where
  the interpretation of \(Y^*\) is itself checked through the learned semantics.
- \(\operatorname{Proc}_{\leftrightarrow}=\min_{i\ne j}
  \operatorname{Proc}_{i\to j}\) measures whether anticipations translate across
  frames.
- \(\operatorname{Read}_{\leftrightarrow}=\min_{i\ne j}\operatorname{CoD}_{i\to j}\)
  measures reciprocal model-based readability. The sibling project
  `../externalization-horizon/paper/main.tex` derives this for linear-Gaussian
  observers and recommends coefficient of determination rather than squared
  correlation because scale and bias errors matter.
- \(\operatorname{Syn}_Y\) is a declared PID or cooperative-game interaction term
  for outcome-relevant complementarity. The PID definition must be named because
  PID is not unique.
- \(\operatorname{CIF}_{i\to Y}\) is intervention-based causal information flow,
  following [Ay and Polani](https://doi.org/10.1142/S0219525908001465), rather than
  transfer entropy interpreted causally.
- \(\Delta^{\mathrm{cpl}}_{1:n}\) is the per-agent coupling contrast defined below.

This vector is evidence for **cooperative coordination** when a preregistered external
criterion says which components and thresholds matter. It is deliberately not a
definition of moral cooperation. Coordinated predation or oppression can score well
on every descriptive component; admissibility must be tested separately.

Total correlation remains in the result ledger only as a dependence baseline:

\[
\mathcal T_Q
=D_{\mathrm{KL}}
\left(Q(\pi_{1:n})\middle\|\prod_iQ(\pi_i)\right).
\]

It is never used as a cooperation gate.

## 6. Joint EFE: keep agent-local contrasts unless aggregation is justified

For each agent, remove the same declared coupling factor \(C\) by intervention inside
the same generative model:

\[
\Delta_i^{\mathrm{cpl}}(\boldsymbol\pi)
=G_i^{do(C=0)}(\boldsymbol\pi)-G_i(\boldsymbol\pi).
\]

The vector

\[
\boldsymbol\Delta^{\mathrm{cpl}}
=(\Delta_1^{\mathrm{cpl}},\ldots,\Delta_n^{\mathrm{cpl}})
\]

preserves disagreement and scale differences. A positive entry means only that the
declared coupling lowers that agent's EFE under the matched model. It does not mean
that the coupling is cooperative, fair, or good.

The [Free-Energy Equilibria](https://openreview.net/forum?id=4Ft7DcrjdO) framework
provides the appropriate strategic baseline: compare a decentralized free-energy
equilibrium with joint free-energy minimization. But a scalar “cooperation gap” is
admissible here only if a common trajectory space and an explicitly justified social
functional \(W(G_1,\ldots,G_n)\) have been supplied. Otherwise report the EFE vector
and its Pareto frontier.

## 7. Virtue: use a slow hierarchical model and compare it with alternatives

Replace the untyped phrase “higher-order regime” with a latent regime \(r_t\) whose
dynamics are slower than situated state inference:

\[
p(r_{1:T},\theta_{1:T},s_{1:T},o_{1:T})
=p(r_1)\prod_t
p(r_t\mid r_{t-1})
p(\theta_t\mid r_t)
p(s_t,o_t\mid s_{t-1},\theta_t,c_t).
\]

Here \(r_t\) changes distributions over ordinary active-inference parameters:
preferences \(C\), likelihood/salience mappings \(A\), transitions \(B\), policy
priors or habits \(D\), precision \(\gamma\), and temporal depth \(H\). Deontic values
are learned policy-cue mappings, not moral authority.

A candidate virtue model \(M_V\) earns support only if it predicts held-out context
transformations better than trait-only, situation-only, and action-frequency models:

\[
\log BF_{V,k}\approx F_k^*-F_V^*.
\]

It must also preserve the relevant semantic-pragmatic organization while allowing
actions to change with context, and it must pass the moral-goodness predicate. This
operationalizes the ecological account of virtue as metastable optimal grip without
claiming that the literature has identified virtue with an EFE term.

## 8. Phenotype-relative good: combine viability, recovery, and capability

Use three non-equivalent descriptive quantities:

1. **Causal viability value of information.** Preserve the intervention-based notion
   of semantic information from
   [Kolchinsky and Wolpert](https://pmc.ncbi.nlm.nih.gov/articles/PMC6227811/).
2. **Allostatic recovery.** Estimate probability and time of return to a phenotype's
   viable region after perturbation. Active-inference models of
   [resilience phenotypes](https://pmc.ncbi.nlm.nih.gov/articles/PMC12098587/) support
   recovery and accommodation as distinct from mere persistence.
3. **Constrained empowerment.** Measure attainable control as channel capacity,

   \[
   \operatorname{Emp}_H(x)
   =\max_{q(a_{t:t+H})}I(A_{t:t+H};O_{t+H}\mid x),
   \]

   subject to viability and risk constraints. The relation between empowerment and
   generalized free energy is developed in
   [Kiefer's constrained-entropy account](https://pmc.ncbi.nlm.nih.gov/articles/PMC12025677/).

Robustness across model uncertainty should use a model ensemble or a robust objective,
not EFE's likelihood-entropy term mislabeled as model uncertainty. A directly relevant
extension is the
[distributionally robust free-energy principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC12820166/),
which minimizes worst-case free energy over a declared ambiguity set.

These quantities establish evidence for “good for phenotype \(P\)” under a declared
environment class. They do not establish moral standing or justify sacrificing one
phenotype for another.

## 9. Replacement ledger for every moral predicate

Some predicates have endogenous empirical grounds but no active-inference derivation
of their normative authority. In those cases the correct replacement for a bolted-on
label is an explicit normative input applied to quantities generated by the model.

| Predicate | Endogenous model quantity | Source or implementation | Irreducible normative input |
|---|---|---|---|
| Affected centre / standing | posterior over candidate affected systems and intervention effects on their viability variables | causal graph plus uncertain-standing benchmark in `benchmarks/uncertain_standing_model.py` | which kinds of vulnerability confer standing |
| Flourishing | vector of viability, allostatic recovery, learning, relationship, and constrained-empowerment trajectories | Kolchinsky-Wolpert; active-inference allostasis | which functionings matter and their thresholds |
| Robustness | worst-case or posterior-ensemble performance across \(m,e,\theta\) | distributionally robust FEP | ambiguity set and precaution rule |
| Capability floors | constrained empowerment plus reachability of protected outcome sets | empowerment literature; finite reachability kernel | protected capabilities and minimum levels |
| Non-externalization | \(P(Y_i\mid do(\pi_j))\), delayed causal effects, and causal information flow across system boundaries | Ay-Polani; `../externalization-horizon` | which burdens may not be displaced |
| Non-domination | asymmetric intervention power over another's transitions, observations, exits, and veto channels | causal-control and empowerment quantities | whether power is arbitrary, authorized, reciprocal, and contestable |
| Public justification | information access, reason-message likelihoods, belief revision, and common-knowledge uncertainty | federated inference and recursive other-models | what counts as a public reason and fair inclusion |
| Contestability | reachable challenge policies, expected cost/retaliation, uptake probability, and posterior change under challenge evidence | ordinary transition, preference, and parameter-learning terms | adequate authority, access, and remedy |
| Epistemic responsiveness | held-out predictive score, calibration, information gain, Bayes factor, and innovation consistency | VFE/model selection; `../externalization-horizon` | relevant evidence and appropriate precision |
| Repair | probability, time, and control cost of restoring protected state sets and changing recurrence-causing parameters | finite reachability and allostatic recovery | adequate restoration, compensation, and decision authority |
| Plural improvement | vector \(\Delta J_i\), robust partial order, and Pareto set | agent-local EFE/outcome predictions | interpersonal comparison and treatment of incomparability |
| Legitimate selection | predicted consequences of candidate procedures, participation, error, and capture | institutional variables in the joint generative model | authority, equality, inclusion, and acceptable procedure |

Active inference therefore supplies the **evidence-generating process** for each row.
It does not turn the final column into a theorem of VFE or EFE.

## 10. Acceptance and falsification rules

A future claim may use these replacements only if:

1. every EFE contrast uses the same root definition and matched variables;
2. semantic maps are fitted on training data and evaluated on held-out contexts;
3. low-information processability estimates are marked unreliable;
4. transfer entropy is not described as causal without an identified intervention;
5. the selected PID or interaction decomposition is named;
6. cooperation labels are withheld from model fitting and used only for blinded
   validation;
7. agent-local effects remain a vector unless commensurability is defended;
8. normative thresholds and standing assumptions appear in the result ledger; and
9. simpler action-frequency, correlation, trait, and situation baselines are run.

The replacement programme fails if the richer quantities do not improve held-out
prediction or intervention discrimination over those simpler baselines.
