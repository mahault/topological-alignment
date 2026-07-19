# Literature Audit: VFE, EFE, Meaning, Cooperation, and Moral Predicates

## Question

Can meaning, shared semantics, cooperation, virtue, and the other terms in the moral-
goodness definition be derived from variational-free-energy (VFE) and expected-free-
energy (EFE) dynamics rather than attached as labels?

## Short answer

Only in a qualified sense.

- VFE supplies a principled objective for posterior inference and parameter or
  structure learning under a specified generative model.
- EFE supplies policy-conditioned predictive distributions and policy scores under
  specified preferences and a fixed formulation.
- Shared latent symbols can emerge through decentralized Bayesian inference.
- Joint action can emerge through coupled inference, recursive models of other agents,
  shared generative models, or joint-policy search.
- None of these results supplies a generally accepted scalar definition of
  cooperation, virtue, domination, legitimacy, or moral goodness.
- Moral predicates can be computed from distributions produced by VFE/EFE dynamics,
  but their moral interpretation and authority require additional assumptions.

The strongest defensible programme is therefore **endogenous descriptive content plus
explicit normative interpretation**, not a derivation of morality from free-energy
minimization alone.

## Verdict vocabulary

- **Established:** directly supported by a published derivation or demonstrated model.
- **Adaptation:** a standard result transported into this project's joint model.
- **Project hypothesis:** mathematically specifiable but not established in the cited
  literature.
- **Rejected:** the earlier formulation does not follow or has a counterexample.

## 1. VFE as inference and learning

### Literature

Active inference distinguishes inference over hidden states, learning of model
parameters, and in some formulations structure learning. The discrete synthesis by
[Da Costa et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/) presents VFE as
model inversion. [Friston et al. on federated inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/)
explicitly model inference, learning, and selection by free-energy minimization across
agents. Learned generative state-space models have also been implemented rather than
hand specified ([Çatal et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7701292/)).

### Verdict

**Established:** posterior beliefs and generative-model parameters can be optimized by
VFE minimization.

**Not established:** an unconstrained VFE objective will discover the morally or
socially correct ontology. It learns whichever latent structure best trades accuracy
and complexity under the chosen model class, data, priors, and approximation.

## 2. Meaning as a shared latent variable

### Literature

The strongest computational precedent is the
[Recursive Metropolis–Hastings Naming Game](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1229127/full).
It proves that a language game can implement decentralized approximate Bayesian
inference over a latent sign shared by multiple agents. The
[Collective Predictive Coding hypothesis](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1353870/full)
generalizes this as society-level representation learning. Work on
[federated inference and belief sharing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/)
models language acquisition through learning compatible likelihood mappings.

Active-inference accounts of communication emphasize shared narratives, shared
generative models, and learned form–meaning relations
([Vasil et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7109408/);
[Friston et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7758713/)).

### Verdict

**Established:** shared signs and likelihood mappings can emerge through decentralized
Bayesian or variational learning.

**Project hypothesis:** the full meaning of a virtue is a learned latent variable over
expected relational consequences. Existing models usually establish reference,
categorization, narrative alignment, or communicative success—not thick moral meaning.

**Important qualification:** expected consequences contribute pragmatic and
inferential role, but do not exhaust meaning. Historical practice, embodiment,
attention, affect, and normative use must also enter the generative process if the
model is intended to represent virtue concepts.

## 3. Shared semantics without identical beliefs

### Literature

[Shared Protentions in Multi-Agent Active Inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11049075/)
uses polynomial-functor and sheaf/topos machinery to describe shared anticipatory
content across agents. Cellular discourse sheaves provide an independent mathematical
account of heterogeneous local representations agreeing through restriction maps
([Hansen and Ghrist](https://arxiv.org/abs/2005.12798)).

### Verdict

**Established separately:** sheaf consistency can represent agreement on overlaps
without equality of private states.

**Project hypothesis:** the restriction maps and gluing residual are learned as part
of the same VFE dynamics. The shared-protentions paper presents the categorical
framework, while discourse-sheaf diffusion has its own sheaf-Laplacian dynamics. The
literature does not prove that ordinary VFE minimization automatically yields the
required sheaf.

### Valid route to an endogenous term

The connection can be made explicit rather than asserted. Put discourse variables and
restriction maps inside the generative model. With Gaussian overlap likelihood

\[
p(y_e\mid z_u,z_v,R_{u e},R_{v e})
\propto
\exp\!\left[-\frac{1}{2\sigma_e^2}
\|R_{u e}z_u-R_{v e}z_v\|^2\right],
\]

the VFE accuracy term contains

\[
F_{\mathrm{overlap}}
=\frac12\sum_{e=(u,v)}
\mathbb E_q\!left[
\|R_{u e}z_u-R_{v e}z_v\|^2_{\Sigma_e^{-1}}
\right],
\]

which reduces to a weighted sheaf-Laplacian quadratic form in the linear-Gaussian
case. This is a **project derivation**. It ensures that semantic consistency is an
actual likelihood contribution to VFE rather than a post-hoc label. It does not show
that consistency is truth or goodness.

## 4. EFE decompositions

### Literature

[Millidge et al.](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy)
show that EFE admits different decompositions and analyze their assumptions. Under
the relevant factorization, risk plus ambiguity can equal extrinsic value minus
epistemic value. The discrete active-inference synthesis also presents standard
policy-selection equations and their assumptions
([Da Costa et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/)).

### Verdict

**Established:** pragmatic/extrinsic and epistemic terms can be exposed from a fixed
root EFE definition under stated factorization and approximation assumptions.

**Rejected:** adding an unspecified “formulation-dependent residual” and presenting it
as a general EFE decomposition. If an approximation creates a residual, it must be
derived for that approximation; it is not a canonical EFE term.

## 5. Multi-agent and joint active inference

### Literature

- [Interactive inference](https://iris.cnr.it/retrieve/16165904-8550-4afe-b529-268e20d27365/Interactive_Inference_A_Multi-Agent_Model_of_Cooperative_Joint_Actions.pdf)
  models agents making intentions legible during joint action.
- [Factorised Active Inference for Strategic Multi-Agent Interactions](https://arxiv.org/abs/2411.07362)
  gives agents explicit beliefs about other agents and studies VFE/EFE dynamics in
  general-sum games. Its ensemble EFE is a sum of expected agent EFEs and is not
  necessarily minimized at the aggregate level.
- [Free-Energy Equilibria](https://openreview.net/forum?id=4Ft7DcrjdO) contrasts
  decentralized equilibria with joint free-energy minimization, providing genuine
  precedent for a cooperation counterfactual.
- [Theory of Mind Using Active Inference](https://arxiv.org/abs/2508.00401) searches
  joint policy spaces recursively without assuming a shared generative model.
- [Federated inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/) shows how
  belief sharing can minimize joint free energy over agents.

### Verdict

**Established:** active-inference agents can model others, search joint policies,
communicate through action, and share beliefs.

**Not established:** there is one canonical “joint EFE” whose minimum defines
cooperation. Individual EFEs may use different preferences, scales, models, and
beliefs, so summation requires commensurability assumptions.

Most computational papers also do not solve the present semantic problem. Iterated
game models usually name actions `cooperate` and `defect` and encode their significance
through a supplied payoff matrix. Joint-action models normally supply the task or
goal. These are valid operational conventions, but they do not demonstrate the
bottom-up emergence of the concept *cooperation* from unlabeled joint consequences.
The proposed experiment must therefore withhold cooperation labels from inference and
use them, if at all, only for blinded external validation.

## 6. Total correlation of the joint policy posterior

The proposed quantity

\[
\mathcal T_Q
=D_{\mathrm{KL}}\!\left(Q(\pi_{1:n})\middle\|\prod_iQ(\pi_i)\right)
\]

is standard total correlation. It measures departure from factorization.

### Verdict

**Rejected as a necessary or sufficient condition for cooperation.**

- High total correlation can reflect coercion, collision avoidance, common causes, or
  redundant imitation.
- Perfect deterministic coordination can give a point-mass joint distribution whose
  product of marginals is the same point mass, hence \(\mathcal T_Q=0\).
- Total correlation does not separate redundant from synergistic information.

It remains usable as an exploratory dependence statistic, not as
\(\operatorname{Coop}\).

## 7. Synergy and complementarity

### Literature

Partial information decomposition (PID) separates redundant, unique, and synergistic
information. The literature also warns that PID is not uniquely defined; different
redundancy axioms yield different decompositions. See
[Lizier, Flecker, and Williams](https://arxiv.org/abs/1303.3440),
[Barrett](https://arxiv.org/abs/1411.2832), and the cooperative-game decomposition of
[Ay, Polani, and Virgo](https://arxiv.org/abs/1910.05979).

### Verdict

**Adaptation:** PID or a cooperative-game interaction decomposition is better suited
than total correlation to testing whether joint contributions carry outcome-relevant
information unavailable from the parts.

**Not sufficient for moral cooperation:** synergistic predation and coordinated
oppression are still synergistic.

## 8. Joint EFE “surplus”

The earlier proposal was

\[
\Delta_G=G_{\mathrm{ind}}-G_{\mathrm{joint}}.
\]

### Verdict

**Project hypothesis with strict admissibility conditions.** The contrast is meaningful
only when both quantities are computed within one declared joint generative model
using the same variables, horizon, preferences, base measure, and normalization. The
“independent” case must be an intervention that removes specified coupling factors,
not a separately fitted model whose evidence and EFE live on another scale.

A safer construction defines a coalition characteristic function inside one model:

\[
v(S)=-\min_{\pi_S}G^{do(\text{couplings outside }S=0)}(\pi_S),
\]

then evaluates complementarity through Harsanyi or Shapley interaction terms. This is
related to cooperative-game information decompositions, but its use with EFE is novel
and requires proofs of invariance and finite counterexamples.

## 9. Virtue, affordances, and expected consequences

### Literature

[Hampson, Hulsey, and McGarry](https://journals.sagepub.com/doi/10.1177/09593543211021662)
describe virtue or moral expertise as maintaining metastable optimal grip on moral
affordances. Their account explicitly includes prospection, anticipating consequences,
social learning, and phronetic orchestration. It does not formulate virtue as an EFE
term or prove that virtue is a function of a shared latent semantic variable.

### Verdict

**Established:** virtue application is context-sensitive, prospective, embodied,
socially learned, and dynamically organized.

**Project synthesis:** virtue is a higher-order regime acting on learned semantic and
policy dynamics. “Virtue is a function of meaning, expected consequences, and shared
semantics” should be presented as the paradigm's hypothesis, not attributed to active
inference or ecological virtue theory as an existing theorem.

## 10. Semantic information and viability

[Kolchinsky and Wolpert](https://pmc.ncbi.nlm.nih.gov/articles/PMC6227811/) define
semantic information as syntactic information causally necessary for maintaining a
system's viability under interventions that scramble correlations.

### Verdict

**Useful but narrower:** this provides a rigorous bridge from information to
phenotype-relative viability. It does not supply linguistic shared meaning, moral
standing, flourishing, or moral goodness. A system can carry viability-relevant
semantic information while harming other systems.

## 11. The other moral predicates

The following theories support the *interpretation* of VFE/EFE-generated trajectory
distributions; none is an ordinary term in VFE or EFE.

| Predicate | Relevant literature | What the equations can provide | What remains normative |
|---|---|---|---|
| Vulnerable affected centre | moral-patient and standing theories; causal intervention | effect distributions and uncertainty over standing-relevant evidence | who has standing and why |
| Flourishing | Aristotelian and capability approaches | viability, development, recovery, relationship, and learning trajectories | which functionings constitute flourishing |
| Robustness | robust decision making and distributionally robust optimization ([Lempert and Turner](https://pubmed.ncbi.nlm.nih.gov/32827199/); [Kuhn et al.](https://arxiv.org/abs/2411.02549)) | sensitivity across environments, models, priors, and semantic mappings | acceptable ambiguity sets and precaution levels |
| Capability floors | Sen–Nussbaum capability approach ([overview](https://plato.stanford.edu/entries/capability-approach/)) | reachable and exercisable option distributions | protected capabilities and thresholds |
| Non-externalization | causal system-boundary and affected-party analysis | joint intervention distributions including delayed and displaced costs | which costs may not be shifted |
| Non-domination | republican freedom as absence of uncontrolled power ([overview](https://plato.stanford.edu/entries/republicanism/)) | causal control, dependency, exit, veto, and intervention distributions | which control is arbitrary or uncontrolled |
| Public justification | public-reason and public-justification traditions ([overview](https://plato.stanford.edu/entries/justification-public/)) | recursive beliefs, reason exchange, information access, and deliberative trajectories | reasonableness and legitimacy of the procedure |
| Contestability | algorithmic contestability ([Lyons et al.](https://arxiv.org/abs/2103.01774)) | challenge uptake, revision probability, cost, and retaliation risk | adequate opportunity and authority to challenge |
| Epistemic responsiveness | Bayesian learning, model comparison, calibration | posterior and structure change after relevant evidence | relevance, warranted precision, and whose testimony counts |
| Repair | restorative-justice literature ([Sherman and Strang](https://dc.law.utah.edu/ulr/vol2003/iss1/2/)) | reachability, time, residual harm, recurrence, and institutional change | what counts as adequate repair and who decides |
| Plural improvement | value pluralism and social choice ([overview](https://plato.stanford.edu/entries/value-pluralism/)) | vector outcomes, partial orders, Pareto sets, and uncertainty | warranted orderings and treatment of incomparability |
| Legitimate selection | social choice, public reason, democratic legitimacy | feasible selection mechanisms and their predicted consequences | authority, inclusion, equality, and acceptable procedure |

## 12. Correct architecture

The literature supports the following layered model:

\[
\begin{array}{c}
\text{generative model with social, semantic, and institutional variables}\\
\downarrow\\
\text{VFE: state inference + parameter/structure learning}\\
\downarrow\\
\text{EFE: policy-conditioned protentions + action selection}\\
\downarrow\\
\text{derived diagnostics: processability, causal influence, synergy, reachability}\\
\downarrow\\
\text{explicit normative interpretation and constrained selection.}
\end{array}
\]

Nothing descriptive should be attached as an unanalyzed label. But not every moral
concept should be forced into a canonical VFE/EFE decomposition either. The empirical
grounds of a moral judgment should emerge from the dynamics; the judgment's authority
must remain explicit.

## 13. Immediate proof and experiment obligations

1. Derive the overlap/sheaf quadratic from an explicit likelihood and verify it in a
   finite model.
2. Show when learned semantic maps are identifiable, including permutation and
   non-identifiability counterexamples.
3. Replace total correlation as a cooperation gate with PID or coalition interaction
   diagnostics, retaining total correlation only as a dependence baseline.
4. Define the independent or coalition intervention inside a single joint model and
   prove that the EFE contrast compares like with like.
5. Decompose any coalition gain into pragmatic and epistemic contributions only from
   a fixed root EFE definition.
6. Construct adversarial examples: coercive efficiency, synergistic harm, semantic
   consensus around falsehood, and deterministic cooperation with zero total
   correlation.
7. Test whether the moral constraints discriminate these cases without using action
   labels as inputs.
