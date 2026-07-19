# Topological Alignment

## Virtue, pragmatics, and active inference

This project develops a geometric account of alignment in which agents are compared
not only by what they currently believe or do, but by the dynamical systems that
generate their beliefs, values, and actions. Its central objects are attractor
landscapes, metastable states, transition paths, and the costs of moving between
different forms of sense-making.

For a claim-by-claim assessment of the existing research, closest prior art, limits of
the present hypotheses, and proposed experiments, see the
[deep literature review](docs/VIRTUE_ACTIVE_INFERENCE_LITERATURE_REVIEW.md).
For the dependency-ordered questions, formal proof obligations, experiment sequence,
go/no-go gates, and paradigm-level failure conditions, see
[Proving Out the Paradigm](docs/PARADIGM_PROOF_OBLIGATIONS.md).
The standards and toolchain for accepting mathematical theorems are specified in
[Mathematical Proof Validation](docs/MATHEMATICAL_PROOF_VALIDATION.md).
The ordered milestones, exit criteria, and current status are maintained in the
[Research and Verification Roadmap](docs/ROADMAP.md).
The explicit philosophical passage from phenotype-relative functional good to moral
standing and admissibility is developed in
[From Phenotype-Relative Good to Moral Goodness](docs/NORMATIVE_BRIDGE.md).
Its process-level interpretation as a constrained multi-agent generative model is
specified in
[A Normatively Constrained Active-Inference Interpretation](docs/ACTIVE_INFERENCE_NORMATIVE_FORMALIZATION.md).
The category-by-category translation ledger is
[Translating Moral Goodness into Active-Inference Terms](docs/MORAL_GOODNESS_ACTIVE_INFERENCE_TRANSLATION.md).

This perspective also suggests a synthesis of virtue ethics, consequentialism, and
active inference.

The closest existing formulation is the ecological account of moral expertise
developed by Hampson, Hulsey, and McGarry, which already describes *phronesis* as
maintaining metastable optimal grip on a field of moral affordances. The contribution
pursued here is therefore not the bare claim that virtue is metastable grip. It is the
formal integration of that account with active inference and its extension to
pluralistic, multi-agent topological alignment.

Virtue ethics describes the relatively slow organization of a person: their
character, habits of attention, affective dispositions, and characteristic ways of
maintaining a grip on the world. But a virtue such as courage or humility does not by
itself determine what to do in a particular situation. Its pragmatic entailments
change with the context.

Consequentialism begins at the opposite end. It compares actions by their expected
outcomes, but it must already assume which outcomes count as good, whose outcomes
matter, how uncertainty should be treated, and how far into the future evaluation
should extend. It evaluates the pragmatics of action relative to a conception of the
good that it does not itself derive.

Active inference can connect these levels. On this account:

```text
virtue / character
        -> salience and affordances
        -> context-sensitive policy selection
        -> anticipated and actual consequences
        -> learning and transformation of character
```

A virtue is therefore neither a fixed rule nor a preferred behavioural output. It is
a higher-order, metastable organization of the processes through which an agent
perceives, evaluates, and acts.

## Virtue as a metastable control regime

The motivating phenomenological model treats virtues as complementary tendencies
whose adaptivity depends on their mutual organization. Pride, for example, can act
as an enabling constraint: it expands the agent's felt field of action and sustains an
embodied sense of "I can." Humility acts as a selective constraint: it distinguishes
viable opportunities from overreach.

Neither pole is intrinsically virtuous in isolation:

| Organization | Enabling tendency | Selective tendency | Result |
|---|---|---|---|
| Integrated | Pride | Humility | Adaptive agency and existential grip |
| Excess | Pride dominates | Insufficient humility | Arrogance, shamelessness, hubris |
| Deficiency | Insufficient pride | Humility dominates | Humiliation and self-incapacitation |

Virtue is not a static midpoint between excess and deficiency. It is a dynamically
maintained viable region whose location and expression change with the situation.
This recasts the Aristotelian mean as a metastable regime rather than a scalar
average.

An **abstract virtue** can consequently be understood as a schema whose realization
is constructed with a context. Courage can lead an agent to advance, retreat, ask for
help, tolerate discomfort, or refuse a reckless demand. These applications may differ
not only in action but in phenomenology and practical meaning. Their relation need not
be strict identity of regulatory topology; it may consist in partial overlaps, family
resemblance, or systematic transformations among local realizations.

This requires distinguishing two dynamical objects:

1. The **virtue concept** is an abstract, socially and linguistically stabilized
   attractor that organizes interpretation across a community and history.
2. A **virtue realization** is the local embodied organization enacted when that
   concept interacts with a particular agent, affordance field, and situation.

Application is not merely retrieval of fixed content. Local realizations feed back
into the abstract concept, changing its future inferential and pragmatic meaning. The
relation is therefore recursive:

\[
V_t \xrightarrow{\;R_{c,i}\;} V_{c,i}
\xrightarrow{\;\text{social uptake and learning}\;} V_{t+1},
\]

where \(V_t\) is the historically situated concept and \(V_{c,i}\) its realization by
agent \(i\) in context \(c\).

One feature does not vary in the same way: **a virtue is, qua virtue, good**. "Virtue"
is a success term and a thick evaluative concept, not a morally neutral label for a
recurring psychological pattern. If apparent courage is destructive recklessness, or
apparent humility is servile self-erasure, the problem is not simply that courage or
humility has acquired another equally valid realization. The application may have
failed to instantiate the virtue at all.

The proposal is therefore not that goodness is fixed while only behaviour changes,
nor that every contextual use determines its own goodness. Rather, the abstract
concept presents a context-sensitive normative demand:

\[
V \text{ is good},
\qquad
R_{c,i}(V) \text{ may or may not adequately realize } V \text{ in } c.
\]

What counts as adequate cannot be read from topology, stability, social acceptance,
or subjective grip alone. It requires substantive judgment about flourishing, truth,
harm, agency, justice, and relations to others. Active inference can model how an
agent interprets and enacts that demand; it cannot convert an enacted pattern into a
virtue merely by describing its dynamics.

### Phenotype-relative goodness

The first formal proof obligation is conditional: show that a virtue regime is good
relative to the flourishing of a phenotype under a specified range of environments.
Here **phenotype** does not mean a fixed biological essence or diagnostic category.
It denotes an embodied organization of capacities, needs, vulnerabilities,
developmental history, temporal scales, and social dependencies:

\[
P = (K_P, A_P, N_P, T_P, D_P),
\]

where \(K_P\) is a viability region, \(A_P\) the available capacities, \(N_P\) the
phenotype's characteristic needs, \(T_P\) its relevant temporal horizons, and \(D_P\)
its dependencies on other agents and environments.

Metastability is valuable relative to \(P\) when it supports a balance between
stability and adaptive transition. A good regime must do more than persist. Across a
declared environment class \(\mathcal E\), it should:

- keep trajectories within or return them toward \(K_P\);
- recover from perturbation without rigidly suppressing relevant error;
- switch policies when environmental demands change;
- retain epistemic sensitivity and meaningful future options;
- satisfy the phenotype's needs across the relevant temporal horizon; and
- avoid achieving self-maintenance by destroying the viability or agency of affected
  others.

A phenotype-relative functional can make the claim explicit:

\[
J_P(V;\mathcal E)
=
\mathbb E_{e\sim\mathcal E}
\left[
\int_0^T
\bigl(
w_v\,\mathrm{Viab}_P
+w_e\,\mathrm{Epistemic}
+w_r\,\mathrm{RelEmp}
-w_h\,\mathrm{Harm}
\bigr)dt
-\lambda\tau_{\mathrm{recovery}}
\right].
\]

A candidate realization \(V_c\) is functionally better for phenotype \(P\) than a
comparison regime \(U_c\) only if it robustly improves \(J_P\) across relevant
perturbations, not merely in one preferred environment, while respecting floors on
the viability and agency of affected others.

This yields a conditional result:

\[
J_P(V_c;\mathcal E) > J_P(U_c;\mathcal E)
\quad\Longrightarrow\quad
V_c \text{ is better for } P
\text{ under } \mathcal E
\text{ and the stated criteria}.
\]

It does not yet prove moral goodness simpliciter. That requires an explicit bridge
principle: why phenotype-relative flourishing has normative standing, how conflicts
among phenotypes should be adjudicated, and why another agent's viability cannot be
treated merely as an instrumental constraint. Keeping this bridge visible prevents
metastability or survival from silently becoming a complete ethics.

## Four levels of normative agency

The proposed account distinguishes four interacting levels.

### 1. Existential viability

An embodied agent must remain capable of acting and sense-making. Precariousness
creates a basic normative asymmetry between trajectories that preserve agency and
those that produce breakdown. Its phenomenological correlate is an embodied sense
of grip: a situated sense that one can respond to the world.

Viability is a source of normativity, but it is not yet a complete theory of moral
goodness. Locally stable forms of grip can still be exploitative, delusional, or
destructive.

### 2. Felt metastable states

Confidence, humility, anxiety, shame, pride, and similar phenomena are embodied
modes of readiness. They alter salience, felt affordances, temporal horizons, and the
policies that appear possible. They are not merely private representations of an
otherwise unchanged decision problem.

### 3. Virtue and character

Virtues are slow, learned organizations of these affective and inferential processes.
They shape:

- prior preferences over forms of life and experience;
- precision assigned to evidence, errors, needs, and other agents;
- priors over familiar or trusted policies;
- temporal depth and sensitivity to delayed consequences;
- the agent's self-model and perceived capacities;
- the rate and direction of learning.

Virtues belong to the generative organization from which practical reasoning
emerges, rather than being terminal rewards attached to individual actions.

### 4. Pragmatic entailments

The practical meaning of a virtue is generated through its interaction with a
particular context. The same organization can therefore entail different actions as
affordances, evidence, dependencies, and risks change.

Virtue ethics has substantial accounts of right action, moral perception, expertise,
and *phronesis*. Its remaining limitation is narrower: these qualitative resources
generally underdetermine an explicit, empirically estimable process mapping character
and context to policy selection.

## Active inference as the bridge

Let:

- \(x_t\) denote an agent's embodied-affective state;
- \(c_t\) denote its current context;
- \(A(x_t,c_t)\) denote its perceived affordance field;
- \(V\) denote a virtue organization;
- \(\pi(a\mid x,c,V)\) denote its context-sensitive distribution over actions.

The agent's dynamics may be represented schematically as

\[
\dot{x}_t = F(x_t,c_t,V) + \eta_t,
\]

where \(\eta_t\) captures endogenous and environmental variability. A virtue is not
a particular value of \(x\), but a metastable region \(\mathcal V_c\) in which the
relevant processes remain mutually regulating:

\[
\mathcal V_c = \left\{x :
G(x,c) \geq G_{\min},\;
\mathfrak E(x) \geq \mathfrak E_{\min},\;
V\text{ remains responsive to context}
\right\}.
\]

Here \(G\) measures situated grip or viability, while \(\mathfrak E\) measures
empowerment: the agent's retained capacity to influence and revise its future states.
Virtuous dynamics remain within, or can recover, this viable region under changing
conditions.

The virtue remains relatively stable while its pragmatic entailments change:

\[
c_1 \neq c_2
\quad\Longrightarrow\quad
\pi(\cdot\mid x,c_1,V) \neq \pi(\cdot\mid x,c_2,V).
\]

This explains how a shared virtue can produce heterogeneous behaviour without
making virtue contentless.

### Expected free energy

Active inference evaluates policies using expected free energy, which can be
decomposed schematically into pragmatic and epistemic contributions:

\[
G(\pi) \approx
\underbrace{\mathbb E[-\log P(o\mid C)]}_{\text{preference or pragmatic cost}}
-
\underbrace{I(s;o\mid\pi)}_{\text{epistemic value}}.
\]

This improves upon a purely outcome-directed model because a policy can be valuable
both for approaching preferred outcomes and for reducing relevant uncertainty. The
appropriate action may be to investigate, listen, experiment, or suspend judgment
rather than immediately optimize a represented consequence.

Expected free energy does not, however, solve ethics on its own. The prior preference
distribution \(P(o\mid C)\), model structure, precision assignments, and temporal
horizon already embody normative commitments. Without an account of how these
commitments arise and remain corrigible, active inference becomes a sophisticated
form of consequentialism with an epistemic term.

## Practical wisdom as model governance

This framework gives one part of *phronesis* a possible mechanistic interpretation:

> Practical wisdom governs the model through which a situation becomes actionable:
> its relevant states, affected agents, affordances, time horizon, preferences,
> confidence assignments, and candidate policies.

Precision governance is one mechanism within this broader process. It controls which
predictions, errors, affordances, policies, and levels of the model dominate action;
it does not supply their moral content.

A practically wise agent must determine:

- which evidence deserves confidence;
- which affordance should become salient;
- whether uncertainty calls for action or information gathering;
- when a familiar virtue is being applied appropriately;
- when persistence has become rigidity or courage has become recklessness;
- when consequences require local correction or deeper character revision.

Under this interpretation, complementary virtues regulate one another partly by
modulating precision. Pride prevents threat and incapacity from monopolizing action
selection; humility prevents an overconfident self-model from suppressing relevant
prediction errors.

## The inverse limitations of virtue ethics and consequentialism

| Framework | What it captures | What it tends to leave implicit |
|---|---|---|
| Virtue ethics | Slow character organization, moral perception, practical wisdom, and forms of flourishing | A formal process model from character and context to policy |
| Consequentialism | Evaluation of actions, rules, motives, or dispositions by their consequences | The axiology used to rank those consequences |
| Active inference | Policy selection under preferences and uncertainty | Why its priors and preference structures are morally justified |
| Topological alignment | Structure, flexibility, and compatibility of generative dynamics | The complete substantive content of the good |

The proposal is therefore not that active inference replaces ethics. Rather, it can
show where distinct ethical theories operate within a shared dynamical architecture
and make their hidden assumptions explicit.

## Pluralism without simple relativism

Different people may associate "good" with different felt states and actions because
their bodies, histories, environments, cultural practices, and repertoires of
affordances differ. This produces at least three forms of apparent disagreement:

1. **Different realizations:** agents instantiate a similar normative organization
   through different affective and behavioural patterns.
2. **Different contexts:** the same virtue has different pragmatic entailments because
   the situations differ.
3. **Different landscapes:** agents possess genuinely incompatible conceptions of
   viability, flourishing, or the good.

Only the third is necessarily a deep normative conflict. Alignment should therefore
not be defined as identity of outputs or internal states. It may instead require a
structure-preserving relationship between heterogeneous normative dynamics.

## Relation to topological alignment

The broader project models slow beliefs, values, and identity commitments as an
attractor landscape, while faster perceptual and behavioural dynamics trace paths
within it. The ethical synthesis refines this picture:

```text
values       = relatively abstract viability and preference conditions
virtues      = metastable regimes that regulate pursuit of those conditions
phronesis    = contextual inference from those regimes to policies
actions      = local pragmatic realizations
consequences = feedback that updates judgment and, more slowly, character
```

Two agents may produce the same action from very different landscapes: one flexible
and evidence-responsive, another compulsive or fear-driven. Conversely, agents can
choose different actions while instantiating structurally comparable virtues.

Normative alignment should consequently compare such properties as:

- responsiveness of attractors to genuine evidence;
- capacity to recover from perturbation;
- preservation of the agent's and others' empowerment;
- topology and cost of transitions between normative basins;
- compatibility among different forms of flourishing;
- openness to revision of the underlying conception of the good.

Topology alone is not morally informative. Two systems can have the same basins and
loops while assigning opposed meanings to them. A usable alignment object must
therefore decorate its topology with semantic content, transition direction,
affective and material cost, control allocation, and effects on other agents.

This motivates the following working definition:

> **Normative alignment is compatibility between the dynamical organizations through
> which agents maintain, enact, and revise viable forms of grip.**

Compatibility does not demand identical beliefs, virtues, experiences, or actions. It
requires that different normative regimes remain epistemically responsive, mutually
non-destructive, and capable of affordable coordination.

## Moral constraints beyond existential grip

Existential grip is necessary for embodied agency but insufficient for morality. An
agent, institution, or community can occupy a deep and phenomenologically compelling
attractor while dominating others or systematically resisting evidence.

A beneficial normative landscape must therefore include additional constraints:

- **epistemic adequacy:** sensitivity to evidence and resistance to manipulation;
- **relational empowerment:** maintenance of meaningful capacities for action and
  revision for the self, other agents, and cooperative groups, including attention to
  how control is distributed;
- **relational non-domination:** refusal to secure one agent's grip by destroying
  another's agency;
- **cooperative compatibility:** affordable paths toward coordination;
- **pluralism:** support for distinct but mutually accessible forms of flourishing;
- **reversibility and corrigibility:** preservation of routes through which commitments
  and interventions can be reconsidered.

Consequences then play a recursive role. If purported courage repeatedly causes
needless harm, or purported humility repeatedly enables domination, those outcomes
must be capable of changing not only the selected action but the generative
organization that presented it as virtuous.

## Research programme

This synthesis suggests several directions for theoretical and empirical work:

1. Model virtues as structured slow regions of generative-model parameter space,
   spanning preferences, policy priors, precision, self- and other-models, learning,
   and temporal depth.
2. Formalize practical wisdom as context-sensitive model governance, with precision
   allocation as one candidate mechanism.
3. Identify phenomenological and behavioural signatures of metastable virtue regimes
   and their breakdowns.
4. Distinguish different realizations of a shared virtue from genuinely different
   normative attractor landscapes.
5. Measure when consequences produce local policy learning versus structural change
   in character.
6. Extend topological alignment metrics to compare virtue-generating dynamics across
   agents without requiring identical state spaces or behaviours.
7. Test whether empowerment, epistemic openness, and transition accessibility
   distinguish adaptive virtue regimes from rigid but locally stable forms of grip.
8. Replace individual empowerment with measures of self-, other-, and joint
   empowerment, and test these separately from willingness to accept correction.

The resulting view treats ethical agency as a circular, multiscale process:

\[
\text{character}
\rightarrow \text{inference}
\rightarrow \text{action}
\rightarrow \text{consequence}
\rightarrow \text{learning}
\rightarrow \text{character}.
\]

Virtue ethics describes the slow geometry of this process. Consequentialism evaluates
some of its projected pragmatic effects. Active inference models the inferential bridge
between them. Topological alignment asks whether the resulting dynamical systems can
remain truthful, flexible, plural, and cooperatively compatible over time.
