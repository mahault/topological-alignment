# Topological Alignment

For a visual, executable account of all 20 canonical claims—including each claim's
motivation, formal object, assumptions, proof standard, experiment, falsifier,
current evidence, graphs, and pedagogical animations—open the
[Where We Are notebook](notebooks/where_we_are.ipynb) or its
[standalone HTML view](notebooks/where_we_are.html).
An alternative presentation-oriented rendering is available as the
[Research Dashboard notebook](notebooks/RESEARCH_DASHBOARD.ipynb) and
[Research Dashboard HTML](notebooks/RESEARCH_DASHBOARD.html).
The public collaborator-facing versions are the
[GitHub Pages research dashboard](https://mahault.github.io/topological-alignment/)
and the [detailed visual guide](https://mahault.github.io/topological-alignment/where-we-are.html).
Deployment and update instructions are in
[Public Research Dashboard](docs/GITHUB_PAGES.md).
The rationale for the definitions, the exact mathematical mapping, the interpretation
of the mechanism ablations, and the experiment genealogy are recorded in
[Definition Decisions and Virtue Mapping](docs/DEFINITION_DECISIONS_AND_MAPPING.md).

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
For a claim-by-claim review of whether meaning, semantic alignment, cooperation, and
the moral predicates genuinely follow from VFE/EFE dynamics, see the
[VFE/EFE Semantics and Cooperation Literature Audit](docs/VFE_EFE_SEMANTICS_COOPERATION_LITERATURE_AUDIT.md).
The corresponding equation-by-equation replacements for every unsupported construct
are recorded in the
[Supported Replacements Ledger](docs/SUPPORTED_REPLACEMENTS_LEDGER.md).
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
Its initial adversarial test suite is documented in the
[Active-Inference Moral Translation Audit](docs/ACTIVE_INFERENCE_TRANSLATION_AUDIT.md).
The repository-wide findings, corrections, rerun evidence, and remaining blockers are
recorded in the [Adversarial Audit of 2026-07-19](docs/ADVERSARIAL_AUDIT_2026-07-19.md).
The current canonical account of felt goodness, multi-scale enabling relations,
practical wisdom, and virtue attractors is
[Multi-Scale Goodness and Virtue Attractors](docs/MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md).
The repository-wide disposition of every experiment, definition, and pipeline is in
the [Experiment, Definition, and Pipeline Review of 2026-07-19](docs/EXPERIMENT_DEFINITION_PIPELINE_REVIEW_2026-07-19.md).
The first redesigned study and its successful adversarial design-recovery simulation
are documented in the
[V0 Felt-Goodness Measurement Pilot](docs/V0_FELT_GOODNESS_MEASUREMENT_PILOT.md).
The first finite mechanism test connecting counterfactual phenotype dynamics, VFE
updates, EFE-derived feeling, and captured institutional priors is
[V1 Multi-Scale Enablingness and EFE Calibration](docs/V1_MULTISCALE_ENABLING_EFE_SIMULATION.md).
The next constructed dynamical test—virtue as selective stability of meaning relative
to goodness—is documented in
[V2/V3 Virtue-Attractor Simulation](docs/V2_VIRTUE_ATTRACTOR_CALIBRATION_SIMULATION.md).
Its relationship to the semantic, social, and multi-agent machinery already built in
sibling projects is assessed in the
[Cross-Project Integration Assessment](docs/CROSS_PROJECT_INTEGRATION_ASSESSMENT_2026-07-20.md).
The first explicit probabilistic model is the
[Finite Active-Inference Externalization Counterexample](docs/FINITE_ACTIVE_INFERENCE_COUNTEREXAMPLE.md).

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

A virtue is therefore neither a fixed rule nor a preferred behavioural output. The
testable model treats it as a slow hierarchical latent regime over preferences,
likelihood and transition beliefs, policy priors, precision, and temporal depth. It
organizes how agents make sense of expected relational consequences and perceive,
evaluate, coordinate, act, and revise. Its content is partly socially stabilized:
the same word does not denote the same virtue merely because different agents utter
it.

More precisely, practical meaning is an agent- and context-indexed profile of expected
consequences, policies, affordances, affect, learned cue mappings, parameter beliefs,
and precision. Shared meaning exists when heterogeneous profiles remain reciprocally
processable on relevant overlaps. A virtue is hypothesized to be a slow metastable
regime that organizes these profiles and transforms them appropriately with context.
It is not one action, trait score, felt state, or verbal label.

Likewise, cooperation is not an atomic action label or a single statistic. Candidate
cooperative coordination is evaluated through a vector: joint achievement, mutual
semantic processability, reciprocal readability, outcome-relevant synergy, causal
contribution, and each agent's matched coupling contrast. Dependence alone cannot
define it, because coercion and exploitation are also statistically joint. Moral
cooperation additionally requires the constraints developed below.

These objects must arise from explicit inference, prediction, or intervention. Shared
signs and local likelihood mappings are learned first; processability, sheaf
consistency, causal contribution, and readability are then measured as distinct
diagnostics. EFE generates protentions and policy posteriors, but agent-local EFEs
remain a vector unless a common aggregation rule is justified. Total correlation is
retained only as a dependence baseline. Normative diagnostics then test whether the
felt and socially stabilized sense of goodness tracks actual enabling relations
rather than coercion or exploitation.

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
recurring psychological pattern. Attractorhood itself is morally neutral: dogmatism
and servility may also be stable, confident, and socially reproduced.

The proposal is therefore not that goodness is fixed while only behaviour changes,
nor that every contextual use determines its own goodness. Rather, the abstract
concept presents a context-sensitive normative demand:

\[
V \text{ is good},
\qquad
R_{c,i}(V) \text{ may or may not adequately realize } V \text{ in } c.
\]

The present hypothesis is that goodness is first encountered as a fallible embodied
heuristic for multi-scale relations that enable a phenotype's viable, recoverable,
revisable modes of life. A successful virtue realization is therefore not merely an
attractor: its felt orientation must be calibrated to those enabling relations and
its enactment must actually sustain them across relevant perturbations. Active
inference can model the estimation, action, and revision dynamics; it does not make
the heuristic infallible.

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

A higher-scale process can be tested through its counterfactual contribution to a
phenotype's **attainable metastability** \(\mathcal A_P\):

\[
\operatorname{En}_{P}^{(\ell)}
=
\mathcal A_P
-
\mathcal A_P^{\operatorname{do}(X^{(\ell)}\ \mathrm{removed\ or\ scrambled})}.
\]

Because agents cannot solve this counterfactual problem online, felt goodness is
modeled as a learned embodied compression of expected changes in attainable
metastability and enabling relations:

\[
g_i(t)\approx f_{\psi_i}\!\left(
\mathbb E_{q_i}[\Delta\mathcal A_i,
\Delta\operatorname{En}_{i}^{(1:L)}\mid o_{1:t}]
\right).
\]

This is not a scalar utility and does not prove moral goodness simpliciter. The target
is a partially ordered, phenotype- and scale-indexed vector. Capability floors,
non-externalization, non-domination, contestability, repair, and plural improvement
are retained as error-correcting diagnostics of what the heuristic tracks. They are
not bolted-on constituents of felt goodness. The bridge to standing and conflicts
among phenotypes remains an explicit philosophical obligation.

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

The following diagnostics test whether felt and socially stabilized goodness tracks
actual enabling relations across phenotypes and scales:

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

These diagnostics are not the definition of felt goodness. They expose characteristic
ways in which the heuristic can be captured, truncated, or rationalized. Consequences
then play a recursive role. If purported courage repeatedly causes
needless harm, or purported humility repeatedly enables domination, those outcomes
must be capable of changing not only the selected action but the generative
organization that presented it as virtuous.

## Research programme

This synthesis suggests several directions for theoretical and empirical work:

1. Identify attainable metastability and counterfactual enabling contributions at
   individual, relational, institutional, and ecological scales.
2. Test whether immediate felt goodness estimates those contributions beyond comfort,
   reward, approval, individual viability, and explicit reasons.
3. Model virtues as structured slow regions of generative-model parameter space,
   spanning preferences, policy priors, precision, self- and other-models, learning,
   and temporal depth.
4. Formalize practical wisdom as context-sensitive calibration and model governance, with precision
   allocation as one candidate mechanism.
5. Identify phenomenological and behavioural signatures of metastable virtue regimes
   and their breakdowns.
6. Distinguish different realizations of a shared virtue from genuinely different
   normative attractor landscapes.
7. Measure when consequences produce local policy learning versus structural change
   in character.
8. Extend topological alignment metrics to compare virtue-generating dynamics across
   agents without requiring identical state spaces or behaviours.
9. Test whether relational capacity, epistemic openness, and transition accessibility
   distinguish adaptive virtue regimes from rigid but locally stable forms of grip.
10. Replace individual empowerment with measures of self-, other-, and joint
   empowerment, and test these separately from willingness to accept correction.

The current executable milestone is V0. Its simulation shows that the proposed
balanced, temporally separated measurement design can recover a programmed
multi-scale signal and reject proxy-only, confounded, and prompt-contaminated cases.
It is a design check, not human evidence.

V1 then shows that, inside a declared finite model, nested EFE contrasts can be
calibrated to separately computed counterfactual enablingness, while approval-captured
priors and preferences systematically accept institution-preserving harms. This is a
mechanism and adversarial test, not moral or human validation.

V2/V3 adds slow semantic organization. It tests the sharper claim that virtue is an
attractor of the closed-loop calibration of meaning to goodness: recovery after
low-precision noise, transformation after diagnostic evidence, and persistence of
that transformation across later ordinary contexts.

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
