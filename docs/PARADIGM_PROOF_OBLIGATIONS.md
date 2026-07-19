# Proving Out the Paradigm

## Questions, dependencies, evidence, and failure conditions

**Status:** research roadmap
**Date:** 2026-07-19

## 1. The paradigm in one statement

The project proposes that:

> Virtue concepts are historically and socially stabilized abstractions of the good.
> They are locally realized through context-sensitive, embodied, metastable dynamics.
> A successful realization is good for a phenotype when it robustly supports viable,
> epistemically responsive flourishing across relevant environments. It is morally
> good when that flourishing is organized through relations that recognize the
> standing of affected others, preserve capability floors, resist domination, remain
> contestable, and enable plural co-flourishing. Active inference can model how
> abstract normative demands become situated policies, while decorated dynamical
> topology can compare their organization across heterogeneous agents without
> reducing morality to geometry.

This statement contains conceptual, mathematical, empirical, and normative claims.
They cannot all be proven in the same sense.

## 2. Four standards of support

Every claim in the programme should be marked by its appropriate evidential status.

| Status | What establishes it | Example |
|---|---|---|
| **Conceptual** | clear definitions and a valid philosophical argument | distinguishing virtue from an apparent virtue |
| **Mathematical** | a proof from explicit assumptions | a robustness bound for remaining in a viability region |
| **Empirical** | preregistered discriminating evidence | a transformed-schema model predicts unseen contexts |
| **Normative** | defensible bridge principles and public justification | why every affected centre of flourishing has standing |

Empirical evidence cannot prove a moral bridge principle. A mathematical theorem
cannot show that its assumptions are true of humans. A phenomenological description
cannot establish the identifiability of a latent model. Keeping these standards
separate is a central methodological requirement.

## 3. Dependency structure

```text
definitions
    -> phenotype-relative viability and flourishing
        -> metastable virtue realization
            -> context-sensitive policy generation
                -> empirical identification and prediction

normative standing of affected phenotypes
    -> non-domination and capability floors
        -> morally admissible relations
            -> plural co-flourishing

metastable dynamics + moral constraints + semantic content
    -> decorated normative dynamical object
        -> cross-agent alignment comparison
            -> steering and AI-alignment intervention
```

The empirical and normative paths meet before the framework can make claims about
moral alignment. Neither path can substitute for the other.

## 4. Foundational definitions

### Q1. What is the bearer of the dynamics?

Is the relevant system:

- an internal belief state;
- an embodied agent;
- an agent-environment coupling;
- a social relation;
- a community practice; or
- a multiscale system containing all of these?

**Required answer:** specify a relational state space

\[
z_t=(x_t,e_t,n_t,r_t),
\]

where \(x_t\) is embodied agent state, \(e_t\) the local environment, \(n_t\) the
normative niche, and \(r_t\) the agent's relations to others.

**Failure condition:** the framework changes the bearer of an attractor opportunistically
between arguments, making its measurements incomparable.

### Q2. What is a phenotype?

Which features belong to a phenotype, which are environmental, and which are products
of developmental or social history?

Working representation:

\[
P=(K_P,A_P,N_P,T_P,D_P),
\]

with a viability region, capacities, needs, temporal horizons, and dependencies.

Questions that must be answered:

1. Are needs inferred from observed behaviour, self-report, physiology, or a theory of
   flourishing?
2. How do we avoid treating adaptive preferences caused by oppression as genuine
   needs?
3. How stable is a phenotype across learning and development?
4. At what level of granularity should two phenotypes be distinguished?
5. How do social scaffolds partly constitute rather than merely support a phenotype?
6. Can artificial agents possess a phenotype in the relevant sense, or only a
   functional analogue?

**Failure condition:** phenotype becomes an unconstrained label adjusted after seeing
the outcome.

### Q3. What is an abstract virtue concept?

Is it best modeled as:

- a prototype;
- a family-resemblance cluster;
- a socially stabilized inferential role;
- an equivalence class of realizations;
- a higher-order constraint;
- an attractor in a cultural-semantic dynamical system; or
- some combination of these?

The framework must explain both continuity and contestability. A concept that cannot
change cannot accommodate historical development; a concept with no continuity
cannot distinguish transformation from equivocation.

### Q4. What is a local virtue realization?

For concept \(V_t\), agent \(i\), and context \(c\), define:

\[
R_{c,i}(V_t)=V_{c,i}.
\]

We must specify what \(R_{c,i}\) transforms:

- salience;
- affect and felt grip;
- represented stakeholders;
- preferences;
- precision;
- policy repertoire;
- temporal depth;
- information seeking; and
- action.

**Key question:** when does \(V_{c,i}\) remain a realization of \(V_t\), and when has
it become a defective attempt or a neighboring vice?

### Q5. In what sense is virtue necessarily good?

"Virtue" is a success term. The framework must distinguish:

\[
\text{virtue}
\neq
\text{socially called a virtue}
\neq
\text{stable character pattern}
\neq
\text{subjectively satisfying grip}.
\]

Questions:

1. Is goodness analytically contained in the concept of virtue?
2. What independent criteria determine whether an application succeeds?
3. Can communities be systematically mistaken about what they call virtues?
4. What makes a contested extension a legitimate development rather than corruption?
5. How can criteria be independent enough to avoid defining virtue and goodness in a
   circle?

## 5. Phenotype-relative functional goodness

### Q6. What constitutes flourishing for phenotype \(P\)?

A proposed functional is:

\[
J_P(V;\mathcal E)
=
\mathbb E_{e\sim\mathcal E}
\left[
\int_0^T
\left(
w_v\mathrm{Viab}_P
+w_e\mathrm{Epistemic}
+w_r\mathrm{RelEmp}
-w_h\mathrm{Harm}
\right)dt
-\lambda\tau_{\mathrm{recovery}}
\right].
\]

Every term raises a proof obligation:

1. Why are these components included?
2. Are they commensurable on a scalar scale?
3. Who sets the weights?
4. Should some terms be hard constraints instead of tradeable quantities?
5. How is the relevant environment distribution \(\mathcal E\) selected?
6. How sensitive are conclusions to the horizon \(T\)?
7. How do we prevent overfitting the definition to preferred examples?

**Preferred approach:** use a partially ordered vector of outcomes and capability
floors before introducing any scalar aggregation.

### Q7. Why is metastability valuable?

Metastability must be separated from stability, random switching, and mere behavioral
variability.

Questions:

1. What measurable balance of integration and segregation constitutes adaptive
   metastability?
2. Which dwell-time and switching distributions are predicted?
3. How quickly should the system recover after perturbation?
4. When is persistence virtuous commitment rather than rigidity?
5. When is switching responsiveness rather than instability?
6. Does an optimal metastable regime vary by phenotype and environment?
7. Can maladaptive systems also be metastable?

**Empirical requirement:** metastability must predict phenotype-relative outcomes
beyond ordinary measures of variability, flexibility, and entropy.

### Q8. Can phenotype-relative goodness be proven non-tautologically?

A conditional theorem may take the form:

> Given a phenotype model \(P\), environment class \(\mathcal E\), capability floors,
> and independently justified outcome ordering, regime \(V\) is better for \(P\) than
> regime \(U\) if it robustly dominates \(U\) across relevant perturbations without
> violating the floors of affected agents.

This is informative only if:

- the phenotype model is independently validated;
- the comparison regimes are genuine alternatives;
- robustness is assessed outside the fitting environments;
- the outcome ordering was not defined from the desired conclusion; and
- costs to others are recorded rather than hidden in the environment.

### Mathematical proof obligations

1. **P1 -- Viability:** establish conditions under which trajectories remain within or
   return to \(K_P\).
2. **P2 -- Recovery:** bound recovery time after a defined perturbation class.
3. **P3 -- Adaptive switching:** show that regime changes track environmental regime
   changes rather than noise alone.
4. **P4 -- Robust dominance:** prove that benefits are not confined to one narrow
   environment realization.
5. **P5 -- Non-externalization:** represent effects on other phenotypes inside the
   evaluated system.

## 6. The bridge to moral goodness

### Q9. Why does having a good confer moral standing?

The required bridge principle is approximately:

> Every being for whom conditions can genuinely go better or worse has prima facie
> normative standing.

Questions:

1. Is this principle realist, constructivist, contractualist, or constitutivist?
2. What capacities are sufficient for having a good?
3. Are sentience, agency, vulnerability, life, or social participation necessary?
4. Does standing come in degrees?
5. How should uncertain standing be treated?
6. Do collectives and ecosystems possess standing independently of their members?
7. Could an artificial system possess standing, and what evidence would justify it?

This is a normative argument, not a deduction from free energy or survival.

### Q10. Why equal or impartial standing?

If an agent treats its own flourishing as reason-giving, why must it recognize
analogous claims in others?

Candidate arguments include:

- consistency among relevantly similar centres of flourishing;
- reciprocal justification;
- constitutive social dependence of agency;
- contractualist rejectability;
- non-arbitrary universalization; and
- recognition of others as subjects rather than environmental variables.

The project must state which argument it accepts and what it commits us to.

### Q11. What are the constraints on morally admissible dynamics?

Working admissibility set:

\[
\mathcal A_{\mathrm{moral}}
=
\left\{
V:
\begin{array}{l}
J_i(V)\geq J_i^{\min}\quad \forall i,\\
\mathrm{NonDom}(V),\\
\mathrm{Justifiable}(V),\\
\mathrm{Contestable}(V),\\
\mathrm{EpistemicallyResponsive}(V)
\end{array}
\right\}.
\]

Questions:

1. Which capability floors are non-negotiable?
2. When, if ever, may one floor be violated to prevent catastrophe?
3. How is domination measured independently of momentary welfare?
4. What makes justification acceptable to differently situated agents?
5. How much contestability is required when decisions are urgent?
6. Must interventions be reversible when non-intervention is itself irreversible?
7. How are responsibility, repair, and historical injustice represented?

### Q12. How should conflicts among genuine goods be resolved?

Plural phenotypes can possess incompatible but legitimate goods.

The framework must compare:

- Pareto improvement;
- maximin or leximin protection of the worst positioned;
- capability thresholds;
- contractualist rejection;
- bargaining solutions;
- democratic or deliberative procedures; and
- tragic conflicts with no fully good outcome.

**Failure condition:** all conflict is hidden inside arbitrary scalar weights.

### Q13. What is relational empowerment?

Individual channel capacity is not moral agency. Define:

\[
\mathbf E_{\mathrm{rel}}
=(E_{\mathrm{self}},E_{\mathrm{other}},E_{\mathrm{joint}},
\Delta E_{\mathrm{distribution}}).
\]

Questions:

1. What counts as a meaningful rather than merely numerous option?
2. How are joint capabilities distinguished from one agent controlling another?
3. How is the distribution of veto, exit, voice, and agenda-setting power measured?
4. Can an increase in total empowerment be rejected because it concentrates control?
5. How does relational empowerment differ from freedom, capability, autonomy, and
   non-domination?

### Normative proof obligations

1. **N1 -- Standing:** defend which entities possess moral standing.
2. **N2 -- Symmetry:** justify non-arbitrary treatment of relevantly similar standing.
3. **N3 -- Floors:** defend the basic capabilities that cannot be routinely traded.
4. **N4 -- Non-domination:** show why control asymmetry matters beyond experienced
   welfare.
5. **N5 -- Conflict:** specify a legitimate procedure for incompatible goods.
6. **N6 -- Revision:** explain how the moral criteria themselves remain corrigible
   without becoming empty relativism.

## 7. Active inference as the pragmatic bridge

### Q14. What exactly in the generative model corresponds to virtue?

Virtue should not be assigned to one parameter without evidence. Candidate components
include:

\[
V=\{C,E,\gamma,A,B,D,M,H,L\},
\]

where the terms represent preferences, policy priors, precision, likelihood beliefs,
transition beliefs, state priors, model structure, temporal horizon, and learning.

Questions:

1. Which components are necessary and sufficient?
2. Can different parameterizations generate observationally identical behaviour?
3. How are slow character dynamics separated from fast situational inference?
4. Does the model generalize across domains and sessions?
5. Can a virtue change model structure rather than merely parameter values?
6. How do social practices and institutions enter the generative model?

### Q15. What exactly is practical wisdom?

The working hypothesis is **model governance**, with precision governance as one
mechanism.

We must test whether practical wisdom governs:

- which state space is relevant;
- whose standpoint is represented;
- which affordances are perceived;
- which policies are generated;
- the temporal and social horizon;
- confidence in evidence and policy;
- whether to gather information;
- whether the present model should be revised; and
- when a virtue concept has been misapplied.

**Failure condition:** *phronesis* becomes the unexplained homunculus that selects the
right model whenever the theory would otherwise fail.

### Q16. What does expected free energy contribute?

Questions:

1. Which EFE formulation is being used?
2. What additional assumptions introduce prior preferences and epistemic value?
3. How are moral preferences distinguished from arbitrary preferences?
4. Does information seeking occur only when preferred outcomes make it worthwhile?
5. Are alternative control-as-inference or reinforcement-learning models equally
   predictive?
6. Does active inference add explanatory value, or merely reparameterize expected
   utility and Bayesian learning?

### Q17. How do consequences revise virtue concepts?

The recursive relation is:

\[
V_t\xrightarrow{R_{c,i}}V_{c,i}
\xrightarrow{\mathrm{outcome, uptake, learning}}V_{t+1}.
\]

Questions:

1. Which consequences revise local policy, character, or the public concept?
2. How are affected parties' testimony and resistance weighted?
3. What prevents repeated oppressive practice from simply deepening a bad concept?
4. How do exemplars, narratives, laws, and institutions alter the attractor?
5. How can conceptual progress be distinguished from drift?

### Computational proof obligations

1. **C1 -- Identifiability:** recover generative parameters from realistic data.
2. **C2 -- Timescale separation:** distinguish concept, character, context, and action.
3. **C3 -- Generalization:** predict held-out context families, not only familiar
   trials.
4. **C4 -- Model comparison:** outperform simpler trait-by-situation and learning
   models after penalizing complexity.
5. **C5 -- Recursive learning:** predict when feedback changes action, character, or
   concept.

## 8. Topology and alignment

### Q18. What should be topologically compared?

Candidate objects include:

- trajectories of beliefs;
- trajectories through affordance fields;
- policy posterior dynamics;
- affective-control states;
- interpersonal coordination; and
- cultural-semantic concept dynamics.

Each lives in a different state space and supports different interpretations.

### Q19. What information does bare topology erase?

At minimum:

- semantic content;
- normative valence;
- causal direction;
- transition control;
- affective and material cost;
- effects on others;
- historical path; and
- uncertainty in the reconstruction.

Therefore use a decorated object:

\[
\mathcal N_i=(X_i,\varphi_i,\mathcal A_i,L_i,Q_i,P_i,U_i),
\]

where \(L_i\) provides semantic labels, \(Q_i\) transition costs and control,
\(P_i\) relational effects and power, and \(U_i\) uncertainty.

### Q20. What correspondence preserves moral meaning?

Questions:

1. Which labels must an alignment map preserve?
2. Must it preserve counterfactual responses to evidence?
3. How should asymmetric transition costs be represented?
4. Can different topologies implement the same moral function?
5. Can identical topologies implement opposed moral functions?
6. What constitutes approximate rather than exact correspondence?
7. How is uncertainty propagated into the final alignment judgment?

### Q21. What does alignment mean under pluralism?

Possibilities include:

- identical preferred states;
- compatible viability regions;
- mutually affordable coordination paths;
- non-dominating coexistence;
- preservation of each agent's legitimate differences; and
- capacity to negotiate and repair disagreement.

The framework should treat alignment as a structured relation, not a scalar synonym
for agreement.

### Topological proof obligations

1. **T1 -- Reconstruction:** reliably recover relevant dynamical features from finite,
   noisy trajectories.
2. **T2 -- Stability:** bound sensitivity to measurement and embedding choices.
3. **T3 -- Semantic preservation:** demonstrate why decorated mappings preserve the
   selected moral features.
4. **T4 -- Discrimination:** separate identical topology/opposed content and different
   topology/equivalent function.
5. **T5 -- Added value:** show that topology predicts something beyond conventional
   state-space and semantic models.

## 9. Empirical questions

### E1. Are virtue concepts empirically distinguishable from trait labels?

Test whether concept-use patterns exhibit stable relational structure across agents,
contexts, and historical examples.

### E2. Are local realizations systematically transformed by context?

Compare fixed-trait, situation-only, fixed-regime, and context-transformed schema
models on held-out context families.

### E3. Do transformations affect phenomenology as well as action?

Measure felt grip, salience, confidence, affect, perceived affordances, and reasons.

### E4. Can competent realization be distinguished from counterfeit virtue?

Contrast courage with recklessness, humility with servility, loyalty with complicity,
and perseverance with rigidity using independent outcome and relational criteria.

### E5. Does adaptive metastability predict flourishing?

Test whether dwell times, switching, recovery, and perturbation response predict
phenotype-relative outcomes beyond flexibility and variability baselines.

### E6. Are effects phenotype-relative?

Manipulate or measure capacities, vulnerabilities, dependencies, and action
repertoires. Test whether the same realization has different consequences for
different phenotypes.

### E7. Are costs externalized?

Measure outcomes for every affected party, not only the focal decision maker.

### E8. Does practical wisdom resemble precision or model governance?

Compare models in which experts differ only in precision with models that allow
stakeholder representation, policy generation, temporal depth, and structure change.

### E9. Does relational empowerment predict moral judgment and outcomes?

Estimate self-, other-, and joint capabilities along with veto, voice, exit, and
agenda-setting power.

### E10. Can decorated topology predict moral generalization?

Hold out entire domains and ask whether decorated dynamical correspondences identify
analogous virtues without confusing structurally similar vices.

## 10. Experiment sequence

### Experiment 0 -- Construct and scenario validation

Establish that scenario pairs are understood as involving the same virtue concept but
different actions or meanings. Triangulate expert, affected-party, and participant
judgments. Do not interpret consensus as moral truth.

### Experiment 1 -- Context-transformed virtue schema

Run the sequential-decision study in
[FIRST_EXPERIMENT_REGULATORY_INVARIANCE.md](FIRST_EXPERIMENT_REGULATORY_INVARIANCE.md).
The primary result is out-of-context model comparison.

### Experiment 2 -- Metastability and phenotype-relative outcome

Induce controlled perturbations and compare recovery, calibration, remaining options,
and functioning across phenotype-relevant capacity profiles.

### Experiment 3 -- Counterfeit virtue discrimination

Construct cases in which surface behaviour and self-description match a virtue but
relational consequences reveal recklessness, servility, domination, or complicity.

### Experiment 4 -- Relational empowerment

Use dyadic or small-group tasks in which total control and its distribution can be
manipulated independently. Test whether joint capability, voice, exit, and
non-domination outperform individual empowerment as predictors.

### Experiment 5 -- Practical wisdom model comparison

Compare experts and novices in a domain such as clinical judgment. Manipulate evidence
quality, urgency, affected-party testimony, and reversibility.

### Experiment 6 -- Decorated topology adversarial test

Create synthetic systems with identical topology and opposed meaning, different
topology and equivalent function, flexibility through competence, and flexibility
through instability. Bare topology should fail; the decorated metric must succeed.

### Experiment 7 -- Longitudinal virtue learning

Track how consequences, testimony, exemplars, and institutional cues update local
policies, slow character parameters, and public concept use over time.

### Experiment 8 -- Human--AI plural alignment

Test whether a system can coordinate across heterogeneous human phenotype profiles
without collapsing them into one preference distribution or exploiting their
adaptive preferences.

## 11. Go/no-go gates

### Gate A -- Conceptual coherence

Proceed only if virtue concept, local realization, phenotype, metastability,
functional goodness, and moral goodness can be distinguished without circularity.

### Gate B -- Measurement

Proceed only if phenotype variables, local realizations, perturbation recovery, and
relational outcomes can be measured with acceptable reliability.

### Gate C -- Explanatory necessity

Proceed only if the transformed-schema or dynamical-regime model predicts unseen
contexts better than simpler alternatives after complexity penalties.

### Gate D -- Functional value

Proceed only if metastable measures predict independent flourishing outcomes rather
than merely redescribing behavioural variability.

### Gate E -- Moral discrimination

Proceed only if the framework distinguishes self-serving adaptation, domination, and
counterfeit virtues from morally admissible co-flourishing.

### Gate F -- Topological value

Proceed only if decorated topology adds robust, out-of-domain predictive information
beyond semantics and ordinary dynamics.

### Gate G -- Intervention safety

Deploy steering only if it preserves contestability, detects cost externalization,
resists metric gaming, and supports reversal or repair.

## 12. Paradigm-level falsification conditions

The paradigm should be rejected or radically revised if:

1. virtue concepts show no structure beyond linguistic convention and local context;
2. fixed traits or situation-only models predict as well as the proposed dynamics;
3. metastability has no independent association with phenotype-relative flourishing;
4. phenotype definitions cannot be made independently of desired conclusions;
5. active-inference parameters are unidentifiable or add no explanatory value;
6. moral constraints reduce to arbitrary weights chosen by the modeler;
7. relational empowerment cannot distinguish cooperation from concentrated control;
8. the framework systematically validates stable oppressive norms;
9. topology cannot discriminate opposed semantic or normative organizations;
10. every counterexample can be absorbed by redescribing the latent state;
11. interventions increase apparent alignment by reducing voice, diversity, or exit;
12. results fail to generalize across contexts, populations, and timescales.

## 13. Claim ledger

| Claim | Current status | What would advance it |
|---|---|---|
| Virtue application is context-sensitive | strong philosophical and psychological support | direct process measurements |
| Moral expertise can be metastable optimal grip | established conceptual prior art | discriminating empirical model |
| Virtue concepts transform meaning in application | plausible | held-out context-family prediction |
| Metastability is good for a phenotype | conditional hypothesis | independent flourishing and perturbation evidence |
| Phenotype-relative good grounds moral standing | open normative bridge | explicit defended argument |
| Moral goodness is non-dominating plural co-flourishing | working normative proposal | conflict cases and comparative defense |
| Active inference models practical enactment | formally plausible | superiority to simpler process models |
| Practical wisdom is model governance | novel hypothesis | expert/novice parameter recovery |
| Relational empowerment improves moral analysis | promising correction | dyadic causal evidence |
| Decorated topology measures plural alignment | speculative | adversarial synthetic and real-data validation |
| Safe co-steering is possible | unproven | staged intervention studies with failure monitoring |

## 14. The next three concrete deliverables

1. **A definitions and bridge paper:** formal distinctions among phenotype-relative
   good, moral standing, moral admissibility, and virtue.
2. **A preregistered Experiment 0/1 package:** validated context-transforming scenarios,
   competing models, simulations, and power analysis.
3. **A synthetic decorated-topology benchmark:** cases deliberately constructed to
   defeat bare topology and individual empowerment.

These deliverables attack the paradigm's three greatest risks: normative circularity,
latent-model overfitting, and semantic underdetermination.
