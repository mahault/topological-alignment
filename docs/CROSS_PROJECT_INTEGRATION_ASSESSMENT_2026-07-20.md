# Cross-Project Integration Assessment — 2026-07-20

## Decision

The present repository is highly related to five sibling projects, but at a different
level. Those projects provide pieces of the multi-agent process; this repository asks
whether a slowly stabilized semantic-pragmatic regime is calibrated to
phenotype-relative enablingness and can therefore qualify as a virtue realization.

The clean architecture is:

\[
\text{enablingness/goodness target}
\rightarrow \text{agent-local meaning and policy}
\rightarrow \text{externalized anticipations}
\rightarrow \text{cross-agent translation}
\rightarrow \text{joint consequences and interventions}.
\]

No sibling project, by itself, supplies the normative ground. Conversely, this repo's
current V2/V3 simulation is single-agent and cannot yet validate shared meaning or
cooperation. The relationship is therefore complementary rather than redundant.

## 1. `shared-protention-alignment`: very high, direct formal dependency

Relevant artifacts:

- `../shared-protention-alignment/core/morphism.py`
- `../shared-protention-alignment/sheaf/cellular_sheaf.py`
- `../shared-protention-alignment/docs/SYNTHESIS.md`
- `../shared-protention-alignment/docs/VALIDATION.md`

This is the closest match to the claim that meaning is socially shared without being
identical. It defines a directed, entropy-normalized processability score by fitting
one channel across an anticipatory stream:

\[
\operatorname{Proc}_{i\to j}
=\max\left(0,1-
\frac{\min_T\;\mathbb E_t D_{\mathrm{KL}}(\Pi_{j,t}\|\Pi_{i,t}T)}
{I_{\mathrm{time}}(\Pi_j)}\right).
\]

It also distinguishes context-free from context-conditioned translation and
trajectory-specific from operator-level compatibility. This is exactly what V4 needs:
two agents may attach different actions, affects, and local representations to the
same virtue sign while retaining transformable expectations over relevant futures.

The current repo already adapted this implementation in
`benchmarks/coordination_diagnostics.py`, and documents the source in
`docs/SUPPORTED_REPLACEMENTS_LEDGER.md`. That adaptation is appropriate for unit
tests, but V4 should avoid silent divergence. Either import a pinned sibling package
or maintain frozen cross-repo conformance fixtures for processability, reliability,
and held-out channel tests.

Important limit: processability is compatibility of anticipations, not cooperation
and not goodness. Coordinated predation can be mutually processable.

## 2. `externalization-horizon`: high, necessary observability layer

Relevant artifacts:

- `../externalization-horizon/paper/main.tex`
- `../externalization-horizon/simulations/externalization_sim.py`

This project formalizes whether one agent deposits enough information in the shared
environment for another agent to infer its hidden state, and shows that readability
depends on the observer's model. Its use of coefficient of determination rather than
squared correlation is already adapted in `benchmarks/coordination_diagnostics.py`.

This supplies a necessary distinction for shared semantics:

- **externalization:** is there usable evidence of an agent's internal organization?
- **readability:** can this particular observer infer it without bias or scale error?
- **processability:** can the observer translate the inferred anticipatory structure
  into its own frame?

None is sufficient alone. High externalization with a misspecified observer remains
unreadable; high readability may concern a semantically irrelevant state; mutual
readability may support exploitative coordination. V4 should therefore retain these
as separate coordinates rather than collapsing them into “alignment.”

## 3. `empathy-prisonner-dilemma`: high as an adversarial joint-outcome task

Relevant artifacts:

- `../empathy-prisonner-dilemma/src/empathy/prisoners_dilemma/metrics/exploitability.py`
- `../empathy-prisonner-dilemma/src/empathy/prisoners_dilemma/tom/tom_core.py`
- `../empathy-prisonner-dilemma/src/empathy/prisoners_dilemma/tom/inversion.py`

This project records the full joint-action distribution, distinguishes mutual
cooperation from unilateral exploitation, computes policy exploitability, and tests
how opponent modelling and social EFE change action. Those are substantially better
controls than an individual “cooperate” action rate.

It is a useful first V4 substrate because it can test whether a shared virtue sign
such as trust, loyalty, or forgiveness predicts a joint distribution rather than an
isolated action. It also provides ready-made counterfeit cases: stable willingness to
cooperate can be exploitable, and high joint dependence can consist of asymmetric
harm.

Important limit: Prisoner's Dilemma payoffs stipulate the local practical good. They
do not derive phenotype-relative enablingness, standing, or moral goodness. The task
must therefore expose affected-party enabling outcomes separately from the game label.

## 4. `group-formation/tom`: medium-high for social meaning dynamics

Relevant artifacts:

- `../group-formation/tom/README.md`
- `../group-formation/tom/tom/experiments/exp1_disparate_models/analysis.py`
- `../group-formation/tom/tom/experiments/exp2_scripts/`

This project models initially disparate generative models, policy mutual information,
shared scripts, co-location, specialization, perturbation, and post-shock group
identity. That makes it the closest existing substrate for V6: virtue concepts as
socially stabilized but revisable attractors.

Its current emergent metrics must not be imported as definitions. Mutual information
between policies and co-location can be generated by a common environmental cause,
imitation, coercion, or symmetric failure. “Shared script” requires held-out
cross-agent prediction and translation, and “cooperation” requires randomized
coupling and contribution tests. The dynamics are reusable; the construct validation
must be upgraded using this repo's diagnostic vector.

## 5. `aif-meta-cogames`: medium-high as the ecological generalization test

Relevant artifacts:

- `../aif-meta-cogames/src/aif_meta_cogames/aif_agent/theory_of_mind.py`
- `../aif-meta-cogames/docs/DESIGN.md`
- `../aif-meta-cogames/docs/TEAM_INTENTION_DESIGN_20260707.md`

This project has a richer partially observed world, heterogeneous roles, partner-goal
beliefs, diminishing-return complementarity, joint team utility, observed team reward
rate, and stranger adaptation. It is therefore a strong later test of whether a
virtue-semantic model generalizes beyond a two-action game.

The code's team utility and team labels remain authored aggregations. They are useful
policy mechanisms, not definitions of cooperation or goodness. In particular, a team
can optimize its throughput while externalizing costs to non-team phenotypes. This is
where the present repo's affected-set and multi-scale enabling interventions add
something the game agent does not currently contain.

## 6. `Active_Inference_Social_Lock_In`: high for counterfeit social attractors

Relevant artifacts:

- `../Active_Inference_Social_Lock_In/src/pomdp/simple_step.py`
- `../Active_Inference_Social_Lock_In/notes/dynamical_systems_diagnosis.tex`
- `../Active_Inference_Social_Lock_In/archive/v02-norms-as-shared-precision-priors/`

This project examines latent social context, precision, norms, bistability, and
lock-in. It supplies the strongest adversarial comparison for the claim that a
socially shared attractor is a virtue. The same collective stability could be an
epistemic lock-in that suppresses diagnostic evidence.

Its direct role is therefore a V4/V6 negative control: matched regimes should have
similar persistence and shared public semantics while differing in evidence
sensitivity and calibration to affected phenotypes.

## 7. What is already integrated here

This repository already contains more cross-project integration than a filename
search initially suggests:

- `benchmarks/coordination_diagnostics.py` implements finite joint distributions,
  mutual information, total correlation, a limited whole-minus-parts interaction
  contrast, directed processability, reciprocal readability, intervention
  separation, and per-agent EFE coupling contrasts.
- `docs/SUPPORTED_REPLACEMENTS_LEDGER.md` explicitly sources processability from
  `shared-protention-alignment` and readability from `externalization-horizon`.
- `docs/MORAL_GOODNESS_ACTIVE_INFERENCE_TRANSLATION.md` keeps shared-sign inference,
  processability, sheaf consistency, joint attainment, causal contribution, and
  coupling effects distinct.
- `docs/EXPERIMENT_DEFINITION_PIPELINE_REVIEW_2026-07-19.md` already assigns the
  identified multi-agent test to V4 rather than pretending that V1 or V2 measured
  cooperation.

The present gap is not another definition document. It is one executable generative
model in which these variables are generated together and recovered under
interventions.

## 8. Recommended V4 experiment

Build a dyadic semantic-coordination simulation with a latent shared sign \(z_V\),
agent-local semantic maps \(M_i(V,c)\), phenotype-indexed enabling outcomes, and a
joint policy distribution:

\[
Q(\pi_1,\pi_2,Y,z_V,s_{1:2}\mid o_{1:2},c).
\]

Randomize four factors independently:

1. whether the agents share a sign;
2. whether their anticipations are transformably compatible;
3. whether their policies are causally coupled;
4. whether the joint outcome enables or damages each affected phenotype.

This creates decisive dissociations:

| Case | Dependence | Shared semantics | Joint attainment | Goodness calibration |
|---|---:|---:|---:|---:|
| Common shock | high | low | variable | variable |
| Different frames, shared meaning | variable | high | high | high |
| Coordinated exploitation | high | high | high for focal team | low for affected party |
| Good intention, failed translation | low | low | low | locally high, jointly poor |
| Revisable cooperation | high | high | high | improves after diagnostic evidence |

Primary outcomes should remain a vector:

\[
\left(
Q(Y\in Y^*),
\operatorname{Proc}_{\leftrightarrow},
\operatorname{Read}_{\leftrightarrow},
\operatorname{Syn}_Y,
\operatorname{CIF}_{1:2\to Y},
\Delta^{\mathrm{cpl}}_1,
\Delta^{\mathrm{cpl}}_2,
\operatorname{Cal}(g,\operatorname{En}_{P})
\right).
\]

The decisive virtue hypothesis is not “virtuous agents cooperate more.” It is:

> A virtue regime stabilizes a cross-agent meaning-to-goodness calibration that
> supports jointly enabling policies, resists non-diagnostic social noise, and
> revises shared semantics when reliable evidence reveals hidden externalization.

## 9. Integration order

1. Reuse the finite joint-distribution diagnostics already in this repo.
2. Add cross-repo conformance tests against `shared-protention-alignment` rather than
   immediately copying more code.
3. Port a minimal joint-outcome and exploitability environment from
   `empathy-prisonner-dilemma` into an independently specified V4 simulation.
4. Add externalization/readability ablations.
5. Use `Active_Inference_Social_Lock_In` to construct stable-but-uncalibrated social
   controls.
6. Generalize successful identifiers to `group-formation` and then
   `aif-meta-cogames`.

This order tests the theory at increasing ecological complexity while preserving
identifiability. It also prevents “cooperation,” “shared semantics,” and “virtue” from
collapsing into the same outcome label.
