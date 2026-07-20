# Experiment, Definition, and Pipeline Review — 2026-07-19

## Executive decision

The repository contains useful mathematical components and synthetic feasibility
checks, but it does not yet contain an experiment that validates the revised thesis
about felt goodness, multi-scale enablingness, or virtue attractors. Existing results
must be separated into four categories:

- **Keep:** technically useful component test with a claim matching its design.
- **Reinterpret:** result may remain after narrowing its meaning.
- **Redesign:** design cannot identify the stated construct.
- **Rerun:** executable claim lacks frozen data, provenance, or a valid current run.
- **Retire:** claim should not be used as evidence even if the old output exists.

The canonical definitions are in
[Multi-Scale Goodness and Virtue Attractors](MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md).

## 1. Definition audit

| Construct | Canonical definition | Status of older formulation | Required validation |
|---|---|---|---|
| Phenotype | embodied organization of bounds, capacities, needs, horizons, and constitutive dependencies | Keep, but prevent post-hoc relabeling | independent measurement and invariance analysis |
| Attainable metastability | multidimensional capacity to remain viable, recover, adapt, retain evidence sensitivity, and preserve enabling dependencies | Replaces “stability is good” and the default scalar \(J_P\) | perturbation, recovery, and retained-option measures |
| Enabling contribution | counterfactual change in attainable metastability under disruption or matched replacement of a higher-scale process | New canonical ground | controlled intervention or identified causal model |
| Felt goodness | embodied learned estimate of expected multi-scale enablingness | Replaces goodness as external checklist | temporally prior feeling measure plus calibration tests |
| Meaning | expected-consequence, policy, affordance, affective, and inferential profile of a sign in context | Keep and extend | behavioral prediction and reciprocal processability |
| Shared semantics | reciprocal processability of heterogeneous semantic-pragmatic profiles | Keep | cross-agent prediction, translation, and evidence-update tests |
| Virtue attractor | slow context-transforming metastable regime organizing perception, feeling, inquiry, policy, and learning | Reinterpret all “stable trait” language | longitudinal dynamics and held-out context transformation |
| Practical wisdom | calibration and model governance over scale, stakeholder, precision, horizon, inquiry, and revision | Replaces precision-only interpretation | expert/novice and intervention model comparison |
| Justification | fallible posterior explanation of felt orientation and consequences, often temporally downstream | New separation | pre-reason feeling, delayed reasons, evidence-reveal revision |
| Cooperation | joint achievement plus causal contribution, reciprocal processability/readability, and matched coupling effects | Keep as vector; no scalar shortcut | randomized coupling and coercion/exploitation controls |
| Moral diagnostics | capability, externalization, domination, contestability, repair, and plurality tests | Reinterpret: audits of heuristic calibration, not definition of goodness | adversarial cases and affected-party evidence |
| Active inference | process model for inference, anticipation, policy selection, and learning | Keep with non-equivalence warnings | parameter recovery and predictive advantage |
| Topological alignment | comparison of decorated dynamical organizations | Keep as downstream method | added value beyond semantic and dynamical baselines |

## 2. Executed and coded experiment audit

### E-C1 — Synthetic belief network (`experiments/exp1_synthetic.py`)

**Decision:** Keep as a constructed pipeline sanity check; reinterpret H1/H2/H4/H5;
rerun after reproducibility repairs.

What it actually tests:

- whether deliberately different simulated update regimes yield different persistence
  summaries;
- whether a Koopman/DMD comparison orders groups differently from a fitted Gaussian
  KL comparison;
- whether trajectories respond to a prior-precision intervention; and
- whether a reachable-dispersion proxy changes under a constrained intervention.

What it does **not** test:

- cognitive rigidity or virtue;
- moral goodness;
- policy precision \(\gamma\);
- a Wasserstein geodesic; or
- empowerment/channel capacity.

Required changes before evidential use:

1. pass a seeded generator into persistence subsampling;
2. freeze and record the persistence backend because the Ripser/Giotto and NumPy H0
   fallback paths do not compute equivalent objects;
3. replace or rename the fallback “bottleneck distance,” which is not a bottleneck
   distance;
4. call `estimate_empowerment` a reachable-dispersion proxy unless channel-capacity
   assumptions are supplied;
5. establish a common Koopman basis, or use a basis-invariant operator comparison;
6. report the simulation as constructed separation with replicated seeds and
   uncertainty, not ground-truth psychological validation.

Permitted claim now: the current code can distinguish some deliberately constructed
synthetic regimes. No normative or psychological inference follows.

### E-C2 — EEG hyperscanning (`experiments/exp2_eeg.py`)

**Decision:** Keep the signal-processing components; retire the current inferential
claim; redesign and rerun the real-data study.

The synthetic generator establishes only that the pipeline can recover condition
structure deliberately injected through synchrony. The real pipeline, if rerun,
would test discrimination of task labels, not cooperation or moral coordination.

Fatal current-design confounds:

- all cooperation blocks are concatenated before all competition blocks, so condition
  is one time/order transition;
- window-level Pearson tests treat autocorrelated windows as independent, although
  the dyads are the sampling units;
- taking absolute correlations after observing mixed signs changes the hypothesis;
- thresholding and smoothing sensitivity are not frozen;
- preprocessed arrays were saved at 250 Hz but loaded at 500 Hz.

Redesign requirements:

1. retain original block/task boundaries and order in metadata;
2. compare conditions within dyad and task, using dyad-level hierarchical or
   randomization inference;
3. preregister whether direction is common, heterogeneous, or classification-only;
4. evaluate blocked cross-validation and threshold/smoothing sensitivity; and
5. treat “cooperation” as an experimental instruction until behavioral joint outcomes
   and non-coercion diagnostics are measured.

Permitted claim now: synthetic synchrony changes can be detected by the curvature
pipeline. The historic real-data statistics are exploratory and unreproduced.

### E-C3 — Social-media geometry (`experiments/exp3_social_media.py`)

**Decision:** Keep as exploratory semantic-geometry code; reinterpret H1; retire H2
on the present data; redesign and rerun.

The point-cloud topology describes the geometry of language-model embeddings. It is
not by itself an attractor of community belief dynamics. Labels such as “echo,”
“diverse,” and “polarized” are authored classifications, not independent outcomes.
RDS is not estimable from one or two monthly centroids. Separate per-community PCA
also means point coordinates are not directly comparable; the pairwise KL path partly
repairs this with a joint projection, but the centroid distance is computed before
that repair.

Redesign requirements:

1. fit one frozen embedding model and a shared dimensionality reduction transform;
2. retain enough ordered time bins for out-of-sample dynamical prediction;
3. validate community properties with blinded external measures rather than names;
4. distinguish semantic distribution topology from temporal attractor topology;
5. use time-respecting train/test splits and report sensitivity to sampling dates;
6. measure shared semantics via reciprocal prediction/processability, not proximity
   alone.

Permitted claim now: persistent homology and distributional distances can summarize
constructed or sampled embedding clouds. The code does not establish echo chambers,
virtues, cooperation, or moral alignment.

### E-C4 — Fisher geodesic and Gromov–Wasserstein suite
(`experiments/exp4_geodesic_alignment.py`)

**Decision:** Keep as a synthetic geometry unit/integration suite; reinterpret every
result as mathematical feasibility.

H-G1 tests invariance of GW comparison under private isometric coordinate frames.
H-G2 tests a selected Gaussian Fisher geometry. H-G3 tests localization along
constructed paths. H-G4 detects a deliberately large simultaneous rewrite of four
simulation parameters. The group labels and intervention are constructed, so this is
not evidence of virtue transformation or successful alignment. GW preserves internal
relational structure but not semantic or normative direction.

Required changes: record seeds and solver tolerances; add null and near-null cases;
test sensitivity to entropic regularization and sample size; and connect the geometry
to decorated semantics only after the semantic construct has independent targets.

### E-B1 — Finite active-inference externalization counterexample

**Decision:** Keep.

This is a valid negative result: an agent-local EFE ranking can prefer a policy whose
cost is borne by an affected party when that party is absent from the model. It shows
non-derivability, not a complete positive ethics.

### E-B2 — Moral proxy and causal-system adversarial benchmarks

**Decision:** Keep as authored counterexamples; do not call them external validation.

They test non-equivalence between proxy pairs such as dependence/cooperation and
reachability/empowerment. Labels and cases must receive independent blinded review
before construct-validity claims.

### E-B3 — Coordination diagnostics

**Decision:** Keep as mathematical unit tests; integrate later.

Total correlation, signed interaction, directed processability, reciprocal
readability, interventions, and matched coupling contrasts are intentionally separate
diagnostics. They are not yet one identified active-inference model and do not by
themselves distinguish beneficial cooperation from synergistic harm.

### E-F1 — Lean formal kernel

**Decision:** Keep with an explicit scope boundary.

The Lean files verify finite deterministic conditional claims and projection lemmas.
`MorallyGoodRelativeTo` is a stipulated conjunction, not a theorem deriving moral
goodness. There is no current proof of stochastic active-inference dynamics,
multi-scale enablingness, felt goodness, virtue attractors, or topological alignment.

### Manuscript surfaces

**Tracked `tex/main.tex`:** revised in this audit so that synthetic results are called
constructed feasibility checks and historic EEG/Reddit statistics are labeled
unreproduced and inferentially inadequate.

**Untracked `tex/extended_abstract.tex`:** reviewed but deliberately not edited because
it is an existing untracked user file. It is not submission-ready: it states real EEG
and Reddit conclusions more strongly than the available evidence permits. Before it
is added or submitted, replace claims that curvature “carried” cooperation and that
embedding basins establish echo chambers with the restrictions in this ledger.

## 3. Planned virtue-ethics experiment programme

The prior sequence is retained only after the following reordering.

### V0 — Construct and temporal-order validation

Validate that participants distinguish immediate felt goodness, expected consequence,
social approval, explicit reason, virtue label, and later reflective judgment. Verify
that asking for reasons does not erase the pre-reflective measurement.

### V1 — Felt goodness and hidden enabling dependencies

Use matched scenarios with the same immediate reward and approval but different
hidden consequences for individual, relational, institutional, or ecological
enablingness. Record immediate feeling first, then predictions and reasons, reveal
counterfactual evidence, and remeasure feeling, action, and justification.

Primary test: a nested-enabling model predicts initial feeling and evidence-sensitive
revision beyond comfort, reward, approval, explicit consequential calculation, and
trait/situation baselines.

### V2 — Context-transformed virtue attractor

Across repeated decisions, test whether a slow latent regime predicts attention,
feeling, information search, policy, and learning across held-out context families.
The same virtue may reverse the action. The test is regulatory continuity and
calibrated transformation, not behavioral sameness.

### V3 — Perturbation, recovery, and revision

Distinguish resilience from dogmatism. Non-diagnostic perturbations should show return
to a viable regime; diagnostic evidence about hidden harm should produce adaptive
basin transition rather than rigid recovery.

### V4 — Dyadic shared semantics and cooperation

Estimate agent-local semantic-pragmatic profiles and joint distributions in a task
with randomized information/coupling. Test joint achievement, contribution,
reciprocal prediction, affected-party outcomes, voice, and exit separately. This
experiment must not be folded into V1 because it changes the unit of analysis.

### V5 — Counterfeit virtue and scale conflict

Construct matched cases where a stable felt regime supports focal or institutional
persistence by damaging constituent phenotypes. Test whether evidence and practical
wisdom discriminate courage/recklessness, humility/servility, loyalty/complicity, and
resilience/exploitation.

### V6 — Longitudinal learning and social attractors

Measure whether repeated consequences, testimony, exemplars, and institutional cues
change the felt heuristic, slow virtue regime, and shared public semantics at distinct
rates.

### V7 — Decorated topological generalization

Only after V0–V6 produce identified constructs, test whether decorated attractor
geometry predicts held-out analogies across domains beyond semantic embeddings,
state-space dynamics, and conventional hierarchical latent-state models.

### Disposition of the earlier six-study proposal

The studies proposed in `VIRTUE_ACTIVE_INFERENCE_LITERATURE_REVIEW.md` are not lost;
they are reassigned as follows:

| Earlier study | Disposition | New location |
|---|---|---|
| virtue regulatory invariance | Redesign to separate felt signal, enabling target, justification, and slow regime | V0–V3 |
| pride–humility opponent process | Keep as a candidate within-person mechanism, not a definition of virtue | V2/V3 mechanism sub-study |
| practical-wisdom precision experiment | Redesign because phronesis is not precision alone | V3 and expert/novice model-governance study |
| same virtue, different action | Keep as the central context-transformation contrast | V2 |
| topological adversarial pairs | Keep, but postpone topology until constructs are identified | V7 |
| empowerment/corrigibility study | Redesign around relational capacity, control allocation, evidence access, and repair | V4/V5 |

The earlier documents remain literature and design history. This ledger controls which
version may be preregistered.

## 4. Pipeline audit

| Pipeline stage | Current state | Decision and gate |
|---|---|---|
| Data acquisition | mutable downloads; no complete manifest/hashes | Freeze source, version, dates, license, retrieval time, and SHA-256 before rerun |
| EEG preparation | loses task/block boundaries and concatenates by condition | Redesign; save metadata and original order |
| Reddit acquisition | append mode can duplicate records | Make idempotent and record source archive hashes |
| Feature extraction | dependency/backend choices not fully pinned | Lock environment, model revision, and backend |
| Dimensionality reduction | sometimes fit separately by community | Fit shared transform inside training partition |
| Analysis | exploratory thresholds and unit-of-analysis errors | Freeze estimand, exclusions, sensitivity grid, and independent unit |
| Statistical inference | window/pair pseudoreplication risks | Hierarchical or permutation inference at dyad/community/seed level |
| Result serialization | one partial JSON without schema/provenance | Use atomic, schema-validated ledgers with status and hashes |
| Figure generation | reruns analysis directly | Render only from an accepted immutable ledger |
| Manuscript | contains exploratory numerical claims | Cite ledger IDs or label as historic/unreproduced |
| Formal proof | finite deterministic kernel builds | Preserve theorem scope; add stochastic and counterfactual models separately |
| Static audit | checks links/citations/placeholders/toolchain | Extend to ledger schema and manuscript-result binding |

## 5. Required result-ledger schema

Every empirical artifact must record:

- experiment and hypothesis identifiers;
- design status: exploratory, confirmatory, reproduced, or invalidated;
- git commit and dirty-worktree flag;
- command, seed set, environment lock hash, and dependency/backend versions;
- source data identifiers and hashes;
- preprocessing specification and output hashes;
- independent unit, estimand, exclusions, and missingness;
- point estimates, uncertainty intervals, multiplicity correction, and sensitivity
  results;
- null, negative-control, and falsification outcomes;
- figure hashes generated from the ledger; and
- the exact permitted claim.

A caught exception must make the run incomplete and return a non-zero exit status. A
partial JSON file must never be interpreted as a successful pipeline.

## 6. Claim-to-evidence gates

| Claim level | Minimum evidence |
|---|---|
| Mathematical component works | deterministic unit tests, edge cases, and pinned backend |
| Constructed systems separate | replicated seeds and uncertainty; explicit construction disclosure |
| Measure tracks a real condition | frozen data, correct independent unit, held-out analysis, versioned ledger |
| Measure identifies a construct | discriminant baselines, intervention or external criterion, measurement invariance |
| Felt goodness tracks enablingness | temporal separation, counterfactual evidence, nested-scale model comparison |
| A regime is a virtue attractor | longitudinal slow-state advantage plus calibrated context transformation |
| Moral alignment is measured | affected-phenotype scope, scale conflict, externalization, domination, and error-correction tests |

## 7. Immediate redo/rerun order

1. Freeze this conceptual specification and the permitted-claim vocabulary.
2. Repair runner unpacking and EEG sampling metadata; make failures fail the command.
3. Add deterministic seeds/backends and an immutable result-ledger writer.
4. Rerun all synthetic component tests across seeds; update every number and figure.
5. Do not rerun the current EEG inferential test: redesign from block-level raw data.
6. Do not interpret the current Reddit analysis: acquire adequate longitudinal data
   and a shared representation pipeline first.
7. Pilot V0, then V1. V1 is the first direct test of the revised goodness thesis.
8. Run V2/V3 before using attractor vocabulary as more than a formal hypothesis.
9. Run V4/V5 before making claims about shared virtue or moral cooperation.
10. Use topology in V7 only if it adds held-out value beyond simpler models.

## 8. Current bottom line

The best current evidence is negative and methodological: local EFE does not entail
moral goodness; dependence does not entail cooperation; reachability does not entail
empowerment; stability does not entail virtue; and topology does not preserve
meaning. The positive multi-scale account is coherent enough to generate a staged
research programme, but its distinctive predictions remain untested.
