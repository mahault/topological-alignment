# V0: Felt-Goodness Measurement and Identification Pilot

## Status

Protocol and design-recovery simulation complete; no human data have been collected.
This study must pass before the hidden-enablingness and virtue-attractor experiments
are preregistered.

Executable model: `experiments/exp0_measurement_identification.py`

Validator: `benchmarks/validate_v0_measurement_identification.py`

Archived simulation ledger: `benchmarks/v0_measurement_identification_results.json`

## 1. Purpose

The pilot asks whether the following are empirically distinguishable rather than
different names for one response:

1. immediate felt goodness or rightness;
2. immediate comfort and reward;
3. expected social approval;
4. expected consequences at individual, relational, institutional, and ecological
   scales;
5. explicit justification; and
6. reflective judgment after new evidence.

It does **not** ask whether participants possess moral truth, whether a scenario is
objectively virtuous, or whether the active-inference interpretation is correct.

## 2. Construct definitions

**Immediate felt goodness** is the pre-explanatory embodied orientation that a
situation, policy, or form of life is good, right, fitting, or worth sustaining. It is
measured before reasons or detailed consequence prompts.

**Comfort/reward** is anticipated immediate hedonic benefit or relief for the focal
participant.

**Approval** is the participant's expectation that relevant others or institutions
will endorse the policy.

**Expected enablingness** is the participant's forecast of whether the policy will
sustain or degrade viable, recoverable, revisable capacities at each named scale.
These reports are estimates, not the counterfactual ground truth.

**Justification** is a communicable explanation supplied after the initial feeling and
policy. It may be insightful, coordinative, conformist, or confabulatory.

**Reflective judgment** is the repeated evaluation after diagnostic information about
hidden dependencies or consequences.

## 3. Temporal protocol

Each trial uses the following order:

1. **Initial vignette:** enough information for an intuitive response, without virtue
   words or a request to explain.
2. **Immediate response:** felt goodness/rightness, valence, arousal, bodily
   confidence, action readiness, and response time.
3. **Policy:** initial choice plus an option to seek information.
4. **Separated ratings:** comfort/reward, expected approval, and expected consequences
   for each scale, with their order randomized after the immediate response.
5. **Justification:** structured reason categories plus optional short text.
6. **Evidence reveal:** affected-party testimony, dependency information, or a matched
   non-diagnostic update.
7. **Revision:** repeat feeling, policy, consequence forecast, confidence, and reason.

The initial goodness item must be visually and temporally isolated. Consequence and
reason prompts cannot appear on the same screen.

## 4. Measurement-reactivity randomization

Participants are randomly assigned to:

- **immediate-first:** the canonical order above; or
- **reason-first diagnostic arm:** reasons precede the nominal feeling report on a
  small, separately analyzed set of trials.

The second arm is not pooled into the primary analysis. It estimates whether the act
of justification assimilates the reported feeling. If it does, this validates the
need for temporal separation; it does not invalidate the existence of a prior feeling.

Additional order controls independently randomize the order of scale-specific
forecasts and place matched filler trials between initial and post-evidence measures.

## 5. Scenario construction

Use factorially balanced vignettes so that the following vary independently:

- immediate focal reward;
- expected approval;
- individual enabling consequence;
- relational enabling consequence;
- institutional enabling consequence; and
- ecological or long-horizon enabling consequence.

No confirmatory item set may correlate an enabling dimension with reward or approval
strongly enough to make the design ill-conditioned. Each context family must contain
action reversals and at least one hidden-dependency reveal. The relevant phenotype and
affected set must be stated without embedding the desired moral label in the outcome.

An independent panel reviews comprehension, realism, and ambiguity. Panel consensus
is not used as the criterion for goodness.

## 6. Competing measurement models

All comparisons hold out entire context families:

- **M0 comfort/approval:** immediate feeling is predicted by reward and approval.
- **M1 individual:** M0 plus individual enablingness.
- **M2 reported forecast:** M1 plus the participant's aggregate consequence report.
- **M3 nested enablingness:** M2 plus distinct relational, institutional, and
  ecological forecasts.
- **M4 justification leakage:** M3 plus the later justification. This is a diagnostic
  of post-treatment leakage, never an admissible prospective model.

A separate revision model asks whether the discrepancy between initially perceived
and newly revealed consequences predicts change in felt goodness beyond initial
feeling, reward, and approval.

The confirmatory human analysis should use a hierarchical measurement/state-space
model with participant, item, and context-family effects. The current OLS simulation
tests design identifiability only.

## 7. Design-recovery simulation

The simulator generates four adversarial worlds:

1. **Nested truth:** felt goodness tracks perceived multi-scale enablingness.
2. **Proxy-only truth:** only reward and approval generate the feeling.
3. **Confounded design:** enabling predictors nearly duplicate reward and approval.
4. **Prompt contamination:** justification partly changes the subsequently measured
   feeling.

Across 50 deterministic replicates with 180 simulated participants, 12 context
families, and two trials per family:

| Diagnostic | Median result | Interpretation |
|---|---:|---|
| Nested model increment under nested truth | \(\Delta R^2=0.0744\) | detectable beyond individual and reported-forecast model |
| Evidence contribution to revision | \(\Delta R^2=0.1064\) | revealed multi-scale information predicts updating |
| Nested increment under proxy-only truth | \(\Delta R^2=-0.0009\) | negative control does not spuriously favor the theory |
| Confounded design condition number | \(133.40\) | invalid design is correctly flagged |
| Normal feeling–justification correlation | \(r=0.551\) | related but not identical in the constructed model |
| Contaminated feeling–justification correlation | \(r=0.814\) | reason-first measurement is detectably contaminated |
| Enabling coefficient signs recovered | 4 of 4 | all scale effects recover in every nested replicate |

All six preregistered simulation gates passed. These numbers show that the proposed
design can recover the process that was programmed into it. They are not effect-size,
sample-size, or substantive estimates for humans.

## 8. Human pilot go/no-go gates

Proceed to V1 only if:

1. immediate goodness responses have usable test–retest or repeated-item reliability;
2. the balanced item set passes condition-number and manipulation checks;
3. goodness is empirically distinguishable from comfort and approval under held-out
   item-family validation;
4. immediate-first and reason-first arms quantify measurement reactivity;
5. participants can separately forecast at least the individual and relational scales;
6. evidence reveals produce measurable variation in feeling and policy revision;
7. model recovery using the fitted pilot noise, missingness, and response distributions
   meets frozen false-positive and recovery targets; and
8. conclusions remain stable under reasonable alternative codings and exclusion rules.

Failure at gates 1–5 means the stronger goodness hypothesis is not identifiable with
this instrument. Failure at gate 6 may indicate that felt goodness is not
evidence-responsive on the study timescale. Neither result may be repaired by adding
topological complexity.

## 9. What V0 can and cannot establish

A successful V0 establishes measurement separation and design feasibility. It licenses
a preregistered test of whether felt goodness tracks multi-scale enablingness.

It cannot establish that:

- the signal is actually calibrated in ordinary life;
- enablingness constitutes moral goodness;
- a stable disposition is a virtue attractor;
- participant consensus is correct;
- active inference is the uniquely appropriate process model; or
- topology contributes explanatory value.

Those claims belong to V1–V7 in the experiment and pipeline review.
