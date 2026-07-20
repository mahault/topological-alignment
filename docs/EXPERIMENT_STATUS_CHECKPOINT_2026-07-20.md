# Experiment Status Checkpoint — 2026-07-20

## Pause decision

The project is pausing at the end of the constructed single-agent simulation stage.
V0, V1, and V2/V3 have executable result ledgers and passing deterministic
validators. V4 is the next experiment, but its integrated multi-agent generative
model has not been built or run. V5–V7 remain planned designs.

This checkpoint distinguishes three different claims:

1. a measurement or mechanism can be recovered when it is programmed into a
   simulation;
2. the proposed phenomenon occurs in people or social systems; and
3. active inference is necessary or predictively superior as its explanation.

The present experiments support only selected claims of the first kind. No human or
accepted real-world dataset currently validates the paradigm.

## Current experiment register

| Stage | Status | What the result licenses | What it does not license |
|---|---|---|---|
| V0 — felt-goodness measurement | Design-recovery simulation passed | The balanced, temporally separated design recovers programmed differences among immediate feeling, expected consequences, reasons, and later revision, while flagging proxy-only, confounded, and prompt-contaminated controls. | That people possess the proposed felt-goodness signal or that it tracks goodness. |
| V1 — multi-scale enablingness and EFE calibration | Finite simulation passed | Within the declared model, nested counterfactual enablingness outperforms reward/approval proxies and focal success; captured priors and preferences generate systematic failure; phenotype differences can reverse the appropriate policy. | Human or moral validation, a general causal model, or a virtue process. |
| V2/V3 — virtue-like selective stability | Constructed dynamical simulation passed | A slow meaning-to-goodness calibration can resist non-diagnostic noise, recover, transform after diagnostic evidence of hidden harm, retain that transformation, and alter later policy. Dogmatic, unstable, and opportunistic controls are distinguishable. | That real virtues have these dynamics or that the programmed regime is the uniquely correct model of virtue. |
| V2/V3 robustness | 41/48 sampled configurations passed all gates in all three environments | The selective-stability signature occupies a broad, non-vacuous region of the authored parameter box rather than one hand-tuned point. | Population-level robustness, preregistered confirmation, or robustness outside the sampled model family. |
| V2/V3 ablations | Complete | Slow-centre learning, evidence-precision gating, context interactions, and goodness grounding are necessary for the tested signature within this model. | Necessity of an explicit attractor-pull term or an EFE-to-policy mapping. Both corresponding ablations still pass. |
| V4 — dyadic shared semantics and cooperation | Designed; not run | Nothing empirical yet. | Any claim about shared virtue, cooperation, joint goodness, or affected-party protection. |
| V5 — counterfeit virtue and scale conflict | Designed; not run | Nothing empirical yet. | That the normative bridge reliably distinguishes successful coordination from domination or exploitation. |
| V6 — social concept and realization co-evolution | Designed; not run | Nothing empirical yet. | That public virtue concepts and local realizations form separable social attractors. |
| V7 — decorated topological generalization | Designed; not run | Nothing empirical yet. | That topology adds predictive value beyond semantic and ordinary dynamical models. |

## Results that survive the adversarial reading

The strongest current result is a coherence and discriminability result:

> A virtue-like regime can be operationalized as a selectively stable calibration of
> meaning to phenotype-relative enablingness.

In the constructed V2/V3 model, this means that the regime:

- remains comparatively stable under low-precision, non-diagnostic perturbation;
- recovers after that perturbation;
- changes when reliable evidence reveals previously hidden harm;
- retains a context-sensitive portion of that change in later ordinary contexts; and
- changes policy through its changed semantic-pragmatic organization.

This is stronger than equating virtue with behavioral consistency or stability. A
dogmatic regime may be stable but fail to revise; an unstable regime may revise
without maintaining organization; and an opportunistic regime may remain locally
stable while being uncalibrated to enabling consequences.

## Central negative result

The experiment does **not** yet identify the two mechanisms most closely associated
with the headline terminology:

- removing the explicit attractor-pull term does not destroy the selective-stability
  signature; and
- removing the EFE-to-policy mapping does not destroy it either.

Consequently, *attractor* is currently a dynamical description of the observed
regime, not an identified causal term. Active inference is a compatible process
interpretation, but the existing experiment has not shown that EFE is necessary or
that it beats simpler learning and control models.

An EFE-specific experiment must create a task in which the pragmatic/epistemic policy
decomposition makes a distinct held-out or interventional prediction that a direct
value policy cannot reproduce.

## Evidence boundary at the pause

The repository does not yet establish that:

- felt goodness in people estimates multi-scale enablingness;
- real virtues are attractors or selectively stable calibration regimes;
- goodness as enablingness is sufficient for moral goodness;
- different agents possess reciprocally processable shared virtue meanings;
- virtue improves cooperation or protects affected parties;
- active inference is uniquely necessary or predictively superior; or
- topological comparison adds value after semantic and dynamical baselines.

Historic cryptocurrency, EEG, and Reddit outputs are not part of the accepted
evidence base. Their datasets and machine-readable provenance are absent or their
designs are confounded, so they remain exploratory, unreproduced, or retired.

## Next experiment when work resumes: V4

V4 changes the unit of analysis. It requires Agent A, Agent B, an affected party C,
agent-local semantic frames, a translation channel, a causally manipulable joint
policy distribution, and phenotype-indexed consequences.

The simulation should independently randomize:

1. whether A and B use the same virtue sign;
2. whether their expected-consequence profiles are transformably compatible;
3. whether their policies are causally coupled; and
4. whether the resulting joint outcome enables or damages every affected phenotype.

These factors must generate dissociative cases rather than one authored
“cooperation” label:

| Case | Dependence | Shared semantics | Joint attainment | Goodness calibration |
|---|---:|---:|---:|---:|
| Common shock | high | low | variable | variable |
| Different frames, shared meaning | variable | high | high | high |
| Coordinated exploitation | high | high | high for A/B | low for C |
| Good intention, failed translation | low | low | low | locally high, jointly poor |
| Revisable cooperation | high | high | high | improves after diagnostic evidence |

The V4 virtue hypothesis is therefore not “virtuous agents cooperate more.” It is:

> A virtue regime stabilizes cross-agent meaning-to-goodness calibration that
> supports jointly enabling policies, resists non-diagnostic social noise, and
> revises shared meaning when reliable evidence reveals hidden externalization.

The decisive comparison must test the integrated active-inference model against
simpler direct-value, reinforcement-learning, trait-by-situation, and static semantic
baselines. Cooperation remains a vector of joint attainment, causal contribution,
reciprocal processability/readability, coupling effects, and phenotype-relative
enabling consequences; it must not be collapsed into correlated action.

## Reproducible stopping point

The accepted ledgers are:

- `benchmarks/v0_measurement_identification_results.json`;
- `benchmarks/v1_multiscale_enabling_results.json`;
- `benchmarks/v2_virtue_attractor_results.json`; and
- `benchmarks/v2_robustness_sweep_results.json`.

Their validators are:

```text
python -B benchmarks/validate_v0_measurement_identification.py
python -B benchmarks/validate_v1_multiscale_enabling.py
python -B benchmarks/validate_v2_virtue_attractor.py
python -B benchmarks/validate_v2_robustness_sweep.py
python -B scripts/audit_research_framework.py
```

All five checks passed at this checkpoint. The canonical machine-readable disposition
of older and current experiments remains
`benchmarks/experiment_pipeline_ledger.json`.
