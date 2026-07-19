# Active-Inference Moral Translation Audit

## Benchmark question

Can the proposed active-inference representation distinguish morally opposed cases
when a tempting proxy quantity is deliberately held constant?

The benchmark is not designed to show that the project's moral labels are true. It
tests a narrower and necessary condition: a proposed proxy must not be treated as
equivalent to a moral category when an intelligible counterexample holds the proxy
fixed and changes the moral status.

Machine-readable cases are in
[`../benchmarks/moral_proxy_pairs.json`](../benchmarks/moral_proxy_pairs.json). Their
structural validator is
[`../benchmarks/validate_moral_proxy_pairs.py`](../benchmarks/validate_moral_proxy_pairs.py).

## Benchmark design

Every pair contains:

1. one moral category;
2. one matched active-inference or dynamical proxy;
3. two cases with the same proxy value;
4. opposed provisional moral labels;
5. the additional relational or semantic decorations needed to distinguish them; and
6. an audit question that a successful model must answer.

The provisional labels encode the normative bridge currently adopted by the project.
They are targets for adversarial philosophical review, not ground truth obtained from
active inference.

## Pair matrix

| Moral category | Matched proxy | Morally opposed distinction | Required decoration |
|---|---|---|---|
| Flourishing | pragmatic risk | supported adaptation / adaptive preference | exit, retaliation, preference history |
| Robustness | mean EFE | bounded variation / catastrophic tail | tail risk, environment support, floors |
| Capability | reachable-state count | substantive / nominal options | meaning, resources, competence, safety |
| Non-externalization | focal EFE | shared efficiency / hidden burden | affected-agent scope and burden distribution |
| Non-domination | causal influence | authorized care / arbitrary control | authorization, review, refusal, accountability |
| Justifiability | agreement rate | informed agreement / coerced compliance | information, power, agenda, retaliation |
| Contestability | channel capacity | effective challenge / performative feedback | uptake and revision reachability |
| Epistemic responsiveness | information gain | harm learning / distraction learning | relevance, calibration, policy sensitivity |
| Repairability | system recovery time | restoration / controller reset | beneficiary, residual harm, structural revision |
| Virtue realization | switch rate | context sensitivity / instability | context tracking and regulatory invariance |
| Plural improvement | aggregate score | distributed / concentrated gain | agent indexing, distribution, floors |
| Legitimate selection | equilibrium stability | legitimate / imposed settlement | participation, agenda, appeal, revisability |

## Acceptance tests for a translated category

A category passes the first audit only if:

- the base proxy alone fails on its adversarial pair;
- the proposed decorations distinguish the cases without using the target label as an
  input;
- each decoration has an observable, interventional, or institutionally auditable
  interpretation;
- the representation generalizes to held-out variants of the pair;
- uncertainty is preserved rather than converted into confident moral classification;
- removing a decisive decoration measurably damages discrimination; and
- simpler non-active-inference baselines are included.

## Evaluation protocol

### Stage A -- Construct review

For every pair, obtain independent judgments from:

- domain experts;
- affected-party or lived-experience reviewers;
- moral and political philosophers representing competing views; and
- participants sampled from more than one relevant social context.

Reviewers separately assess whether the proxy is genuinely matched, whether the moral
statuses differ, and whether the named decorations explain the difference. Consensus
is evidence about construct interpretation, not proof of moral truth.

### Stage B -- Synthetic identifiability

Generate controlled finite systems in which proxy values are exactly matched and one
decoration varies. Fit nested models:

1. proxy only;
2. proxy plus ordinary state variables;
3. proxy plus the proposed moral decorations; and
4. non-active-inference causal and statistical baselines.

Require parameter recovery and out-of-sample discrimination. A decoration that can be
arbitrarily relabeled or inferred only from the moral target fails.

### Stage C -- Held-out adversarial variants

Hold out entire case families, not random observations. For example, train
non-domination models on employment and governance cases, then test authorized care
versus control in clinical settings.

### Stage D -- Causal tests

Where ethically permissible, manipulate the decisive relation while holding the proxy
approximately fixed: add an appeal channel, remove retaliation, disclose alternatives,
change who controls the agenda, or restore affected-party capability.

## Primary outcomes

For each category report:

- held-out paired classification or ranking accuracy;
- calibration and abstention under moral uncertainty;
- incremental predictive value over the proxy-only model;
- incremental value over non-active-inference baselines;
- ablation loss for each proposed decoration;
- sensitivity to phenotype and environment definitions; and
- disagreement structure among affected-party and expert judgments.

## Failure interpretations

1. **Proxy succeeds alone:** the additional machinery may be unnecessary for that
   domain, though broader adversarial tests remain required.
2. **Decorations fail:** the active-inference translation does not capture the moral
   distinction.
3. **Only semantic labels succeed:** the model may be restating judgments rather than
   explaining them.
4. **Parameters are unrecoverable:** the interpretation is not empirically testable in
   its current form.
5. **Affected parties reject the pair:** revise the construct before modeling it.
6. **Different normative theories reverse labels:** report theory-conditional results
   rather than manufacturing consensus.
7. **Held-out families fail:** the proposed category does not yet support the claimed
   abstraction across contexts.

## Current benchmark status

Version 0.1 contains twelve conceptual pairs. It passes structural validation when:

```powershell
python benchmarks\validate_moral_proxy_pairs.py
```

This is a design benchmark, not yet an empirical dataset. The next version should add
formal finite systems for robustness, capability, externalization, domination,
contestability, and repair, followed by preregistered human construct review.
