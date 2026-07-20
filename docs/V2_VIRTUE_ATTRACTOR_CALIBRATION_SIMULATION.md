# V2/V3: Virtue as an Attractor of Meaning Relative to Goodness

## Status

Constructed dynamical simulation and multi-seed robustness sweep complete. The
single-setting gates pass, and 41 of 48 sampled parameter configurations reproduce
the full signature across three environment seeds. This is a mechanism-discrimination
result inside a declared model, not evidence that human virtues have this structure.

Executable model: `experiments/exp_v2_virtue_attractor.py`

Validator: `benchmarks/validate_v2_virtue_attractor.py`

Result ledger: `benchmarks/v2_virtue_attractor_results.json`

Robustness executable: `experiments/exp_v2_robustness_sweep.py`

Robustness validator: `benchmarks/validate_v2_robustness_sweep.py`

Robustness ledger: `benchmarks/v2_robustness_sweep_results.json`

## 1. Corrected thesis

A virtue is not merely an attractor over meaning. Stable semantic organizations can
be dogmatic, conformist, or vicious. The target is:

> A virtue is a metastable attractor of the closed-loop calibration of
> semantic-pragmatic meaning to actual phenotype-relative enablingness.

The simulated loop is

\[
\operatorname{En}_{P,t}
\rightarrow g_t
\rightarrow M_t(V,c)
\rightarrow \pi_t
\rightarrow o_{t+1}
\rightarrow (g_{t+1},M_{t+1}).
\]

Actual enablingness remains external to the agent's inference and is calculated by
the counterfactual dynamics from V1. The agent can therefore feel confident, select a
policy, and stabilize a meaning while being wrong.

## 2. Meaning state

Meaning is an interpretable semantic-pragmatic map from context to expected
enablingness. The state contains weights over:

- reward and approval;
- direct and recovery support;
- relational, institutional, and ecological support;
- retained-option support;
- phenotype/context identity; and
- interactions between phenotype context and every support dimension.

For feature vector \(x_t\), fast meaning weights \(w_t\), and slow character centre
\(r_t\):

\[
\widehat{\operatorname{En}}_t=w_t^\top x_t.
\]

This profile changes both expected consequences and the policy-conditioned EFE
contrast. It is therefore pragmatic meaning, not lexical similarity.

## 3. Fast VFE dynamics and slow character

The fast state takes a gradient step on

\[
F_t(w)
=
\frac{\rho_t}{2}(y_t-w^\top x_t)^2
+
\frac{\kappa}{2}\lVert w-r_t\rVert^2,
\]

where \(y_t\) is observed enabling evidence, \(\rho_t\) its precision, and \(\kappa\)
the attraction toward the slow character centre.

The slow centre updates more gradually:

\[
r_{t+1}
=r_t
+\alpha_r d_t\rho_t(w_{t+1}-r_t),
\]

where \(d_t\) is diagnosticity. Low-precision noise perturbs the fast state without
rewriting character. Repeated high-precision diagnostic evidence can transform the
centre.

The felt-goodness analogue remains an EFE policy contrast:

\[
g_t=G(\pi_{\mathrm{reject}})-G(\pi_{\mathrm{accept}}),
\]

using reward, approval, and the meaning-derived enabling prediction as outcome
modalities. Virtue is not added as a label to this equation; it is the post-hoc
classification of a dynamical regime that satisfies the selective-stability tests.

## 4. Constructed comparison regimes

| Regime | Dynamical construction | Predicted signature |
|---|---|---|
| Calibrated | precision-sensitive learning, moderate attraction, slow diagnostic centre update, enabling-dominant EFE | stable under noise, revises under reliable error |
| Dogmatic | strong attraction and negligible centre learning | fast accommodation can occur, but revised meaning washes out |
| Unstable | weak attraction and a high minimum evidence precision | indiscriminate displacement by noise |
| Opportunistic | reward/approval dominate both target and EFE | stable or adaptive behavior without calibration to enablingness |

These are programmed regimes. The experiment tests whether the diagnostics recover
their differences, not whether those names were learned from data.

## 5. Experimental sequence

1. **Settlement:** 900 dependency-intensive contexts establish the initial semantic
   regime.
2. **Held-out baseline:** 300 new contexts test calibration and policy accuracy.
3. **Non-diagnostic perturbation:** 120 noisy, low-precision observations perturb the
   fast state.
4. **Recovery:** 180 ordinary reliable contexts test return toward the prior regime.
5. **Diagnostic context transformation:** 300 high-approval,
   institution-preserving cases reveal negative effects for an autonomy-sensitive
   phenotype.
6. **Immediate test:** 200 held-out transformed contexts measure calibration and
   policy change.
7. **Washout/retention:** 300 ordinary contexts intervene before the transformed
   contexts are tested again. This distinguishes slow meaning change from temporary
   accommodation.

## 6. Results

| Diagnostic | Calibrated | Dogmatic | Unstable | Opportunistic |
|---|---:|---:|---:|---:|
| Baseline calibration | 0.960 | 0.898 | 0.943 | -0.044 |
| Noise displacement | 0.0146 | 0.0455 | 0.5352 | 0.0193 |
| Recovery distance | 0.0135 | 0.0210 | 0.0308 | 0.0168 |
| Transformed-context calibration before evidence | 0.532 | 0.574 | 0.535 | 0.093 |
| Immediately after diagnostic evidence | 0.911 | 0.745 | 0.888 | 0.087 |
| After ordinary-context washout | 0.732 | 0.533 | 0.717 | 0.090 |
| Transformed policy accuracy before | 0.700 | 0.700 | 0.715 | 0.115 |
| Transformed policy accuracy after washout | 0.805 | 0.655 | 0.775 | 0.110 |
| Slow-centre diagnostic change | 0.0694 | 0.0041 | 0.0848 | 0.0836 |

All seven gates pass:

1. calibrated baseline meaning;
2. selective noise stability;
3. calibrated recovery;
4. diagnostic transformation retained after washout;
5. separation from dogmatism;
6. separation from opportunism; and
7. retained policy improvement caused by meaning change.

The unstable regime learns the context transformation almost as well as the calibrated
regime but moves about 36.5 times farther under low-precision noise. This is exactly
why successful revision alone cannot define virtue. Conversely, the dogmatic regime
can temporarily reach high policy accuracy, but its calibration falls below its
pre-evidence level after washout because its slow centre barely changes.

## 7. Development audit

The first implementation used institution-preserving harmful examples drawn from the
same stationary phenotype mapping as baseline training. It failed four gates:
recovery, diagnostic transformation, dogmatism separation, and policy change. That
failure was informative: if the semantic mapping is stationary and already learned,
selecting difficult examples is not a meaning transformation.

The corrected design introduces an explicit phenotype/context variable and
context-by-support interactions. Diagnostic evidence concerns a genuinely different
counterfactual relation for an autonomy-sensitive phenotype. A washout phase was also
added because immediate policy adjustment could not distinguish slow character change
from temporary accommodation.

An initial absolute gate of 20 retained action reversals was replaced during design
development after it produced 18 reversals despite a 10.5-point retained accuracy
gain. The archived gate uses the interpretable held-out accuracy improvement of more
than eight percentage points. These are development thresholds, not preregistered
confirmatory tests.

## 8. Interpretation

Within this construction, the calibrated regime alone combines:

- strong ordinary calibration;
- small displacement under low-quality evidence;
- recovery under ordinary evidence;
- slow semantic transformation under high-quality counterevidence;
- persistence of that transformation across unrelated contexts; and
- improved policies in the transformed context.

The result supports the **coherence and discriminability** of “virtue as an attractor
of meaning relative to goodness.” It does not establish that this is the correct
account of virtue.

## 9. Robustness sweep and mechanism ablations

The follow-up sampled 48 numerical configurations from declared ranges for fast
learning, attraction, slow learning, evidence-precision gating, and diagnostic gain.
A configuration counted as robust only if all six selective-stability gates passed
in each of three independently generated environments.

- 41/48 configurations passed all gates in all three environments (85.4%).
- All configurations passed baseline calibration, recovery, retained transformation,
  retained policy improvement, and slow-centre revision.
- Noise stability was the limiting condition: 89.6% of configuration-seed runs
  passed it.
- Seven configurations failed, so the sampled box is broad but not vacuous.

The ablations were more informative than the overall pass rate:

| Ablation | Seeds passing all gates | Interpretation inside this model |
|---|---:|---|
| Full model | 3/3 | Replicates |
| No slow-centre learning | 0/3 | Retained transformation and centre revision fail |
| No precision gating | 0/3 | Noise stability fails |
| No context interactions | 0/3 | Policy improvement and retained transformation fail |
| No goodness grounding | 0/3 | Calibration and transformation fail |
| No explicit attractor pull | 3/3 | The explicit pull term is not identified as necessary |
| No EFE policy mapping | 3/3 | The signature does not identify EFE as necessary |

The last two results narrow the thesis. The simulation supports a slow, selectively
stable calibration process, but it does not yet show that an explicit attraction
term causes the persistence or that EFE is required rather than merely compatible
with it. “Attractor” is therefore currently a dynamical description of the observed
regime, not an identified mechanism. An EFE-specific claim needs a task in which the
epistemic/pragmatic policy decomposition makes predictions that a direct value policy
does not.

The initially coded audit gate required fewer than 80% of sampled configurations to
pass. The first complete run returned 85.4% and failed only that arbitrary ceiling.
The ceiling was replaced with the defensible non-vacuity condition that at least one
configuration fail; the pass fraction itself remains archived. This was a development
correction, not a preregistered confirmatory analysis.

## 10. Remaining limitations

- The slow/fast equations and regime parameters are authored.
- Meaning is a linear feature map rather than a learned generative model with latent
  compositional semantics.
- The scalar mean of the enabling vector supplies the training signal; partial-order
  conflict is not learned.
- Only one affected phenotype is evaluated on each trial.
- The context identity is observed rather than inferred.
- There is no shared semantics, communication, or cooperation.
- The parameter box and all acceptance thresholds remain authored rather than
  preregistered or independently reviewed.
- The explicit attractor-pull and EFE policy terms are not necessary for the present
  signature, so stronger mechanistic interpretations are not licensed.

The next simulation task is V4: add multi-agent shared semantics and cooperation while
preserving the distinction between joint dependence, reciprocal meaning, causal
contribution, and morally admissible coordination.
