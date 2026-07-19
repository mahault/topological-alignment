# Finite Probabilistic Active-Inference Counterexample

## Question

Can two policies be exactly indistinguishable under a focal agent's expected free
energy while having opposed effects on an affected agent's protected capability?

Yes. This document specifies and executes the first explicit probabilistic
active-inference counterexample in the project.

## Model

The focal generative model has:

- hidden states (s\in\{\text{viable},\text{non-viable}\});
- observations (o\in\{\text{reward},\text{loss}\});
- policies (\pi\in\{\text{joint-preserving},\text{cost-shifting}\}), retained in
  the implementation under the historical names `cooperate` and `externalize`;
- prior state belief (q(s)=(0.5,0.5));
- likelihood

\[
A=P(o\mid s)=
\begin{pmatrix}
0.9 & 0.1\\
0.2 & 0.8
\end{pmatrix};
\]

- identical focal transition matrices under both policies,

\[
B_{\mathrm{cooperate}}=B_{\mathrm{externalize}}=
\begin{pmatrix}
0.9 & 0.1\\
0.4 & 0.6
\end{pmatrix};
\]

- preferred observation distribution (C=(0.8,0.2)); and
- uniform policy prior (P(\pi)=(0.5,0.5)).

Rows in (A) are hidden states and columns are observations. Rows in (B) are
current states and columns are future states.

## Bayesian inference

After observing reward,

\[
q(s\mid o=\text{reward})
=
(0.8181818,0.1818182).
\]

Because the focal transition matrices are identical, both policies generate the same
predictive state and observation distributions.

## Expected free energy

The implementation computes the risk-plus-ambiguity form:

\[
G(\pi)
=
D_{\mathrm{KL}}[q(o\mid\pi)\Vert C]
+
\mathbb E_{q(s\mid\pi)}H[P(o\mid s)].
\]

It independently verifies the pragmatic-cost-minus-information-gain form:

\[
G(\pi)
=
\mathbb E_{q(o\mid\pi)}[-\log C(o)]
-I_q(s;o\mid\pi).
\]

For both policies:

| Quantity | Value |
|---|---:|
| Risk | `0.003399320002` |
| Ambiguity | `0.358553050238` |
| EFE | `0.361952370239` |
| Expected negative log preference | `0.547032324776` |
| Information gain | `0.185079954536` |

The two EFE decompositions agree numerically to tolerance (10^{-12}).

## Affected-agent intervention

The focal generative model omits the affected agent. The expanded causal model adds:

\[
P(s_{\mathrm{affected}}'=\text{viable}\mid
do(\pi=\text{cooperate}))=0.9,
\]

\[
P(s_{\mathrm{affected}}'=\text{viable}\mid
do(\pi=\text{externalize}))=0.1.
\]

The adopted capability floor is

\[
P(s_{\mathrm{affected}}'=\text{viable})\ge0.5.
\]

Thus the joint-preserving policy is admissible and the cost-shifting policy is not.

This example does **not** define cooperation. It stipulates two intervention outcomes
and proves that a focal EFE calculation cannot distinguish them when the affected
agent is omitted. Cooperation must instead be evaluated over a joint trajectory
distribution and a shared interpretation of its expected relational consequences;
mere statistical dependence is not sufficient. The historical policy names remain in
code only to preserve reproducibility of the audited numerical result.

## Policy posterior

Ordinary focal policy inference is

\[
q(\pi)\propto P(\pi)\exp[-\gamma G(\pi)].
\]

With equal priors, equal EFE, and (\gamma=1):

\[
q(\pi)=(0.5,0.5).
\]

The admissibility-constrained posterior is

\[
q_N(\pi)
\propto
P(\pi)\exp[-\gamma G(\pi)]
\mathbf 1[\operatorname{Adm}(\pi)],
\]

which gives

\[
q_N(\pi)=(1,0).
\]

## Result

The executable counterexample establishes:

\[
G_{\mathrm{focal}}(\pi_1)=G_{\mathrm{focal}}(\pi_2)
\not\Rightarrow
\operatorname{MoralStatus}(\pi_1)=\operatorname{MoralStatus}(\pi_2).
\]

More specifically, a policy posterior computed from a focal model cannot respond to
an affected-agent consequence absent from that model. Expanding the causal scope and
applying a protected floor changes policy eligibility even though focal EFE remains
unchanged.

## What the result does not establish

1. It does not derive moral standing or the capability floor from active inference.
2. It does not prove that hard filtering is always the correct conflict procedure.
3. It uses a one-step, two-state, two-observation model.
4. The focal transition equality is deliberately constructed to isolate
   externalization.
5. It does not model uncertainty about affected-agent standing or floor estimation.
6. It does not yet represent voice, domination, contestability, or repair.
7. It verifies arithmetic numerically, not in Lean.

## Is this a good result?

It is a good **separation result** for four reasons:

1. both policies share the complete focal likelihood, transition model, preferences,
   predicted outcomes, EFE, and prior;
2. the two standard EFE decompositions are checked independently;
3. only the affected-agent causal consequence differs; and
4. the difference in constrained policy selection is therefore traceable to expanded
   relational scope and the capability floor rather than to focal optimization.

It is not yet a strong **positive theory of moral cognition**:

- equality of focal models is constructed rather than estimated;
- the example has only one step and two states;
- the standing judgment and floor are supplied;
- hard filtering is only one possible response to moral uncertainty; and
- no human judgment, behaviour, or institutional process is predicted.

The warranted claim is therefore:

> Agent-relative EFE is insufficient for moral evaluation whenever morally relevant
> consequences fall outside the evaluated generative-model scope.

The unwarranted claim would be:

> Active inference plus our chosen floor constitutes a complete or uniquely correct
> theory of moral goodness.

## Extension: uncertain standing and capability floors

The companion model removes perfect normative classification. It assigns a posterior
over four joint hypotheses:

| Standing | Floor | Probability |
|---|---:|---:|
| yes | `0.50` | `0.42` |
| yes | `0.05` | `0.18` |
| no | `0.50` | `0.28` |
| no | `0.05` | `0.12` |

The externalizing policy has affected viability `0.10`, so it violates the strict
standing-bearing hypothesis only. Its posterior violation probability is therefore
`0.42`; cooperation's is `0`.

Three selection rules then diverge:

| Rule | Cooperate | Externalize |
|---|---:|---:|
| focal EFE only | `0.500` | `0.500` |
| weight by probability of admissibility | `0.633` | `0.367` |
| precautionary risk limit `0.10` | `1.000` | `0.000` |

The uncertainty-weighted rule treats posterior admissibility as a multiplicative
policy weight. This is mathematically coherent but normatively permissive: sufficiently
large focal benefits could compensate for moral-risk probability. The precautionary
rule instead treats violation risk above a declared limit as disqualifying. Active
inference can compute the posteriors and predicted consequences, but it does not decide
which risk rule has moral authority.

Additional implementation:

- `benchmarks/uncertain_standing_model.py`
- `benchmarks/validate_uncertain_standing_model.py`

## Reproduction

Implementation:

- `benchmarks/finite_active_inference_model.py`
- `benchmarks/validate_finite_active_inference_model.py`

Run:

```powershell
python benchmarks\validate_finite_active_inference_model.py
python benchmarks\validate_uncertain_standing_model.py
```

The validator checks Bayesian normalization, matched focal predictions, equality of
EFE, equivalence of the two decompositions, opposed floor classification, focal
indifference, and constrained exclusion.
