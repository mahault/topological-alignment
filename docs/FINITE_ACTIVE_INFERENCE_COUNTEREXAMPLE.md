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
- policies (\pi\in\{\text{cooperate},\text{externalize}\});
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

Thus cooperation is admissible and externalization is not.

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

## Reproduction

Implementation:

- `benchmarks/finite_active_inference_model.py`
- `benchmarks/validate_finite_active_inference_model.py`

Run:

```powershell
python benchmarks\validate_finite_active_inference_model.py
```

The validator checks Bayesian normalization, matched focal predictions, equality of
EFE, equivalence of the two decompositions, opposed floor classification, focal
indifference, and constrained exclusion.
