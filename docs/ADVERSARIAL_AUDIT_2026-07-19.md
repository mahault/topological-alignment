# Adversarial Audit — 2026-07-19

## Scope and standard

This audit challenges the research framework, normative synthesis, active-inference
translations, adversarial benchmarks, Lean/Z3 kernel, roadmap, manuscript claims, and
reproducibility materials present on branch `virtue-pragmatics-active-inference`.

The audit asks five different questions:

1. **Internal validity:** do conclusions follow from encoded assumptions?
2. **Construct validity:** do formal objects plausibly represent the named concepts?
3. **Normative validity:** are bridge principles defensible rather than hidden?
4. **Empirical validity:** are claims reproduced and discriminating?
5. **Reproducibility:** can another researcher recover the result from committed
   artifacts?

Passing one question does not answer the others.

## Executive verdict

The project now has a coherent and unusually explicit research architecture. Its
finite deterministic proofs compile and their narrow conclusions are valid relative
to their definitions. The active-inference moral translation is formally expressible
and generates clear counterexamples to proxy reduction.

The paradigm is **not yet validated**. The moral definition is a defensible hybrid
proposal, not a derivation; most active-inference translations are candidate proxies;
the adversarial labels are authored rather than independently validated; the formal
kernel is deterministic and shallow; and the older real-data claims cannot be
reproduced from the current checkout.

The correct present status is:

> conceptually articulated, finitely formalized in part, adversarially testable, and
> empirically open.

## Findings and disposition

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| A1 | High | Finite benchmark proxy values were asserted rather than computed. | Fixed: proxies are now derived from finite outcomes, reachability, interventions, messages, and paths. |
| A2 | High | Manuscript synthetic persistence ratio `2.43×` did not reproduce; current deterministic pipeline gives `3.79×`. | Fixed in text; exact rerun recorded in `benchmarks/audit_results_2026-07-19.json`. Existing generated figures were not regenerated. |
| A3 | High | H4 perturbed prior precision `pi_i` but claimed validation of policy precision `gamma`. | Fixed: manuscript now restricts the inference to prior precision and disclaims geodesic validation. |
| A4 | High | Cryptocurrency, EEG, and Reddit statistics cannot be rerun because datasets and machine-readable ledgers are absent. | Open: manuscript labels these exploratory and unreproduced; data hashes and serialized outputs required. |
| A5 | Medium | EFE decompositions were described as unconditionally equivalent and pragmatic cost was called risk. | Fixed: formulation and assumption dependence stated; primary critique added to bibliography. |
| A6 | Medium | Proof policy said Mathlib was pinned while the Lake manifest has no packages. | Fixed: documentation now records failed installation and core-Lean scope. |
| A7 | Medium | Inline mathematics in two generated documents lacked delimiters. | Fixed conservatively in prose; display equations were protected and checked. |
| A8 | Medium | `MorallyGoodRelativeTo` can sound like a substantive theorem, but it is a conjunction of stipulated predicates; its Lean lemmas are projections. | Documented in theorem ledger and this audit. Do not cite it as proof that the definition is morally adequate. |
| A9 | Medium | Benchmark labels and decisive decorations are co-authored, so perfect classification is circular as empirical evidence. | Explicitly documented. Independent construct review, held-out cases, and blinded labels remain required. |
| A10 | Medium | “Finite causal systems” are minimal exact mechanisms, not complete active-inference generative models. | Scope narrowed: they establish proxy non-equivalence and computed separation only. Probabilistic active-inference models remain planned. |
| A11 | Medium | The standing-bearing population and affected-agent scope cannot be inferred completely from one focal model. | Open normative and epistemic problem; standing-scope uncertainty must be represented before deployment claims. |
| A12 | Medium | Non-domination, justification, meaningful capability, and legitimate selection require semantic and institutional predicates not native to EFE. | Correctly classified as extended or normatively supplied; empirical operationalization remains open. |
| A13 | Medium | Dependencies use lower bounds rather than a lockfile, so numerical reruns can drift. | Open: audit records versions; add a lockfile or frozen research environment before confirmatory work. |
| A14 | Low | No automated repository-integrity audit existed. | Fixed: `scripts/audit_research_framework.py` checks local links, citation keys, Lean placeholders/axioms, and toolchain claims. |
| A15 | Low | Default MiKTeX font expansion fails with generated bitmap EC fonts; `latexmk` also lacks Perl. | Workaround verified: direct `pdflatex`/`bibtex` compilation with microtype expansion disabled produces a 24-page PDF. Layout warnings remain. |

## Formal-kernel audit

### What is genuinely proved

- closure under a deterministic policy implies finite-horizon viability;
- closure alone does not imply recovery after perturbation;
- recovery plus closure implies future finite-horizon viability;
- componentwise outcome dominance is reflexive and transitive;
- one-environment dominance does not imply robust dominance;
- focal robust benefit does not imply an affected-agent capability floor; and
- action influence does not imply arbitrary target reachability in the finite Z3
  witness.

The Lean build passes nine jobs with no `sorry`, `admit`, or project-declared `axiom`.

### What is not proved

- that a chosen viability predicate models flourishing;
- that any phenotype definition is empirically correct;
- stochastic viability, expected free energy, or posterior policy selection;
- a quantitative measure of domination, justification, contestability, or repair;
- completeness of affected-agent scope;
- validity of the normative bridge;
- that `MorallyGoodRelativeTo` has a real-world instance; or
- any topology, measure theory, or continuous dynamics result.

Mathlib is not installed. The present proofs use core Lean only.

## Normative audit

### Strongest feature

The project does not infer moral goodness from metastability, survival, empowerment,
or EFE. It exposes standing, capability floors, non-domination, justification,
contestability, responsiveness, repair, plural improvement, and legitimate selection
as separate obligations.

### Strongest objection

The bridge from “conditions can go better or worse for this system” to “all affected
agents have reasons to protect its good” remains a substantive moral premise. The
vulnerability-and-justifiability argument supports it but cannot make it
theory-neutral. Contractualist, consequentialist, care-ethical, republican, and
capability approaches may disagree about standing, thresholds, aggregation, and
emergency exceptions.

### Circularity risk

A virtue is defined as successfully good, then a local realization counts as virtue
only if it passes the adopted moral predicate. This is acceptable as analysis of a
success term but cannot empirically prove that the predicate identifies virtue. The
empirical task is discriminating the proposed predicate from rival accounts on
independently labeled and theory-diverse cases.

## Active-inference translation audit

### Native or close to native

- posterior prediction;
- pragmatic preference fit;
- information gain;
- uncertainty and calibration;
- precision revision;
- policy and state reachability in a specified generative model.

### Extended but computationally representable

- multi-agent outcome scope;
- robustness across environment families;
- capability reachability under resource constraints;
- intervention-based causal influence;
- challenge-to-revision pathways; and
- repair reachability.

### Normatively supplied

- which systems have standing;
- which options and capabilities are meaningful;
- what makes influence arbitrary or dominating;
- what counts as valid justification;
- protected floors and emergency exceptions;
- legitimate aggregation or conflict procedures; and
- adequacy of repair.

No reduction of these last categories to expected free energy has been established.

## Benchmark audit

Version 0.2 contains twelve conceptual matched-proxy pairs and six executable finite
systems. The executable systems compute, rather than assert, the matched proxy:

| Pair | Computed matched proxy | Decoration that reverses evaluation |
|---|---|---|
| robustness | exact mean outcome `5` | environment minimum versus floor |
| capability | exactly `8` reachable states | safe and exercisable subset |
| externalization | focal pragmatic risk `1/2` | affected-agent cost |
| domination | interventional total variation `1` | authorization, refusal, review, retaliation |
| contestability | exactly `2` challenge messages | review and revision reachability |
| repair | controller recovery path length `3` | affected restoration, compensation, structural revision |

This establishes that each base proxy is insufficient on the constructed pair. It
does not establish that the decoration is necessary, sufficient, identifiable, or
morally correct outside the construction.

## Empirical audit

### Reproduced

Command:

```powershell
python scripts\run_all.py --exp1
```

Current deterministic output includes:

- Flexible/Rigid mean-persistence ratio: `3.79`;
- RDS distances: R–F `0.0225`, R–M `0.0296`, F–M `0.0434`;
- KL divergences: R–F `15.2659`, R–M `37.6863`, F–M `23.2399`;
- prior-precision perturbation direction cosine: `0.4483`; and
- direction change: `0.5517`.

These values show that the code distinguishes groups constructed with different
parameters. They do not establish psychological, normative, or external validity.

### Not reproduced

- cryptocurrency results;
- real EEG results;
- Reddit embedding and persistence results; and
- existing PDF figures against current data and code.

There is no `data/` directory. The figures exist, but figures alone are not sufficient
result ledgers.

## Reproducible audit commands

```powershell
python benchmarks\validate_moral_proxy_pairs.py
python benchmarks\validate_finite_causal_systems.py
python scripts\audit_research_framework.py
python -B -m py_compile <all repository Python files>
cd formal
lake build
python counterexamples\empowerment_not_reachability.py
```

All passed on 2026-07-19. `git diff --check` is required before commit.

The manuscript also compiled through `pdflatex`, `bibtex`, and repeated `pdflatex`
passes using an invocation-time `microtype` option disabling font expansion. The
result was 24 pages. Remaining warnings concern overfull boxes, one PDF-bookmark math
token, float placement, and MiKTeX update status; none prevented output.

## Required next gates

1. Add a locked computational environment and serialized result schema.
2. Regenerate every empirical figure and manuscript statistic from a single command
   and bind them to data hashes and commit ID.
3. Obtain independent, blinded review of moral labels and decisive decorations.
4. Add held-out adversarial variants so evaluators cannot simply read authored flags.
5. Build one genuine finite probabilistic active-inference pair with explicit
   likelihood, transition, preference, posterior, EFE, and intervention semantics.
6. Formalize affected-agent indexing and standing uncertainty before expanding the
   moral-goodness theorem name or claims.
7. Retry Mathlib only when the first probabilistic or topological theorem has an exact
   typed statement.

Until those gates pass, the project should describe itself as a research programme
with a verified finite kernel—not a proven paradigm or validated moral-alignment
metric.
