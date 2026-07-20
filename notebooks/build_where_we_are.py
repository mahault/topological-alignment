"""Build the claim-by-claim research guide notebook.

The generated notebook is intentionally evidence-bound: numerical plots load the
archived result ledgers, while animations are explicitly labelled pedagogical.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "where_we_are.ipynb"


CLAIMS = [
    {
        "id": "C01", "group": "Foundations", "title": "The bearer is relational dynamics",
        "claim": "The object of evaluation is not an isolated belief or action but an agent-in-context dynamical organization that generates perception, feeling, policy, action, and learning.",
        "why": "Equal outputs can conceal different responses to evidence, pressure, and perturbation. Alignment therefore has to concern the process that produces and revises outputs.",
        "formal": r"X=(x,\theta,c);\quad X_{t+1}=F(X_t,o_t,a_t)",
        "assumptions": "A state description can be chosen without making the thesis vacuous; contexts and learning variables are observable or identifiable enough to compare.",
        "test": "Match agents on current outputs, perturb evidence or context, and test whether dynamical descriptors predict divergent future trajectories beyond output baselines.",
        "proof": "A representation theorem or identifiability result must state which changes of coordinates preserve the bearer. Empirical superiority requires held-out predictive comparison.",
        "falsifier": "If current-output models predict perturbation responses as well as dynamical models, the richer bearer adds no explanatory value.",
        "status": "Conceptually supported; not empirically established", "experiment": "V7 after V0–V6",
        "sources": "README; PARADIGM_PROOF_OBLIGATIONS Q1; VIRTUE_ACTIVE_INFERENCE_LITERATURE_REVIEW"
    },
    {
        "id": "C02", "group": "Foundations", "title": "Goodness is phenotype-relative before it is moral",
        "claim": "A functional good is indexed to a phenotype understood as an embodied organization of capacities, needs, vulnerabilities, horizons, and constitutive dependencies—not as a fixed biological essence.",
        "why": "The same process can enable one form of life and disable another. Without an indexed bearer, claims about flourishing silently universalize one agent's needs.",
        "formal": r"P=(K_P,A_P,N_P,T_P,D_P)",
        "assumptions": "Phenotype variables can be specified independently of the outcome being praised; boundaries and needs are not chosen post hoc to save the theory.",
        "test": "Use matched higher-scale processes with different phenotypes and predict sign reversals in enabling contribution. V1 contains a finite constructed reversal.",
        "proof": "Prove conditional statements only: given a declared P and environment class, a regime preserves specified capabilities or viability. This cannot prove moral standing.",
        "falsifier": "If phenotype indexing never changes predictions, it is idle; if P can always be redescribed after failure, the account is tautological.",
        "status": "Defined; finite simulation support", "experiment": "V1 and phenotype-reversal studies",
        "sources": "MULTISCALE_GOODNESS... §§1–2; PARADIGM_PROOF_OBLIGATIONS Q2/Q6"
    },
    {
        "id": "C03", "group": "Foundations", "title": "Metastability is multidimensional and morally neutral",
        "claim": "Attainable metastability is the capacity to remain viable, recover, adapt, retain evidence sensitivity, and preserve meaningful options; mere persistence is neither sufficient nor intrinsically good.",
        "why": "Dogmatism, servility, and oppressive institutions can be stable. A useful functional measure must distinguish recovery from rigidity and stability from adaptive transition.",
        "formal": r"\mathcal A_P=(\mathrm{viability},\mathrm{recovery},\mathrm{adaptation},\mathrm{options},\mathrm{evidence\ sensitivity})",
        "assumptions": "Components and horizons are declared in advance; conflicts are not hidden by an arbitrary scalarization.",
        "test": "Perturb regimes with low-quality noise and diagnostic evidence; require recovery in the first case and retained revision in the second.",
        "proof": "Lean currently proves finite closure/recovery lemmas and counterexamples. A general theorem needs stochastic dynamics, invariance, and robust bounds.",
        "falsifier": "If persistence alone predicts the same outcomes as the multidimensional construct, or if the components cannot be measured independently, the extension fails.",
        "status": "Finite formal kernel + constructed tests", "experiment": "V2/V3; later stochastic proof",
        "sources": "formal/TopologicalAlignment/Viability.lean; MULTISCALE_GOODNESS... §3"
    },
    {
        "id": "C04", "group": "Foundations", "title": "Enablingness is counterfactual contribution",
        "claim": "A higher-scale process enables phenotype P to the extent that removing, scrambling, or replacing it reduces P's attainable metastability under a declared intervention.",
        "why": "Co-occurrence with flourishing does not show contribution. Counterfactual comparison exposes dependencies and prevents stability from being relabelled as benefit.",
        "formal": r"\operatorname{En}^{(\ell)}_P=\mathcal A_P-\mathcal A_P^{do(X^{(\ell)}\ \mathrm{removed/scrambled})}",
        "assumptions": "The intervention is coherent; replacement cases are matched; affected phenotypes and horizons are not omitted; causal identification assumptions hold.",
        "test": "Randomize or simulate removal/scrambling and measure the full attainable-metastability vector. Compare focal and affected-party effects.",
        "proof": "In finite models, enumerate interventions and verify inequalities. In data, identification and uncertainty—not deduction—support the causal claim.",
        "falsifier": "No counterfactual decrement, sign instability across reasonable interventions, or benefits that disappear when affected parties are represented.",
        "status": "Operationalized in a finite simulation", "experiment": "V1; V4 affected-party extension",
        "sources": "V1_MULTISCALE_ENABLING_EFE_SIMULATION; ACTIVE_INFERENCE_NORMATIVE_FORMALIZATION §2"
    },
    {
        "id": "C05", "group": "Experience and meaning", "title": "Felt goodness is a fallible embodied heuristic",
        "claim": "The immediate sense that something is good is a learned, compressed estimate of expected phenotype-relative enablingness—not goodness itself and not necessarily a conscious justification.",
        "why": "Agents cannot calculate multi-scale causal counterfactuals online. Affect can guide action quickly, yet reward, approval, captured priors, and missing stakeholders can systematically miscalibrate it.",
        "formal": r"g_t\approx\mathbb E[\operatorname{En}_P\mid o_{\le t},M_t]\quad\text{with}\quad g_t\ne\operatorname{En}_P",
        "assumptions": "Pre-reflective feeling is temporally and psychometrically separable from reward, approval, prediction, and later reasons.",
        "test": "Measure feeling before justification; reveal hidden enabling consequences later; estimate calibration and revision while independently varying reward and approval.",
        "proof": "This is an empirical measurement/causal claim, not a theorem. Identification requires temporal separation, adversarial conditions, and competing latent-variable models.",
        "falsifier": "Feeling is not separable, has no prospective relation to enabling outcomes, or is fully explained by reward/approval across interventions.",
        "status": "Central hypothesis; simulation design recovery only", "experiment": "V0 human item pilot; V1 mechanism test",
        "sources": "V0_FELT_GOODNESS_MEASUREMENT_PILOT; MULTISCALE_GOODNESS... §3"
    },
    {
        "id": "C06", "group": "Experience and meaning", "title": "Meaning is semantic-pragmatic",
        "claim": "The practical meaning of a virtue sign is an agent- and context-indexed profile of expected consequences, induced policies, affordances, affect, learned cue mappings, parameter beliefs, and precision.",
        "why": "A virtue word matters through what it makes salient, predicts, affords, and motivates. Lexical proximity alone cannot capture action reversal or phenotype-relative consequences.",
        "formal": r"M_i(V,c)=(Q_i(o,s\mid V,c),q_i(\pi\mid V,c),A_i,g_i,\theta_i,\rho_i)",
        "assumptions": "Profile components are behaviorally identifiable and the sign is not defined by the same outcomes used to validate it.",
        "test": "Infer profiles from held-out predictions, actions, feelings, and evidence updates; compare with lexical and reward-only baselines.",
        "proof": "Show identifiability under an explicit generative model and predictive advantage. No mathematical proof can establish that these components exhaust meaning.",
        "falsifier": "The profile fails to predict held-out context transformations or adds nothing beyond embeddings, rewards, and trait scores.",
        "status": "Formal operationalization; linear V2 implementation", "experiment": "V2/V3 and V4",
        "sources": "SUPPORTED_REPLACEMENTS_LEDGER §2; VFE_EFE_SEMANTICS_COOPERATION_LITERATURE_AUDIT"
    },
    {
        "id": "C07", "group": "Experience and meaning", "title": "Shared semantics is processability, not identity",
        "claim": "Agents share a practical meaning when there is evidence for a common sign and their heterogeneous anticipatory profiles are reciprocally translatable on relevant held-out overlaps; identical beliefs or actions are unnecessary.",
        "why": "Different frames and vocabularies can support the same coordinated understanding. Conversely, identical words can conceal incompatible expected futures.",
        "formal": r"\operatorname{Proc}_{i\to j}=\max(0,1-r_{ij}/I_{time}(\Pi_j));\quad \operatorname{Proc}_{\leftrightarrow}=\min_{i\ne j}\operatorname{Proc}_{i\to j}",
        "assumptions": "A single learned channel must generalize to held-out overlaps; target temporal information is non-vacuous; the shared sign is independently inferred.",
        "test": "Fit translation on training contexts, evaluate held-out contexts and directionality, then test whether translated predictions improve joint action.",
        "proof": "Prove channel properties and, where claimed, operator-morphism or sheaf-gluing conditions. Processability remains a diagnostic unless embedded in the likelihood.",
        "falsifier": "Only in-sample translation works, compatibility vanishes under frame changes, or shared labels predict no cross-agent understanding.",
        "status": "Implemented in sibling repo; not yet inside V2", "experiment": "V4 dyadic semantics",
        "sources": "shared-protention-alignment/core/morphism.py; CROSS_PROJECT_INTEGRATION_ASSESSMENT"
    },
    {
        "id": "C08", "group": "Virtue dynamics", "title": "Virtue is a success term",
        "claim": "A stable disposition counts as a virtue only if its organization is successfully calibrated to goodness; attractorhood, confidence, social agreement, and persistence are morally neutral by themselves.",
        "why": "Dogmatism can be stable and socially shared. Calling every attractor a virtue would erase the evaluative grammar of virtue and make vicious stability a success.",
        "formal": r"\mathrm{Virtue}(R,P,c)\Rightarrow \mathrm{Attractor}(R)\land\mathrm{Calibrated}(R,\operatorname{En}_P)\land\mathrm{Admissible}(R)",
        "assumptions": "The goodness and admissibility targets are specified independently of the regime being classified.",
        "test": "Use matched stable regimes—calibrated, dogmatic, opportunistic, exploitative—and ask whether independent outcomes discriminate them.",
        "proof": "Partly conceptual analysis of a success term, partly a normative bridge, and partly empirical discrimination. The implication is not derivable from dynamics alone.",
        "falsifier": "If competent virtue attribution tracks stability alone after consequences are controlled, or independent moral criteria cannot separate counterfeit cases.",
        "status": "Canonical definition; normative bridge open", "experiment": "V2/V3 and V5 counterfeit virtue",
        "sources": "MULTISCALE_GOODNESS... §5; NORMATIVE_BRIDGE"
    },
    {
        "id": "C09", "group": "Virtue dynamics", "title": "Virtue is selective stability relative to goodness",
        "claim": "A virtue realization is a metastable closed-loop calibration of meaning to goodness: it recovers after non-diagnostic noise but transforms after reliable evidence of consequential error.",
        "why": "Neither rigidity nor indiscriminate flexibility is virtuous. The relevant dynamical signature combines persistence, evidence sensitivity, and retained improvement.",
        "formal": r"\operatorname{En}_P\to g\to M(V,c)\to q(\pi)\to o'\to(g',M')",
        "assumptions": "Noise and diagnostic evidence can be independently manipulated; the target is independent; timescales are identifiable.",
        "test": "Settlement → held-out baseline → low-precision perturbation → recovery → diagnostic context transformation → washout → retained test.",
        "proof": "A dynamical theorem would require an invariant/metastable set plus noise and adaptation bounds. Current evidence is computational, not a proof of human virtue.",
        "falsifier": "No recovery/revision dissociation, no retained calibration gain, or simpler memory/reward models predict equally well.",
        "status": "Robust constructed signature: 41/48 configs × 3 environments", "experiment": "V2/V3 complete; mechanism identification next",
        "sources": "V2_VIRTUE_ATTRACTOR_CALIBRATION_SIMULATION; v2_robustness_sweep_results.json"
    },
    {
        "id": "C10", "group": "Virtue dynamics", "title": "The same virtue can reverse action across contexts",
        "claim": "A virtue concept does not prescribe one behavior: courage, humility, or loyalty may rationally reverse action when context changes because meaning is indexed to expected consequences and goodness.",
        "why": "Advancing can be courageous in one context and reckless in another; retreat can be cowardice or responsible protection. Behavioral invariance is the wrong criterion.",
        "formal": r"R_{c_1,i}(V)\ne R_{c_2,i}(V)\quad\text{while}\quad \mathrm{Cal}(R_{c_k,i},\operatorname{En}_{P_i})\ \text{is preserved}",
        "assumptions": "Contexts genuinely change causal relations rather than merely relabel actions; a higher-order continuity criterion is independently defined.",
        "test": "Hold virtue sign constant, manipulate context consequences, and test action reversal with retained regulatory calibration.",
        "proof": "Construct countermodels showing action invariance is neither necessary nor sufficient; empirically estimate cross-context latent continuity.",
        "falsifier": "Virtue competence is better captured by fixed action tendencies, or proposed continuity can absorb arbitrary reversals.",
        "status": "Philosophically plausible; constructed V2 support", "experiment": "V2/V3; human context-family study",
        "sources": "MULTISCALE_GOODNESS... §§4–5; FIRST_EXPERIMENT_REGULATORY_INVARIANCE"
    },
    {
        "id": "C11", "group": "Virtue dynamics", "title": "Active inference supplies pragmatics, not morality",
        "claim": "VFE/EFE can model belief updating, anticipation, policy choice, and learning, but an agent-local generative model does not automatically represent every affected party or derive moral preferences.",
        "why": "EFE evaluates policies relative to encoded preferences and modeled outcomes. Omitted harms have no term; alternative EFE decompositions require declared assumptions.",
        "formal": r"G_i(\pi)=\mathbb E_{Q_i(o,s\mid\pi)}[\log Q_i(s\mid\pi)-\log P_i(o,s)]",
        "assumptions": "One root EFE functional, factorization, preference representation, and sign convention are fixed before interpretation.",
        "test": "Construct policies that are locally EFE-optimal but harm an omitted affected party; compare full and captured models.",
        "proof": "A finite counterexample proves non-derivability from agent-local EFE. Positive morality still requires bridge premises and representation assumptions.",
        "falsifier": "The limitation would fail only if affected standing and consequences followed from the formalism without being encoded—which current counterexamples deny.",
        "status": "Finite counterexample validated; EFE necessity for V2 not identified", "experiment": "Finite counterexample; V4 EFE ablation",
        "sources": "FINITE_ACTIVE_INFERENCE_COUNTEREXAMPLE; Millidge et al. 2021"
    },
    {
        "id": "C12", "group": "Virtue dynamics", "title": "Practical wisdom is model governance",
        "claim": "Phronesis is not merely high precision; it governs which scales, stakeholders, horizons, models, evidence sources, and revision policies should control inference and action.",
        "why": "High precision can intensify error. Wisdom requires deciding when confidence is warranted, whose testimony matters, what is omitted, and when to seek information or revise the model.",
        "formal": r"\phi_t:\ (M,H,\rho,\mathcal S,\mathcal I)\mapsto(M',H',\rho',\mathcal S',\mathcal I')",
        "assumptions": "Governance variables can be distinguished from ordinary inference and are not defined by expert success post hoc.",
        "test": "Compare precision-only, horizon-only, and full governance models on expert/novice and adversarial omission tasks using held-out prediction.",
        "proof": "Model comparison and parameter recovery can support the process claim; normative adequacy needs separate argument.",
        "falsifier": "A simpler precision parameter predicts decisions and revisions equally well, or governance parameters are non-identifiable.",
        "status": "Novel formal hypothesis; untested", "experiment": "V6/V7 practical-wisdom comparison",
        "sources": "PARADIGM_PROOF_OBLIGATIONS Q15; MULTISCALE_GOODNESS... §6"
    },
    {
        "id": "C13", "group": "Social and moral", "title": "Abstract virtue concepts and local realizations co-evolve",
        "claim": "A public virtue concept is a historically and socially stabilized attractor, while a local realization is its context- and phenotype-indexed enactment; social uptake of realizations changes the future concept.",
        "why": "Concepts persist across speakers without fixed identical content, and contested applications reshape what communities take the virtue to demand.",
        "formal": r"V_t\xrightarrow{R_{c,i}}V_{c,i}\xrightarrow{\mathrm{uptake/learning}}V_{t+1}",
        "assumptions": "Public signs and local latent profiles can be separately estimated over time; social influence is not confused with semantic identity.",
        "test": "Longitudinal naming-game/shared-script simulation and later social data: estimate distinct timescales and intervention effects of exemplars/testimony.",
        "proof": "Dynamical existence/stability results can be proven for a declared model; historical-semantic adequacy is empirical and interpretive.",
        "falsifier": "No separable social timescale, local realizations do not predict concept change, or a static lexical model performs as well.",
        "status": "Conceptual model; sibling dynamics available", "experiment": "V6 using group-formation/social-lock-in",
        "sources": "MULTISCALE_GOODNESS... §§5/10; CROSS_PROJECT_INTEGRATION_ASSESSMENT"
    },
    {
        "id": "C14", "group": "Social and moral", "title": "Cooperation is about joint distributions and interventions",
        "claim": "Cooperative coordination cannot be read from one agent's action label. It requires a declared joint outcome, reciprocal semantic/readability relations, outcome complementarity, causal contribution, and per-agent coupling effects.",
        "why": "Correlation and mutual information can result from common shocks, imitation, coercion, or symmetric failure. Joint success without contribution can be accidental.",
        "formal": r"\mathbf C=(Q(Y\in Y^*),\operatorname{Proc}_{\leftrightarrow},\operatorname{Read}_{\leftrightarrow},\operatorname{Syn}_Y,\operatorname{CIF}_{1:n\to Y},\Delta^{cpl}_{1:n})",
        "assumptions": "Y and interventions are declared; PID choice is named; agent-local EFEs are not silently scalarized; coupling randomization is valid.",
        "test": "Factorially vary common shock, shared sign, semantic compatibility, coupling, and outcome. Use randomized decoupling and affected-party outcomes.",
        "proof": "Finite diagnostics can be proven to separate constructed cases. Real cooperation remains an identified empirical classification, not one theorem.",
        "falsifier": "The vector cannot distinguish common-cause dependence, exploitation, and genuine complementary achievement, or simpler outcome measures suffice.",
        "status": "Finite diagnostics implemented; integrated model not built", "experiment": "V4 next",
        "sources": "coordination_diagnostics.py; SUPPORTED_REPLACEMENTS_LEDGER §5"
    },
    {
        "id": "C15", "group": "Social and moral", "title": "Descriptive cooperation is not moral cooperation",
        "claim": "Agents may coordinate, understand one another, and jointly attain a goal while exploiting or dominating others; moral cooperation additionally requires an independently defended admissibility bridge.",
        "why": "Coordinated predation and oppressive institutions can score highly on dependence, processability, readability, and focal attainment.",
        "formal": r"\mathrm{MoralCoop}=\mathrm{Coordination}\land\mathrm{Admissible}_{\mathcal S}",
        "assumptions": "The affected set and admissibility conditions are specified independently; exit, voice, and burden measures are not inferred from focal performance.",
        "test": "Matched coordinated cases with hidden affected-party harm, asymmetric exit, or domination; require descriptive metrics to match while admissibility changes.",
        "proof": "Counterexamples prove non-equivalence. A positive criterion depends on normative premises, not descriptive statistics alone.",
        "falsifier": "Independent moral judgments and affected-party outcomes do not distinguish the adversarial cases, or the bridge is circular.",
        "status": "Non-equivalence supported by finite counterexamples", "experiment": "V4/V5",
        "sources": "ACTIVE_INFERENCE_TRANSLATION_AUDIT; MORAL_GOODNESS_ACTIVE_INFERENCE_TRANSLATION"
    },
    {
        "id": "C16", "group": "Social and moral", "title": "Moral goodness requires an explicit bridge",
        "claim": "Phenotype-relative functional good does not entail moral goodness without defended premises concerning standing, symmetry, capability floors, non-domination, conflict resolution, contestability, repair, and revision.",
        "why": "A system can promote one phenotype's flourishing by sacrificing another. No descriptive free-energy equation determines whose good counts or how conflicts are legitimately resolved.",
        "formal": r"\mathrm{Good}_P+N1\ldots N6\Rightarrow\mathrm{MorallyAdmissible}_{\mathcal S}\quad\text{(bridge premises explicit)}",
        "assumptions": "Normative premises are independently argued, corrigible, and do not merely encode desired verdicts case by case.",
        "test": "Philosophical reflective-equilibrium and adversarial-case programme, plus empirical checks of whether operational predicates track affected-party evidence.",
        "proof": "This is not derivable from VFE/EFE. Formalization can prove consequences of premises; philosophy must defend the premises and their scope.",
        "falsifier": "Bridge principles conflict, generate unacceptable verdicts, cannot handle plural goods, or are adjusted ad hoc after every counterexample.",
        "status": "Explicit but open normative bridge", "experiment": "V5 + philosophical argument",
        "sources": "NORMATIVE_BRIDGE; PARADIGM_PROOF_OBLIGATIONS Q9–Q13"
    },
    {
        "id": "C17", "group": "Social and moral", "title": "Higher-scale stability can externalize its costs",
        "claim": "Processes larger than an individual can enable that individual's existence and feel good, yet preserve themselves by damaging constituents or outsiders; scale must therefore be indexed and affected parties represented.",
        "why": "Institutions, relationships, and ecologies are genuine enabling conditions, but institutional persistence can coexist with constituent capture, exclusion, or ecological depletion.",
        "formal": r"\operatorname{En}^{(\ell)}_{P_i}>0\not\Rightarrow \forall j\in\mathcal S:\operatorname{En}^{(\ell)}_{P_j}\ge0",
        "assumptions": "Affected-set uncertainty is represented; scale-specific outcomes are measurable; benefits and burdens are not collapsed prematurely.",
        "test": "Compare full versus captured affected sets; perturb higher-scale processes and report phenotype-indexed effect vectors and uncertainty.",
        "proof": "Finite counterexamples prove focal benefit does not imply non-externalization. Positive safety requires explicit floor/admissibility assumptions.",
        "falsifier": "Scale and affected-set expansion never changes rankings, or externalization cannot be distinguished from ordinary trade-offs.",
        "status": "Lean/finite counterexamples; positive model incomplete", "experiment": "V1 extension and V5",
        "sources": "RelationalViability.lean; FINITE_ACTIVE_INFERENCE_COUNTEREXAMPLE"
    },
    {
        "id": "C18", "group": "Topology and pluralism", "title": "Bare topology is insufficient",
        "claim": "Attractor topology alone erases semantic and normative direction; comparisons must decorate dynamical objects with meaning, phenotype, consequences, scale, and admissibility information.",
        "why": "Homeomorphic or geometrically similar landscapes can organize opposed meanings and harms; different geometries can realize equivalent functions.",
        "formal": r"\mathcal D=(X,F,\mu,M,P,\operatorname{En},\mathcal S)\quad\text{not merely}\quad(X,F)",
        "assumptions": "Decorations are independently identified and correspondence maps preserve declared invariants rather than being chosen to force similarity.",
        "test": "Adversarial pairs: same topology/opposed meaning and different topology/equivalent function. Compare decorated metrics to semantic and dynamical baselines.",
        "proof": "Counterexamples establish insufficiency of bare topology; metric/pseudometric and composition theorems remain outstanding.",
        "falsifier": "Decorations add no held-out predictive/discriminative value, or correspondence choices make every pair alignable.",
        "status": "Negative diagnosis; positive metric speculative", "experiment": "V7 only after V4–V6",
        "sources": "PARADIGM_PROOF_OBLIGATIONS Q18–Q20; ADVERSARIAL_AUDIT_2026-07-19"
    },
    {
        "id": "C19", "group": "Topology and pluralism", "title": "Alignment under pluralism is preservation under transformation",
        "claim": "Alignment need not mean convergence to identical beliefs. It can mean that relevant semantic-pragmatic and normative organization survives a structure-preserving translation between heterogeneous agents.",
        "why": "Plural agents can coordinate and understand one another while retaining different histories, embodiments, vocabularies, and local models.",
        "formal": r"\Phi_{i\to j}(\mathcal D_i)\simeq_{\mathcal I}\mathcal D_j",
        "assumptions": "The invariant set I is justified in advance; maps generalize; transformation does not erase voice or difference by coercion.",
        "test": "Held-out translation across frames and phenotypes, with adversarial cases where forced convergence improves proximity but harms processability or exit.",
        "proof": "Define a category of decorated systems, prove identity/composition and invariant preservation. None of those general theorems is complete.",
        "falsifier": "Only convergence predicts success, transformations fail held-out, or declared invariants omit morally decisive differences.",
        "status": "Processability proxy exists; decorated theory incomplete", "experiment": "V4 then V7",
        "sources": "shared-protention-alignment; CROSS_PROJECT_INTEGRATION_ASSESSMENT"
    },
    {
        "id": "C20", "group": "Topology and pluralism", "title": "Virtue ethics, consequentialism, and active inference play different roles",
        "claim": "Virtue ethics describes slow character organization, consequential evaluation asks about expected consequences relative to an assumed good, and active inference models the inferential-pragmatic loop connecting character, meaning, policy, consequences, and revision.",
        "why": "Virtue labels underdetermine action; consequences require a prior account of value; active inference supplies neither value nor virtue by itself but can model their enactment and learning.",
        "formal": r"\text{slow regime}\to M(V,c)\to G(\pi)\to q(\pi)\to o\to\text{learning}",
        "assumptions": "The three traditions are not reduced to one another; goodness targets and bridge principles remain explicit.",
        "test": "Compare models: trait-only, consequence-only, active-inference-only, and integrated slow/fast model on context reversal, perturbation, and revision.",
        "proof": "Explanatory necessity is empirical model comparison; conceptual coherence is philosophical; dynamical properties can be proven conditionally.",
        "falsifier": "The integrated model adds no prediction, one simpler account subsumes the others, or its latent components are non-identifiable.",
        "status": "Integrative thesis; partially simulated", "experiment": "V2–V7 model-comparison programme",
        "sources": "README; VIRTUE_ACTIVE_INFERENCE_LITERATURE_REVIEW"
    },
]


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(r"""
# Where we are: a claim-by-claim guide to virtue, goodness, meaning, and active inference

### Written for a philosopher and a mathematician

This notebook states **every canonical claim in the current paradigm**, why we think
it is plausible, what assumptions it needs, how it would be tested, what “proof” can
and cannot mean, what would falsify it, and where the evidence currently stands.

The central proposal is:

> A virtue is a selectively metastable calibration of semantic-pragmatic meaning to
> phenotype-relative goodness, enacted through active-inference dynamics and embedded
> in social systems whose shared meanings and joint consequences must be tested.

**Essential caution.** Mathematical proof shows what follows from encoded assumptions.
Simulation shows what a constructed model can do. Empirical evidence estimates what
happens in a target population. Normative argument defends what ought to count. None
of these substitutes for the others.
"""),
    code(r"""
from pathlib import Path
import json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

ROOT = Path.cwd()
while not (ROOT / 'benchmarks').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if not (ROOT / 'benchmarks').exists():
    raise RuntimeError('Run inside the topological-alignment repository')

def load_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

v2 = load_json('benchmarks/v2_virtue_attractor_results.json')
robust = load_json('benchmarks/v2_robustness_sweep_results.json')

NAVY='#17324d'; BLUE='#2878b5'; TEAL='#2a9d8f'; GOLD='#e9c46a'
ORANGE='#f4a261'; RED='#d65a4a'; GREY='#d9e1e8'; DARK='#52616b'
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({'figure.dpi':120, 'axes.titleweight':'bold', 'text.color':NAVY})

display(HTML('''<style>
.claimbox{border-left:6px solid #2878b5;background:#f4f8fb;padding:12px 16px;margin:12px 0;border-radius:8px}
.warningbox{border-left:6px solid #e9c46a;background:#fff8e5;padding:12px 16px;margin:12px 0;border-radius:8px}
.proofbox{border-left:6px solid #2a9d8f;background:#eef8f5;padding:12px 16px;margin:12px 0;border-radius:8px}
table{font-size:.92em} code{white-space:normal}
</style>'''))
"""),
    md(r"""
## 1. The thesis in one diagram

The upper loop is descriptive and computational. The moral bridge is an explicit
additional layer: active inference does not manufacture it.
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(15,6))
ax.set_xlim(0,15); ax.set_ylim(0,6); ax.axis('off')
nodes=[
 (0.3,3.2,1.9,1.0,'Phenotype P\nneeds + dependencies',TEAL),
 (2.7,3.2,1.9,1.0,'Enablingness En(P)\ncounterfactual target',TEAL),
 (5.1,3.2,1.6,1.0,'Felt good g\nfallible estimate',GOLD),
 (7.2,3.2,1.8,1.0,'Meaning M(V,c)\nconsequences + affordances',BLUE),
 (9.5,3.2,1.6,1.0,'Policy q(pi)\nVFE / EFE',ORANGE),
 (11.6,3.2,2.1,1.0,'Observed consequences\n+ new evidence',RED)]
for x,y,w,h,label,color in nodes:
    ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',fc=color,ec='white',lw=2))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5,fontweight='bold',color=NAVY if color==GOLD else 'white')
for a,b in zip(nodes[:-1],nodes[1:]):
    ax.annotate('',xy=(b[0]-.06,3.7),xytext=(a[0]+a[2]+.06,3.7),arrowprops=dict(arrowstyle='->',lw=2,color=NAVY))
ax.annotate('',xy=(7.9,3.1),xytext=(12.7,3.1),arrowprops=dict(arrowstyle='->',lw=2,color=RED,connectionstyle='arc3,rad=-.35'))
ax.text(10.4,1.8,'learning transforms meaning and the slow character regime',ha='center',color=RED)
ax.add_patch(patches.FancyBboxPatch((3.0,.25),9.0,.85,boxstyle='round,pad=.08',fc='#fff8e5',ec=GOLD,lw=2))
ax.text(7.5,.67,'MORAL BRIDGE: standing • symmetry • floors • non-domination • conflict • contestability • repair',ha='center',fontweight='bold')
ax.annotate('',xy=(4.0,3.1),xytext=(5.0,1.15),arrowprops=dict(arrowstyle='->',lw=2,color=GOLD,linestyle='--'))
ax.text(7.5,5.45,'Virtue hypothesis: selective stability of meaning relative to goodness',ha='center',fontsize=17,fontweight='bold')
ax.text(7.5,4.85,'recover after noise • revise after diagnostic evidence • improve actual enabling consequences',ha='center',color=DARK)
plt.show()
"""),
    md(r"""
## 2. What counts as a proof here?

The word **prove** is dangerously ambiguous across this project. The following
standards are complementary, not rungs on one scale.
"""),
    code(r"""
proofs=pd.DataFrame([
 ['Conceptual','A distinction or implication follows from defended meanings','Counterexamples, conceptual analysis, non-circular definitions','Does not show the world instantiates it'],
 ['Mathematical','A conclusion follows from explicit formal assumptions','Lean theorem, analytic derivation, verified finite enumeration','Does not validate assumptions or moral premises'],
 ['Computational','An executable model has the stated behavior reproducibly','Seeded simulations, ablations, source hashes, adversarial cases','Does not establish humans or unique mechanism'],
 ['Empirical','A claim predicts observations under an identified design','Preregistration, interventions, held-out prediction, uncertainty','Supports rather than deductively proves a population claim'],
 ['Normative','Premises about standing and admissibility survive objection','Argument, reflective equilibrium, hard cases, public justification','Cannot be derived from VFE/EFE or data alone'],
],columns=['Standard','What it establishes','Required instrument','What it cannot establish'])
display(proofs.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
## 2.1 Intellectual provenance: why these are live hypotheses

The synthesis is new, but its components are not invented from nothing:

| Contribution used here | Intellectual anchor | What it supports—and does not support |
|---|---|---|
| Virtue as context-sensitive skilled regulation | Aristotle's *phronesis* tradition; ecological moral expertise | Supports virtue as organized practical sensitivity, not our mathematical model |
| Metastable optimal grip on moral affordances | Hampson, Hulsey, and McGarry, provided PCS manuscript | Closest conceptual prior art for virtue as metastable grip; does not supply goodness calibration or the experiments |
| VFE/EFE perception and policy dynamics | [Discrete active-inference synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/) | Supplies a process formalism, not moral premises |
| Formulation dependence of EFE decompositions | [Millidge, Tschantz & Buckley](https://direct.mit.edu/neco/article/33/2/447/95615/Whence-the-Expected-Free-Energy) | Requires us to declare the root functional and assumptions |
| Information valuable for maintaining a system | [Kolchinsky & Wolpert](https://doi.org/10.1098/rsif.2018.0625) | Motivates viability-relative semantic information; does not establish moral goodness |
| Shared anticipatory organization without identical belief | [Albarracin et al., Shared Protentions](https://doi.org/10.3390/e26040303) | Grounds heterogeneous shared anticipation |
| Alignment as mutual processability | [Takahashi, 2026](https://arxiv.org/abs/2605.29930) | Motivates preservation under transformation; the metric is supplied by the sibling project |
| Cooperative joint action through interactive inference | [Maisto, Donnarumma & Pezzulo](https://arxiv.org/abs/2210.13113) | Supports multi-agent inferential coordination, not moral cooperation |
| Intervention rather than association for causal contribution | [Ay & Polani](https://doi.org/10.1142/S0219525908001465) | Motivates causal information-flow tests |

These sources make the components plausible. They do **not** jointly entail the
paradigm. The integration still has to earn explanatory necessity experimentally.
"""),
    md(r"""
## 2.2 How we chose the definitions

The definitions were not read directly out of active-inference equations, and they
were not chosen merely because they sounded philosophically attractive. We used a
seven-step decision rule:

1. **Preserve the philosophical role.** Virtue must remain a success term; meaning
   must explain practical consequences; cooperation must be genuinely joint.
2. **Keep the target independent of the estimate.** Actual enabling consequences
   cannot be defined by the feeling, preference, or posterior being calibrated.
3. **Map to generated quantities.** Operational terms must arise from model states,
   likelihoods, transitions, preferences, precisions, policies, observations, or
   explicit interventions. Labels added after the equations explain nothing.
4. **Require dissociations.** Definitions must distinguish reward from goodness,
   stability from virtue, dependence from cooperation, and agreement from shared
   meaning.
5. **Demand counterexamples and ablations.** A component earns a mechanistic role
   only when removing it breaks a prediction it was introduced to explain.
6. **Separate proof obligations.** Formal validity, empirical identification, and
   normative justification are assessed independently.
7. **Prefer the weakest adequate claim.** When evidence does not identify a unique
   mechanism, we downgrade from causal explanation to compatibility or description.

This procedure is why several early formulations were rejected rather than patched
with extra labels.
"""),
    code(r"""
decisions=pd.DataFrame([
 ['Goodness','Reward, approval, pleasure, or persistence','Counterfactual phenotype-relative enablingness','The target must remain independent of felt or preferred outcomes.'],
 ['Felt goodness','Goodness itself','Fallible estimate of expected enablingness','Feeling can guide action while still being calibrated or mistaken.'],
 ['Meaning','Lexical similarity or consequences alone','Context-indexed semantic-pragmatic profile','Meaning organizes predictions, affordances, affect, policy, and learning.'],
 ['Virtue','Any stable trait or attractor','Selective calibration of meaning to goodness','Dogmatism is stable; virtue must resist noise and revise under reliable error.'],
 ['Shared semantics','Identical beliefs or correlated actions','Reciprocal processability on relevant overlaps','Different representations can share meaning; common causes can create correlation.'],
 ['Cooperation','A cooperate action, mutual information, or one joint EFE','Vector of joint attainment, translation, readability, complementarity, causal contribution, and per-agent coupling','Coordination can be coercive, exploitative, or externally harmful.'],
 ['Moral goodness','Whatever minimizes VFE/EFE','Functional goodness plus an explicit normative bridge','Inference describes regulation; it does not decide standing, floors, or legitimate trade-offs.'],
],columns=['Construct','Rejected shortcut','Current operational definition','Decision reason'])
display(decisions.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
## 2.3 The mapping: philosophy to model to measurement

The mapping is a chain of operational hypotheses. It is not an identity claim that
Aristotle already meant these equations, nor that a parameter is literally a moral
faculty.

\[
\operatorname{En}_{P,t}
\rightarrow g_t
\rightarrow M_i(V,c)_t
\rightarrow q_i(\pi_t)
\rightarrow o_{t+1}
\rightarrow (g_{t+1},M_{i,t+1},r_{i,t+1}).
\]

The left side supplies the independent success condition. The middle supplies felt
orientation, practical meaning, and action. The return arrow supplies learning and
character change.
"""),
    code(r"""
mapping=pd.DataFrame([
 ['Phenotype / form of life',r'$P=(K_P,A_P,N_P,T_P,D_P)$','Viability bounds, capacities, needs, horizons, dependencies','Indexes whose functioning is evaluated'],
 ['Functional goodness',r'$\operatorname{En}_P$','Counterfactual change under removal or scrambling','Independent calibration target'],
 ['Felt goodness',r'$g_t$','EFE contrast or learned embodied prediction','Fallible practical heuristic'],
 ['Situated construal',r'$w_t$','Fast semantic weights','What the present context appears to mean'],
 ['Character / disposition',r'$r_t$','Slow learned centre','Cross-situational persistence and retained learning'],
 ['Evidence sensitivity',r'$\rho_t$','Observation/evidence precision','Resistance to noise without indifference to evidence'],
 ['Practical relevance',r'$d_t$','Diagnosticity weighting','Whether evidence should revise the slow regime'],
 ['Context sensitivity',r'$x(c)\otimes x(\mathrm{support})$','Context-feature interactions','Why the same virtue may reverse its action'],
 ['Explicit attraction candidate',r'$\kappa\lVert w_t-r_t\rVert^2$','Fast pull toward slow centre','Candidate mechanism; not necessary in current ablation'],
 ['Practical meaning',r'$M_i(V,c)$','Consequences, policies, affordances, affect, cue mappings, beliefs, precision','Content of the virtue for agent i in context c'],
 ['Action selection',r'$q_i(\pi)\propto e^{-\gamma_iG_i(\pi)}$','Policy posterior under a declared EFE formulation','Contextual pragmatic enactment'],
 ['Shared meaning',r'$\operatorname{Proc}_{i\leftrightarrow j}$','Held-out translation of anticipatory streams','Compatibility without identical beliefs'],
 ['Cooperative organization',r'$Q(\pi_1,\pi_2,Y)$ plus interventions','Joint outcomes, causal coupling, readability, complementarity','Joint success without collapsing agent-local effects'],
 ['Moral admissibility','Bridge constraints outside bare VFE/EFE','Standing, symmetry, floors, non-domination, contestability, repair','Why functional success is not yet moral rightness'],
],columns=['Philosophical role','Model object','Operational reading','What the mapping contributes'])
display(mapping.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
<div class="warningbox"><b>Rule of interpretation.</b> A green simulation gate means
“this programmed mechanism produced the declared diagnostic in this model.” It does
not mean “virtue has been discovered,” “moral goodness has been derived,” or “the
mechanism is uniquely necessary.”</div>

## 3. Dependency structure

The moral and topological claims are downstream. If goodness, meaning, or shared
semantics fail, topology cannot rescue the paradigm.
"""),
    code(r"""
fig,ax=plt.subplots(figsize=(15,7)); ax.set_xlim(0,15); ax.set_ylim(0,8); ax.axis('off')
layers=[
 (0.5,6.3,3.0,.9,'C02–C04  PHENOTYPE + ENABLINGNESS',TEAL),
 (4.3,6.3,2.5,.9,'C05  FELT GOODNESS',GOLD),
 (7.6,6.3,2.5,.9,'C06  MEANING',BLUE),
 (11.0,6.3,3.1,.9,'C08–C10  VIRTUE DYNAMICS',ORANGE),
 (2.1,3.8,3.0,.9,'C07  SHARED SEMANTICS',BLUE),
 (6.0,3.8,3.0,.9,'C14  JOINT COORDINATION',RED),
 (9.9,3.8,3.1,.9,'C15–C17  MORAL BRIDGE',GOLD),
 (5.1,1.2,4.8,.9,'C18–C19  DECORATED TOPOLOGICAL ALIGNMENT',DARK)]
for x,y,w,h,label,color in layers:
 ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=color,ec='white',lw=2))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontweight='bold',color=NAVY if color==GOLD else 'white')
arrows=[((3.5,6.75),(4.25,6.75)),((6.8,6.75),(7.55,6.75)),((10.1,6.75),(10.95,6.75)),((8.8,6.25),(3.8,4.75)),((12.3,6.25),(7.5,4.75)),((5.1,4.25),(5.95,4.25)),((9.0,4.25),(9.85,4.25)),((7.5,3.7),(7.5,2.15)),((11.4,3.7),(9.0,2.15))]
for start,end in arrows: ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=2,color=NAVY))
ax.text(7.5,7.65,'What has to work before topology becomes evidential',ha='center',fontsize=17,fontweight='bold')
plt.show()
"""),
    md("## 4. Master claim register\n\nThe 20 claims below are the canonical scope of this guide. Their detailed cards follow."),
    code("""
claims = pd.DataFrame(CLAIMS_DATA)
display(claims[['id','group','title','status','experiment']].style.hide(axis='index').set_properties(**{'text-align':'left'}))
""".replace("CLAIMS_DATA", repr(CLAIMS))),
]


group_descriptions = {
    "Foundations": "What kind of thing is being evaluated, and what functional goodness means.",
    "Experience and meaning": "How goodness is felt and how virtue signs acquire practical content.",
    "Virtue dynamics": "What makes a stable regime a virtue rather than rigidity or opportunism.",
    "Social and moral": "How meanings become shared, actions become joint, and descriptive success becomes morally assessable.",
    "Topology and pluralism": "What topology may compare once semantic and normative structure is identified.",
}

last_group = None
for claim in CLAIMS:
    if claim["group"] != last_group:
        cells.append(md(f"## 5.{list(group_descriptions).index(claim['group'])+1} {claim['group']}\n\n{group_descriptions[claim['group']]}"))
        last_group = claim["group"]
    dependencies = claim.get("dependencies", "See dependency graph and stated assumptions")
    cells.append(md(rf"""
### {claim['id']} — {claim['title']}

<div class="claimbox"><b>Claim.</b> {claim['claim']}</div>

**Why we think so.** {claim['why']}

**Formal object.**

\[
{claim['formal']}
\]

**Assumptions / dependencies.** {claim['assumptions']} {dependencies}

**How we test it.** {claim['test']}

**What “proof” requires.** {claim['proof']}

**What would count against it.** {claim['falsifier']}

**Current status.** **{claim['status']}**

**Experiment / next gate.** {claim['experiment']}

**Where the full argument lives.** `{claim['sources']}`
"""))


cells.extend([
    md(r"""
## 6. Experiment genealogy and setup

The programme is cumulative. Each experiment exists because the previous one leaves
one specific ambiguity unresolved. This table shows how the hypotheses were earned
rather than merely announced.
"""),
    code(r"""
experiment_genealogy=pd.DataFrame([
 ['Legacy E-C1','Can topology distinguish deliberately different update regimes?','Synthetic belief-network trajectories; persistence and operator summaries','Constructed separation is feasible','Topology might describe belief-generating dynamics','Sanity check only; psychological labels withdrawn'],
 ['Legacy E-C2','Can curvature recover injected dyadic synchrony?','Synthetic EEG plus an unreproducible/confounded real pipeline','Injected synchrony is detectable','Dyadic dynamics may carry coordination information','Real cooperation claim retired; redesign required'],
 ['Legacy E-C3','Can embedding-cloud geometry summarize discourse?','Synthetic/Reddit semantic point clouds and persistence','Geometry summarizes clouds','Semantic organization may have measurable structure','Not an attractor; real result unreproduced'],
 ['Legacy E-C4','Can Fisher/GW comparisons work on constructed systems?','Synthetic Gaussian paths and relational structures','Selected geometry computations work','Decorated topology may compare dynamical organizations','Mathematical feasibility only'],
 ['V0','Is felt goodness separable from reward, approval, predictions, and reasons?','Factorially varied vignettes; temporally separated reports; hidden evidence reveal','Constructed design is identifiable','Goodness can be modeled as a fallible felt heuristic','Human item pilot still required'],
 ['V1','Can that heuristic be calibrated to an independent functional target?','Finite phenotypes and higher-scale processes; remove/scramble interventions; VFE inference; EFE policy','Counterfactual enablingness, capture failures, and phenotype reversals reproduce','Felt goodness can estimate phenotype-relative enablingness','Finite-model evidence only'],
 ['V2/V3','What dynamics distinguish virtue from rigidity, instability, and opportunism?','Four authored regimes; noise, recovery, diagnostic transformation, washout','Selective-stability diagnostics separate the regimes','Virtue may be slow calibration of meaning to goodness','Constructed mechanism-discrimination result'],
 ['Robustness','Is V2 one hand-tuned point?','48 sampled parameter configurations in three environments; six gates','41/48 pass every gate in every environment','The signature occupies a broad sampled region','Authored parameter box'],
 ['Ablations','Which components actually explain the V2 signature?','Remove slow learning, precision gating, context interactions, goodness grounding, explicit pull, or EFE mapping','First four break; explicit pull and EFE removal do not','Narrow virtue mapping and downgrade unsupported mechanism claims','Architecture-relative necessity only'],
 ['V4 next','Can shared meaning, coupling, and goodness be experimentally separated?','Factorial dyads varying sign, translation, coupling, and affected-party outcomes','Not yet run','Virtue should stabilize jointly enabling shared meaning','Decisive next simulation'],
 ['V5 planned','Can stable shared coordination still be morally counterfeit?','Matched domination, exploitation, exclusion, and repair cases','Not yet run','Functional success is not sufficient for moral virtue','Tests normative bridge'],
 ['V6 planned','How do public virtue concepts and local realizations co-evolve?','Longitudinal testimony, exemplars, institutions, perturbations, and multiple learning rates','Not yet run','Concept and enactment should change on separable timescales','Social-attractor test'],
 ['V7 planned','Does topology add anything after constructs are identified?','Held-out adversarial pairs; semantic and ordinary dynamical baselines','Not yet run','Decorated topology should add predictive value','Topology removed if baselines tie or win'],
],columns=['Experiment','Question inherited from prior stage','Setup','Result/status','Hypothesis it motivates or tests','Permitted inference'])
display(experiment_genealogy.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
### Setup template used for every experiment

| Field | What must be specified before interpreting a result |
|---|---|
| Unit | phenotype, agent, dyad, group, trajectory, or community |
| Manipulation | what is randomized, intervened on, removed, scrambled, or revealed |
| Independent target | what the agent's estimate or behavior is calibrated against |
| Observables | feelings, beliefs, policies, actions, joint outcomes, and revision |
| Controls | reward, approval, common causes, coercion, exploitation, noise, and simpler models |
| Gate | the declared result that must hold |
| Falsifier | the observation that makes us reject or narrow the claim |
| Inference class | conceptual, mathematical, computational, empirical, or normative |
"""),
    md(r"""
## 6.1 What the individual experiments actually look like

### V0 — separating felt goodness from its confounds

Each trial temporally separates immediate feeling, expected consequences, social
approval, justification, evidence reveal, and later revision. Hidden enabling
dependencies let reward and approval disagree with actual consequences.
"""),
    code(r"""
fig,ax=plt.subplots(figsize=(14,3.6)); ax.set_xlim(0,14); ax.set_ylim(0,3); ax.axis('off')
steps=[('Scenario',.3,TEAL),('Immediate\nfeeling',2.1,GOLD),('Reward +\napproval',4.0,ORANGE),('Expected\nconsequences',5.9,BLUE),('Initial\njustification',7.9,DARK),('Hidden evidence\nrevealed',9.8,RED),('Feeling + reason\nrevision',11.9,TEAL)]
for label,x,color in steps:
 ax.add_patch(patches.FancyBboxPatch((x,1.0),1.5,.85,boxstyle='round,pad=.05',fc=color,ec='white'))
 ax.text(x+.75,1.43,label,ha='center',va='center',fontsize=9,fontweight='bold',color=NAVY if color==GOLD else 'white')
for a,b in zip(steps[:-1],steps[1:]): ax.annotate('',xy=(b[1]-.05,1.43),xytext=(a[1]+1.55,1.43),arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.8))
ax.text(7,2.55,'V0 trial timeline: measurement precedes explanation',ha='center',fontsize=15,fontweight='bold')
plt.show()
"""),
    md(r"""
### V1 — counterfactual enablingness and captured active inference

V1 holds the finite world explicit. A higher-scale process is removed or scrambled;
phenotype outcomes are recomputed; an agent infers hidden process type via VFE and
selects policies via EFE. Reward-dominant preferences and captured priors create
predictable miscalibration.
"""),
    code(r"""
v1=pd.DataFrame([
 ['Ground truth','Counterfactual enabling vector En(P)','Independent simulation intervention'],
 ['Inference','Posterior over hidden process','VFE/Bayesian update'],
 ['Feeling proxy','EFE contrast: reject minus accept','Policy-relative, fallible'],
 ['Failure controls','Captured prior; reward/approval dominance','Can feel good while actual En(P)<0'],
 ['Crucial reversal','Same process, different phenotype','Goodness is indexed rather than universal'],
],columns=['Layer','Quantity','Purpose'])
display(v1.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
### V2/V3 — selective stability under noise and evidence

The sequence is fixed: settlement, held-out baseline, low-precision perturbation,
recovery, diagnostic context change, immediate test, ordinary-context washout, and
retained transformed-context test.
"""),
    code(r"""
metrics=pd.DataFrame(v2['primary_results']).T
regimes=['calibrated','dogmatic','unstable','opportunistic']; cols=[TEAL,GOLD,RED,DARK]
fig,axes=plt.subplots(2,2,figsize=(13,8)); x=np.arange(4); w=.34
axes[0,0].bar(regimes,metrics.loc[regimes,'baseline_calibration'],color=cols); axes[0,0].axhline(.75,ls='--',color=NAVY); axes[0,0].set_title('Baseline calibration')
axes[0,1].bar(regimes,metrics.loc[regimes,'noise_displacement'],color=cols); axes[0,1].axhline(.08,ls='--',color=NAVY); axes[0,1].set_title('Noise displacement (lower is better)')
axes[1,0].bar(x-w/2,metrics.loc[regimes,'trap_calibration_before'],w,label='before',color=GREY); axes[1,0].bar(x+w/2,metrics.loc[regimes,'trap_calibration_retained'],w,label='retained',color=BLUE); axes[1,0].set_xticks(x,regimes); axes[1,0].set_title('Transformed-context calibration'); axes[1,0].legend()
axes[1,1].bar(x-w/2,metrics.loc[regimes,'trap_policy_accuracy_before'],w,label='before',color=GREY); axes[1,1].bar(x+w/2,metrics.loc[regimes,'trap_policy_accuracy_retained'],w,label='retained',color=ORANGE); axes[1,1].set_xticks(x,regimes); axes[1,1].set_title('Transformed-context policy accuracy'); axes[1,1].legend()
for ax in axes.flat: ax.tick_params(axis='x',rotation=16)
fig.suptitle('Archived V2/V3 results: four constructed regimes',fontsize=16,fontweight='bold'); fig.tight_layout(); plt.show()
"""),
    code(r"""
passed=robust['robust_configuration_count']; total=robust['configuration_count']
rates=pd.Series(robust['per_gate_pass_rates']).sort_values()
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].pie([passed,total-passed],labels=['robust','failed ≥1 seed'],colors=[TEAL,GREY],autopct='%1.1f%%',startangle=90,wedgeprops={'width':.42,'edgecolor':'white'}); axes[0].text(0,0,f'{passed}/{total}',ha='center',va='center',fontsize=23,fontweight='bold'); axes[0].set_title('Configurations passing all three environments')
bars=axes[1].barh(rates.index,rates.values,color=[RED if v<.95 else BLUE for v in rates]); axes[1].set_xlim(0,1.05); axes[1].bar_label(bars,labels=[f'{v:.1%}' for v in rates.values]); axes[1].set_title('Gate pass rates across configuration × environment')
fig.tight_layout(); plt.show()
"""),
    code(r"""
labels={'full_model':'Full model','no_slow_centre_learning':'− slow centre','no_precision_gating':'− precision gate','no_attractor':'− explicit pull','no_context_interactions':'− context interactions','no_goodness_grounding':'− goodness ground','no_efe_policy_mapping':'− EFE mapping'}
gate_names={'baseline_calibration':'baseline','noise_stability':'noise','recovery':'recovery','retained_transformation':'transform','retained_policy_improvement':'policy','slow_centre_revision':'centre'}
abl=pd.DataFrame({labels[k]:v['gate_pass_rates'] for k,v in robust['ablations'].items()}).T.rename(columns=gate_names)
plt.figure(figsize=(12,6)); sns.heatmap(abl,annot=True,fmt='.2g',vmin=0,vmax=1,cmap=sns.color_palette([RED,ORANGE,GOLD,'#b7ddd3',TEAL],as_cmap=True),linewidths=1)
plt.title('Ablations: explicit pull and EFE mapping are not yet necessary',fontsize=15,fontweight='bold'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.show()
"""),
    md(r"""
### Why the mechanism-ablation panel matters for virtue

Section 5 of the shorter dashboard is not merely a technical sensitivity check. It
asks which parts of the construction earn the right to interpret the simulated
regime as **virtue-like**, rather than as generic adaptive control.

| Component | Philosophical work it was meant to do | Ablation result | Consequence for the virtue claim |
|---|---|---|---|
| Slow-centre learning | Character should persist across situations yet retain genuine learning | Removing it fails | Supports a slow dispositional layer as necessary *in this construction* |
| Evidence-precision gating | Practical wisdom should distinguish noise from trustworthy counterevidence | Removing it fails | Supports selective, not indiscriminate, stability |
| Context interactions | The same virtue may rationally reverse action when consequences change | Removing them fails | Supports context-sensitive practical meaning rather than a fixed rule |
| Goodness grounding | Virtue is a success term, not merely a coherent disposition | Removing it fails | Without calibration to an independent good, the regime cannot count as virtue-like |
| Explicit attraction pull | “Attractor” was proposed as the mechanism of return | Removing it still passes | We cannot claim the explicit pull causes the signature; “attractor” is currently descriptive |
| EFE policy mapping | Active inference was proposed as the distinctive pragmatic bridge | Removing it still passes | EFE is compatible with the result but not yet uniquely explanatory |

Three logical cautions matter:

1. **Failure under ablation is architecture-relative necessity**, not a theorem that
   every possible virtue must contain that exact parameter.
2. **Survival under ablation refutes necessity for this test.** It prevents us from
   presenting attractive terminology as an identified cause.
3. **No ablation proves moral sufficiency.** Slow learning, precision gating, context
   sensitivity, and goodness calibration could still occur in a locally successful
   but dominating social system. V4 and V5 must test shared consequences and the
   normative bridge.

The positive virtue mapping is therefore currently:

\[
\underbrace{r_t}_{\text{character}}
+\underbrace{\rho_t d_t}_{\text{practical evidence sensitivity}}
+\underbrace{M(V,c)}_{\text{contextual practical meaning}}
+\underbrace{\operatorname{Cal}(g,\operatorname{En}_P)}_{\text{success condition}}.
\]

That combination explains why a virtue is stable without being rigid, adaptable
without being fickle, context-sensitive without being empty, and good without being
defined by the agent's confidence. Whether it is sufficient for **moral** virtue is
still open.
"""),
    md(r"""
### Animation 1 — selective stability (pedagogical one-dimensional projection)

This animation is **not an additional result**. It explains the diagnostic logic in
one dimension. Low-quality noise should move an unstable regime but not rewrite a
calibrated slow centre. Reliable evidence should transform the calibrated centre but
not the dogmatic one.
"""),
    code(r"""
rng=np.random.default_rng(7); T=100; truth=np.full(T,.65); observed=truth.copy(); precision=np.ones(T)*.85; diagnostic=np.zeros(T)
observed[25:48]=rng.normal(-.25,.45,23); precision[25:48]=.08
truth[62:]=-.55; observed[62:]=truth[62:]+rng.normal(0,.08,T-62); precision[62:]=.95; diagnostic[62:88]=1
configs={'calibrated':(.20,.10,.035,.12),'dogmatic':(.20,.40,.001,.12),'unstable':(.32,.005,.06,1.0)}
traces={}
for name,(alpha,kappa,beta,minp) in configs.items():
 fast=np.zeros(T); slow=np.zeros(T); fast[0]=slow[0]=.55
 for t in range(T-1):
  p=max(precision[t],minp); fast[t+1]=fast[t]+alpha*p*(observed[t]-fast[t])+kappa*(slow[t]-fast[t]); slow[t+1]=slow[t]+beta*diagnostic[t]*p*(fast[t+1]-slow[t])
 traces[name]=(fast,slow)
fig,ax=plt.subplots(figsize=(11,5)); ax.set_xlim(0,T-1); ax.set_ylim(-1.1,1.1); ax.axvspan(25,48,color=GREY,alpha=.45,label='low-precision noise'); ax.axvspan(62,88,color=GOLD,alpha=.28,label='diagnostic evidence'); ax.plot(truth,color='black',lw=2,ls=':',label='actual target')
line_objs={}
for name,color in zip(configs,[TEAL,GOLD,RED]):
 line_objs[name]=ax.plot([],[],color=color,lw=2.5,label=name+' fast meaning')[0]
ax.set_title('Pedagogical animation: selective stability'); ax.set_xlabel('time'); ax.set_ylabel('meaning / target projection'); ax.legend(loc='lower left',ncol=2,fontsize=8)
def update(frame):
 for name,line in line_objs.items(): line.set_data(np.arange(frame),traces[name][0][:frame])
 return tuple(line_objs.values())
ani=FuncAnimation(fig,update,frames=range(2,T,2),interval=90,blit=True); animation_html=ani.to_jshtml(fps=12,default_mode='loop'); plt.close(fig); display(HTML(animation_html))
"""),
    md(r"""
### V4 — the next decisive multi-agent experiment

V4 independently randomizes: shared sign, frame compatibility, causal coupling, and
affected-party outcome. This creates cases in which dependence, understanding,
achievement, and goodness come apart.
"""),
    code(r"""
v4_cases=pd.DataFrame([
 ['Common shock','High','Low','Variable','Variable','Dependence ≠ cooperation'],
 ['Different frames / shared meaning','Variable','High','High','High','Translation without convergence'],
 ['Coordinated exploitation','High','High','High focal','Low affected','Coordination ≠ morality'],
 ['Good intention / failed translation','Low','Low','Low','Mixed','Good local orientation can fail jointly'],
 ['Revisable cooperation','High','High','High','Improves','Target virtue signature'],
],columns=['Case','Dependence','Shared semantics','Joint attainment','Goodness calibration','Purpose'])
display(v4_cases.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
### Animation 2 — shared meaning without identical representations (pedagogical)

Agent A and Agent B use different local codes. A fixed translation channel maps B's
anticipation into A's frame. When the context changes, both reverse their practical
expectation while remaining mutually translatable. Surface equality is never required.
"""),
    code(r"""
frames=60; ts=np.linspace(0,1,frames); a=np.column_stack([.15+.7/(1+np.exp(14*(ts-.55))), .85-.7/(1+np.exp(14*(ts-.55)))])
Tmap=np.array([[0,1],[1,0]])
b=a@Tmap; translated=b@Tmap
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for ax,title in [(ax1,'Agent A frame: protect / challenge'),(ax2,'Agent B frame: confront / shelter')]: ax.set_ylim(0,1); ax.set_title(title); ax.set_ylabel('predicted policy probability')
bars1=ax1.bar(['protect','challenge'],a[0],color=[TEAL,ORANGE]); bars2=ax2.bar(['confront','shelter'],b[0],color=[RED,BLUE]); txt=fig.suptitle('')
def update_sem(k):
 for bar,h in zip(bars1,a[k]): bar.set_height(h)
 for bar,h in zip(bars2,b[k]): bar.set_height(h)
 residual=np.abs(a[k]-translated[k]).sum()
 txt.set_text(f'Different codes, shared practical structure — translation residual {residual:.3f}')
 return (*bars1,*bars2,txt)
ani2=FuncAnimation(fig,update_sem,frames=range(frames),interval=100,blit=False); sem_html=ani2.to_jshtml(fps=10,default_mode='loop'); plt.close(fig); display(HTML(sem_html))
"""),
    md(r"""
## 7. Formal theorem inventory: what is genuinely machine checked

These theorems are deliberately modest. They verify finite implications and
counterexamples from encoded definitions. They do not certify the definitions as
adequate moral theory.
"""),
    code(r"""
theorems=pd.DataFrame([
 ['Capability dominance','Reflexive, transitive; antisymmetry on capability coordinates','Lean checked'],
 ['Phenotype viability','Policy closure implies finite-horizon viability','Lean checked'],
 ['Recovery','Recovery plus closure implies post-recovery viability','Lean checked'],
 ['Closure ≠ recovery','Finite counterexample theorem','Lean checked'],
 ['Robust dominance','Reflexive and transitive','Lean checked'],
 ['One environment ≠ robust','Finite counterexample theorem','Lean checked'],
 ['Focal benefit ≠ non-externalization','Affected-floor counterexample theorem','Lean checked'],
 ['Relative moral-goodness predicate','Projection lemmas to admissibility/plural/procedural fields','Lean checked, conditional'],
 ['Action influence ≠ reachability','Finite solver counterexample','Z3 checked'],
 ['Metric/pseudometric for alignment','Triangle, identity, symmetry as applicable','Not started'],
 ['Decorated map composition','Identity, composition, invariant preservation','Not started'],
],columns=['Object','Result / obligation','Status'])
display(theorems.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
## 8. Cross-project division of labour

The other repositories supply mechanisms and tasks; this repository supplies the
goodness calibration target and keeps descriptive coordination separate from moral
admissibility.
"""),
    code(r"""
projects=pd.DataFrame([
 ['topological-alignment','Phenotype goodness, virtue calibration, normative bridge','Integration layer'],
 ['shared-protention-alignment','Processability, channels, sheaf gluing','V4 shared semantics'],
 ['externalization-horizon','Observability and reciprocal readability','V4 legibility'],
 ['empathy-prisonner-dilemma','Joint outcomes and exploitability','First V4 environment'],
 ['Active_Inference_Social_Lock_In','Stable social lock-in and precision dynamics','Counterfeit-attractor control'],
 ['group-formation/tom','Shared scripts and post-shock group identity','V6 social dynamics'],
 ['aif-meta-cogames','Roles, strangers, ecological complementarity','Later generalization'],
],columns=['Project','Contribution','Use here'])
display(projects.style.hide(axis='index').set_properties(**{'text-align':'left'}))
"""),
    md(r"""
## 9. Experiment roadmap and stopping rules

| Stage | Primary claim | Decisive result | Stop / rethink if |
|---|---|---|---|
| V0 | Felt goodness is separable and fallible | temporal/causal identification beyond reward and approval | construct is not separable |
| V1 | Enablingness can ground calibration | counterfactual and phenotype reversals | target is intervention-fragile or tautological |
| V2/V3 | Virtue has selective-stability signature | recovery + retained diagnostic revision | simpler models explain it; no dissociation |
| **V4 next** | Shared meaning and cooperation are joint/relational | factorial dissociations + causal coupling | processability/readability add no value |
| V5 | Stable coordination can be morally counterfeit | matched exploitation/domination cases | moral bridge cannot discriminate non-ad-hocly |
| V6 | Public concepts and local realizations co-evolve | separable timescales + intervention response | static lexical model wins |
| V7 | Decorated topology adds value | held-out advantage on adversarial pairs | semantic/dynamical baselines tie or win |

## 10. The strongest honest conclusion today

<div class="proofbox"><b>Supported inside the declared simulation family:</b>
selective semantic stability occupies a broad sampled parameter region and depends,
under tested ablations, on slow-centre learning, evidence-precision gating,
context-sensitive meaning, and goodness-grounding.</div>

<div class="warningbox"><b>Not established:</b> that an explicit attraction term is
necessary; that EFE is uniquely necessary; that the model describes human virtue;
that phenotype-relative enablingness by itself is moral goodness; that shared
semantics or cooperation has already been integrated; or that topology adds
predictive value.</div>

The immediate task is therefore **V4**, not another attractive philosophical label:
one identified multi-agent model in which shared signs, translated expectations,
causal coupling, joint consequences, affected-party enablingness, and revision after
hidden harm are generated and tested together.

---

### Canonical supporting documents

- `docs/MULTISCALE_GOODNESS_AND_VIRTUE_ATTRACTORS.md`
- `docs/PARADIGM_PROOF_OBLIGATIONS.md`
- `docs/ACTIVE_INFERENCE_NORMATIVE_FORMALIZATION.md`
- `docs/MORAL_GOODNESS_ACTIVE_INFERENCE_TRANSLATION.md`
- `docs/SUPPORTED_REPLACEMENTS_LEDGER.md`
- `docs/V2_VIRTUE_ATTRACTOR_CALIBRATION_SIMULATION.md`
- `docs/CROSS_PROJECT_INTEGRATION_ASSESSMENT_2026-07-20.md`
- `formal/THEOREM_LEDGER.md`
"""),
])


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": "Where we are — claim-by-claim research guide",
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells and {len(CLAIMS)} canonical claims")
