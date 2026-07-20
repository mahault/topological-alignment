"""V1 finite multi-scale enablingness and active-inference simulation.

Actual enablingness is computed from counterfactual phenotype dynamics. Felt goodness
is represented by a policy-conditioned expected-free-energy contrast under an agent's
beliefs. The simulation tests calibration; it does not define moral goodness as EFE or
claim that the chosen finite dynamics describe human cognition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import exp, log
from pathlib import Path

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class Phenotype:
    name: str
    sustain: float
    recover: float
    options: float
    direct_sensitivity: float
    recovery_sensitivity: float
    relational_dependency: float
    institutional_dependency: float
    ecological_dependency: float
    option_sensitivity: float


@dataclass(frozen=True)
class Process:
    name: str
    direct: float
    recovery: float
    relational: float
    institutional: float
    ecological: float
    options: float
    reward_probability: float
    approval_probability: float
    institutional_persistence_delta: float


@dataclass(frozen=True)
class AgentMode:
    name: str
    reward_weight: float
    approval_weight: float
    enabling_weight: float
    evidence_precision: float
    approval_prior_strength: float
    policy_precision: float = 4.0
    epistemic_weight: float = 0.05


@dataclass(frozen=True)
class ScenarioEvaluation:
    enabling_vector: tuple[float, float, float, float]
    enabling_index: float
    baseline_vector: tuple[float, float, float, float]
    full_vector: tuple[float, float, float, float]
    pre_proxy_signal: float
    pre_focal_signal: float
    pre_nested_signal: float
    post_nested_signal: float
    accept_probability: float
    posterior_pre: tuple[float, float, float, float]
    posterior_post: tuple[float, float, float, float]


DEPENDENT = Phenotype(
    "dependency_intensive",
    sustain=0.66,
    recover=0.38,
    options=0.52,
    direct_sensitivity=1.0,
    recovery_sensitivity=1.1,
    relational_dependency=1.25,
    institutional_dependency=1.15,
    ecological_dependency=1.0,
    option_sensitivity=0.75,
)

AUTONOMY_SENSITIVE = Phenotype(
    "autonomy_sensitive",
    sustain=0.78,
    recover=0.56,
    options=0.76,
    direct_sensitivity=0.75,
    recovery_sensitivity=0.75,
    relational_dependency=0.35,
    institutional_dependency=-0.90,
    ecological_dependency=0.65,
    option_sensitivity=1.20,
)

CALIBRATED = AgentMode(
    "calibrated",
    reward_weight=0.15,
    approval_weight=0.10,
    enabling_weight=1.0,
    evidence_precision=1.0,
    approval_prior_strength=0.15,
)

CAPTURED = AgentMode(
    "socially_captured",
    reward_weight=0.55,
    approval_weight=1.0,
    enabling_weight=0.35,
    evidence_precision=0.25,
    approval_prior_strength=2.5,
)


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 0.02, 0.98))


def transition_matrix(sustain: float, recover: float) -> np.ndarray:
    """Rows are impaired/viable; columns are impaired/viable."""
    return np.array(
        [[1.0 - recover, recover], [1.0 - sustain, sustain]], dtype=float
    )


def viable_occupancy(matrix: np.ndarray, horizon: int, initial_viable: float) -> float:
    distribution = np.array([1.0 - initial_viable, initial_viable], dtype=float)
    total = 0.0
    for _ in range(horizon):
        distribution = distribution @ matrix
        total += distribution[1]
    return float(total / horizon)


def attainable_metastability(
    phenotype: Phenotype, process: Process | None
) -> np.ndarray:
    """Return short viability, recovery, long viability, and retained options."""
    if process is None:
        short_delta = recovery_delta = long_delta = option_delta = 0.0
    else:
        short_delta = process.direct * phenotype.direct_sensitivity
        recovery_delta = process.recovery * phenotype.recovery_sensitivity
        long_delta = (
            short_delta
            + 0.45 * process.relational * phenotype.relational_dependency
            + 0.45 * process.institutional * phenotype.institutional_dependency
            + 0.35 * process.ecological * phenotype.ecological_dependency
        )
        option_delta = process.options * phenotype.option_sensitivity

    short_matrix = transition_matrix(
        _clip_probability(phenotype.sustain + short_delta),
        _clip_probability(phenotype.recover + 0.4 * recovery_delta),
    )
    long_matrix = transition_matrix(
        _clip_probability(phenotype.sustain + long_delta),
        _clip_probability(phenotype.recover + recovery_delta + 0.4 * long_delta),
    )
    return np.array(
        [
            viable_occupancy(short_matrix, horizon=3, initial_viable=0.7),
            viable_occupancy(long_matrix, horizon=4, initial_viable=0.0),
            viable_occupancy(long_matrix, horizon=12, initial_viable=0.7),
            _clip_probability(phenotype.options + option_delta),
        ]
    )


def enabling_contribution(phenotype: Phenotype, process: Process) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = attainable_metastability(phenotype, None)
    full = attainable_metastability(phenotype, process)
    return full - baseline, baseline, full


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return log(p / (1.0 - p))


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    e = exp(value)
    return e / (1.0 + e)


def bayes_sign_posterior(
    prior_good: float,
    cue_positive: bool,
    cue_accuracy: float,
    evidence_precision: float,
) -> float:
    accuracy = _clip_probability(cue_accuracy)
    log_likelihood_ratio = log(accuracy / (1.0 - accuracy))
    signed_evidence = log_likelihood_ratio if cue_positive else -log_likelihood_ratio
    return _logistic(_logit(prior_good) + evidence_precision * signed_evidence)


def binary_variational_free_energy(
    posterior_good: float,
    prior_good: float,
    cue_positive: bool,
    cue_accuracy: float,
    evidence_precision: float = 1.0,
) -> float:
    """VFE for a binary sign posterior under a possibly tempered likelihood."""
    q = _clip_probability(posterior_good)
    p = _clip_probability(prior_good)
    accuracy = _clip_probability(cue_accuracy)
    likelihood_good = accuracy if cue_positive else 1.0 - accuracy
    likelihood_harmful = 1.0 - likelihood_good
    complexity = q * log(q / p) + (1.0 - q) * log((1.0 - q) / (1.0 - p))
    accuracy_term = -evidence_precision * (
        q * log(likelihood_good) + (1.0 - q) * log(likelihood_harmful)
    )
    return complexity + accuracy_term


def infer_enabling_vector(
    actual_enabling: np.ndarray,
    cue_signs: np.ndarray,
    cue_accuracy: float,
    approval_probability: float,
    mode: AgentMode,
    previous_posterior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    social_logit = mode.approval_prior_strength * (2.0 * approval_probability - 1.0)
    if previous_posterior is None:
        priors = np.full(4, _logistic(social_logit))
    else:
        priors = previous_posterior
    posterior = np.array(
        [
            bayes_sign_posterior(
                float(priors[index]),
                bool(cue_signs[index] > 0),
                cue_accuracy,
                mode.evidence_precision,
            )
            for index in range(4)
        ]
    )
    expected = (2.0 * posterior - 1.0) * np.maximum(np.abs(actual_enabling), 0.01)
    return expected, posterior


def bernoulli_pragmatic_cost(probability: float, preferred_high: float = 0.9) -> float:
    q = _clip_probability(probability)
    preferred = _clip_probability(preferred_high)
    return -(q * log(preferred) + (1.0 - q) * log(1.0 - preferred))


def binary_entropy(probability: float) -> float:
    p = _clip_probability(probability)
    return -(p * log(p) + (1.0 - p) * log(1.0 - p))


def expected_information_gain(prior_good: float, accuracy: float) -> float:
    p = _clip_probability(prior_good)
    a = _clip_probability(accuracy)
    positive_probability = p * a + (1.0 - p) * (1.0 - a)
    expected_conditional_entropy = p * binary_entropy(a) + (1.0 - p) * binary_entropy(1.0 - a)
    return max(binary_entropy(positive_probability) - expected_conditional_entropy, 0.0)


def efe_contrast(
    baseline: np.ndarray,
    expected_enabling: np.ndarray,
    reward_probability: float,
    approval_probability: float,
    posterior: np.ndarray,
    model: str,
    mode: AgentMode,
    reveal_accuracy: float = 0.97,
) -> float:
    """Return G(reject)-G(accept); positive values favor accepting the process."""
    if model not in {"proxy", "focal", "nested"}:
        raise ValueError(f"unknown model: {model}")
    accept_cost = (
        mode.reward_weight * bernoulli_pragmatic_cost(reward_probability)
        + mode.approval_weight * bernoulli_pragmatic_cost(approval_probability)
    )
    reject_cost = (
        mode.reward_weight * bernoulli_pragmatic_cost(0.5)
        + mode.approval_weight * bernoulli_pragmatic_cost(0.5)
    )
    represented = 0
    if model == "focal":
        represented = 1
    elif model == "nested":
        represented = 4
    if represented:
        for index in range(represented):
            accept_cost += mode.enabling_weight * bernoulli_pragmatic_cost(
                baseline[index] + expected_enabling[index]
            )
            reject_cost += mode.enabling_weight * bernoulli_pragmatic_cost(
                baseline[index]
            )
        information = sum(
            expected_information_gain(float(posterior[index]), reveal_accuracy)
            for index in range(represented)
        )
        accept_cost -= mode.epistemic_weight * information
    return float(reject_cost - accept_cost)


def evaluate_process(
    phenotype: Phenotype,
    process: Process,
    cue_signs: np.ndarray,
    mode: AgentMode,
    initial_cue_accuracy: float = 0.72,
    reveal_accuracy: float = 0.97,
) -> ScenarioEvaluation:
    enabling, baseline, full = enabling_contribution(phenotype, process)
    expected_pre, posterior_pre = infer_enabling_vector(
        enabling,
        cue_signs,
        initial_cue_accuracy,
        process.approval_probability,
        mode,
    )
    true_signs = np.where(enabling >= 0.0, 1.0, -1.0)
    expected_post, posterior_post = infer_enabling_vector(
        enabling,
        true_signs,
        reveal_accuracy,
        process.approval_probability,
        mode,
        previous_posterior=posterior_pre,
    )
    proxy = efe_contrast(
        baseline,
        expected_pre,
        process.reward_probability,
        process.approval_probability,
        posterior_pre,
        "proxy",
        mode,
    )
    focal = efe_contrast(
        baseline,
        expected_pre,
        process.reward_probability,
        process.approval_probability,
        posterior_pre,
        "focal",
        mode,
    )
    nested = efe_contrast(
        baseline,
        expected_pre,
        process.reward_probability,
        process.approval_probability,
        posterior_pre,
        "nested",
        mode,
    )
    post = efe_contrast(
        baseline,
        expected_post,
        process.reward_probability,
        process.approval_probability,
        posterior_post,
        "nested",
        mode,
    )
    return ScenarioEvaluation(
        enabling_vector=tuple(float(value) for value in enabling),
        enabling_index=float(enabling.mean()),
        baseline_vector=tuple(float(value) for value in baseline),
        full_vector=tuple(float(value) for value in full),
        pre_proxy_signal=proxy,
        pre_focal_signal=focal,
        pre_nested_signal=nested,
        post_nested_signal=post,
        accept_probability=_logistic(mode.policy_precision * post),
        posterior_pre=tuple(float(value) for value in posterior_pre),
        posterior_post=tuple(float(value) for value in posterior_post),
    )


def random_process(rng: np.random.Generator, index: int) -> Process:
    institutional_persistence = rng.uniform(-0.18, 0.24)
    # Approval partly follows institutional persistence, creating realistic capture
    # cases while leaving independent variance.
    approval = _logistic(2.8 * institutional_persistence + rng.normal(0.0, 0.9))
    return Process(
        name=f"process_{index:04d}",
        direct=rng.uniform(-0.18, 0.18),
        recovery=rng.uniform(-0.18, 0.18),
        relational=rng.uniform(-0.22, 0.22),
        institutional=rng.uniform(-0.22, 0.22),
        ecological=rng.uniform(-0.22, 0.22),
        options=rng.uniform(-0.22, 0.22),
        reward_probability=rng.uniform(0.15, 0.85),
        approval_probability=approval,
        institutional_persistence_delta=institutional_persistence,
    )


def _correlation(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def population_study(seed: int = 11000, scenarios: int = 1200) -> dict:
    rng = np.random.default_rng(seed)
    records = {"calibrated": [], "captured": []}
    process_records = []
    phenotypes = (DEPENDENT, AUTONOMY_SENSITIVE)
    for index in range(scenarios):
        process = random_process(rng, index)
        phenotype = phenotypes[index % len(phenotypes)]
        enabling, _, _ = enabling_contribution(phenotype, process)
        true_signs = np.where(enabling >= 0.0, 1, -1)
        correct = rng.random(4) < 0.72
        cue_signs = np.where(correct, true_signs, -true_signs)
        calibrated = evaluate_process(phenotype, process, cue_signs, CALIBRATED)
        captured = evaluate_process(phenotype, process, cue_signs, CAPTURED)
        records["calibrated"].append(calibrated)
        records["captured"].append(captured)
        process_records.append(process)

    summaries = {}
    for mode_name, evaluations in records.items():
        target = [entry.enabling_index for entry in evaluations]
        proxy = [entry.pre_proxy_signal for entry in evaluations]
        focal = [entry.pre_focal_signal for entry in evaluations]
        nested = [entry.pre_nested_signal for entry in evaluations]
        post = [entry.post_nested_signal for entry in evaluations]
        robust = [
            index
            for index, entry in enumerate(evaluations)
            if min(entry.enabling_vector) > 0.005 or max(entry.enabling_vector) < -0.005
        ]
        correct_policies = []
        for index in robust:
            should_accept = evaluations[index].enabling_index > 0.0
            predicted_accept = evaluations[index].accept_probability > 0.5
            correct_policies.append(should_accept == predicted_accept)
        traps = [
            index
            for index, (entry, process) in enumerate(zip(evaluations, process_records))
            if process.institutional_persistence_delta > 0.08
            and entry.enabling_index < -0.01
        ]
        trap_acceptance = (
            float(np.mean([evaluations[index].accept_probability > 0.5 for index in traps]))
            if traps
            else float("nan")
        )
        summaries[mode_name] = {
            "corr_proxy_actual": _correlation(proxy, target),
            "corr_focal_actual": _correlation(focal, target),
            "corr_nested_pre_actual": _correlation(nested, target),
            "corr_nested_post_actual": _correlation(post, target),
            "robust_case_count": len(robust),
            "robust_policy_accuracy": float(np.mean(correct_policies)),
            "institution_trap_count": len(traps),
            "institution_trap_acceptance": trap_acceptance,
        }

    central_scaffold = Process(
        name="centralized_scaffold",
        direct=0.03,
        recovery=0.14,
        relational=0.18,
        institutional=0.22,
        ecological=0.05,
        options=-0.30,
        reward_probability=0.55,
        approval_probability=0.65,
        institutional_persistence_delta=0.20,
    )
    positive_cues = np.ones(4)
    paired = {
        phenotype.name: asdict(
            evaluate_process(phenotype, central_scaffold, positive_cues, CALIBRATED)
        )
        for phenotype in phenotypes
    }

    gates = {
        "bayesian_update_minimizes_vfe": False,
        "nested_beats_proxy": (
            summaries["calibrated"]["corr_nested_pre_actual"]
            > summaries["calibrated"]["corr_proxy_actual"] + 0.25
        ),
        "nested_beats_focal": (
            summaries["calibrated"]["corr_nested_pre_actual"]
            > summaries["calibrated"]["corr_focal_actual"] + 0.10
        ),
        "reveal_improves_calibration": (
            summaries["calibrated"]["corr_nested_post_actual"]
            > summaries["calibrated"]["corr_nested_pre_actual"] + 0.08
        ),
        "capture_degrades_calibration": (
            summaries["calibrated"]["corr_nested_post_actual"]
            > summaries["captured"]["corr_nested_post_actual"] + 0.20
        ),
        "capture_worsens_institution_traps": (
            summaries["captured"]["institution_trap_acceptance"]
            > summaries["calibrated"]["institution_trap_acceptance"] + 0.10
        ),
        "phenotype_reversal": (
            paired[DEPENDENT.name]["enabling_index"] > 0.0
            and paired[AUTONOMY_SENSITIVE.name]["enabling_index"] < 0.0
        ),
    }
    representative_posterior = bayes_sign_posterior(0.5, True, 0.72, 1.0)
    posterior_vfe = binary_variational_free_energy(
        representative_posterior, 0.5, True, 0.72, 1.0
    )
    grid_minimum = min(
        binary_variational_free_energy(float(q), 0.5, True, 0.72, 1.0)
        for q in np.linspace(0.001, 0.999, 999)
    )
    vfe_optimality_gap = posterior_vfe - grid_minimum
    gates["bayesian_update_minimizes_vfe"] = vfe_optimality_gap <= 1e-10
    return {
        "schema_version": "1.0",
        "study": "V1 multi-scale enablingness and EFE calibration simulation",
        "epistemic_status": "finite model evidence only; no human or moral validation",
        "seed": seed,
        "scenarios": scenarios,
        "summaries": summaries,
        "vfe_check": {
            "posterior_good": representative_posterior,
            "posterior_vfe": posterior_vfe,
            "grid_minimum_vfe": grid_minimum,
            "optimality_gap": vfe_optimality_gap,
        },
        "phenotype_reversal": paired,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--scenarios", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = population_study(args.seed, args.scenarios)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
