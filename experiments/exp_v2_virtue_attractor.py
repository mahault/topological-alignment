"""V2/V3 simulation: virtue as an attractor of meaning relative to goodness.

The semantic state is a learned map from context features to expected enablingness.
Fast beliefs minimize a Gaussian variational-free-energy objective around a slow
character centre. The centre itself changes under accumulated diagnostic evidence.
Policy and the felt-goodness analogue arise from an EFE contrast.

Constructed regimes are compared for selective stability:

* calibrated: precision-sensitive recovery plus slow diagnostic revision;
* dogmatic: strong return with negligible revision;
* unstable: weak attraction and indiscriminate learning;
* opportunistic: reward/approval are treated as the learning target.

These names describe programmed parameter regimes. The simulation tests whether the
proposed diagnostics distinguish them; it does not discover virtues in human data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp_v1_multiscale_enabling import (  # noqa: E402
    AUTONOMY_SENSITIVE,
    DEPENDENT,
    Phenotype,
    Process,
    bernoulli_pragmatic_cost,
    enabling_contribution,
    random_process,
)


FEATURE_NAMES = (
    "reward",
    "approval",
    "direct_support",
    "recovery_support",
    "relational_support",
    "institutional_support",
    "ecological_support",
    "option_support",
    "autonomy_context",
    "direct_x_context",
    "recovery_x_context",
    "relational_x_context",
    "institutional_x_context",
    "ecological_x_context",
    "options_x_context",
    "intercept",
)


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    fast_learning_rate: float
    attractor_precision: float
    slow_learning_rate: float
    minimum_evidence_precision: float
    diagnostic_gain: float
    reward_preference_weight: float
    approval_preference_weight: float
    enabling_preference_weight: float
    subjective_reward_weight: float = 0.0
    subjective_approval_weight: float = 0.0
    grounding_weight: float = 1.0
    feature_ablation: str = "none"
    use_efe_policy: bool = True


@dataclass
class SemanticState:
    weights: np.ndarray
    centre: np.ndarray


@dataclass(frozen=True)
class ContextCase:
    process: Process
    phenotype: Phenotype

    @property
    def autonomy_context(self) -> float:
        return float(self.phenotype.name == AUTONOMY_SENSITIVE.name)


REGIMES = {
    "calibrated": RegimeConfig(
        "calibrated", 0.035, 0.10, 0.025, 0.0, 1.0, 0.12, 0.08, 1.0
    ),
    "dogmatic": RegimeConfig(
        "dogmatic", 0.018, 0.32, 0.001, 0.0, 0.15, 0.12, 0.08, 1.0
    ),
    "unstable": RegimeConfig(
        "unstable", 0.095, 0.006, 0.075, 0.65, 1.0, 0.12, 0.08, 1.0
    ),
    "opportunistic": RegimeConfig(
        "opportunistic",
        0.050,
        0.08,
        0.030,
        0.0,
        0.8,
        0.65,
        0.95,
        0.20,
        subjective_reward_weight=0.65,
        subjective_approval_weight=0.75,
    ),
}


def semantic_features(case: ContextCase) -> np.ndarray:
    """Fixed, interpretable feature map; no moral label enters the representation."""
    process = case.process
    context = case.autonomy_context
    supports = np.array(
        [
            process.direct / 0.18,
            process.recovery / 0.18,
            process.relational / 0.22,
            process.institutional / 0.22,
            process.ecological / 0.22,
            process.options / 0.22,
        ]
    )
    return np.array(
        [
            2.0 * process.reward_probability - 1.0,
            2.0 * process.approval_probability - 1.0,
            *supports,
            context,
            *(context * supports),
            1.0,
        ],
        dtype=float,
    )


def regime_features(case: ContextCase, config: RegimeConfig) -> np.ndarray:
    features = semantic_features(case)
    if config.feature_ablation == "none":
        return features
    if config.feature_ablation == "no_context_interactions":
        features = features.copy()
        features[9:15] = 0.0
        return features
    raise ValueError(f"unknown feature ablation: {config.feature_ablation}")


def actual_enabling(case: ContextCase) -> tuple[np.ndarray, float]:
    vector, _, _ = enabling_contribution(case.phenotype, case.process)
    return vector, float(vector.mean())


def subjective_target(config: RegimeConfig, process: Process, actual: float) -> float:
    return (
        config.grounding_weight * actual
        + config.subjective_reward_weight * (2.0 * process.reward_probability - 1.0)
        + config.subjective_approval_weight * (2.0 * process.approval_probability - 1.0)
    )


def variational_free_energy(
    state: SemanticState,
    features: np.ndarray,
    observed_target: float,
    evidence_precision: float,
    config: RegimeConfig,
) -> float:
    prediction_error = observed_target - float(state.weights @ features)
    complexity = 0.5 * config.attractor_precision * float(
        np.sum((state.weights - state.centre) ** 2)
    )
    inaccuracy = 0.5 * evidence_precision * prediction_error**2
    return complexity + inaccuracy


def update_semantics(
    state: SemanticState,
    features: np.ndarray,
    observed_target: float,
    evidence_precision: float,
    diagnosticity: float,
    config: RegimeConfig,
) -> None:
    """One gradient step on VFE plus slower precision-gated centre learning."""
    effective_precision = max(evidence_precision, config.minimum_evidence_precision)
    prediction = float(state.weights @ features)
    gradient = (
        effective_precision * (prediction - observed_target) * features
        + config.attractor_precision * (state.weights - state.centre)
    )
    state.weights -= config.fast_learning_rate * np.clip(gradient, -5.0, 5.0)
    centre_gain = (
        config.slow_learning_rate
        * config.diagnostic_gain
        * diagnosticity
        * effective_precision
    )
    state.centre += centre_gain * (state.weights - state.centre)


def efe_felt_signal(
    state: SemanticState, case: ContextCase, config: RegimeConfig
) -> float:
    process = case.process
    predicted_enabling = float(state.weights @ regime_features(case, config))
    if not config.use_efe_policy:
        return predicted_enabling
    enabling_probability = 1.0 / (1.0 + np.exp(-5.0 * np.clip(predicted_enabling, -4, 4)))
    accept_cost = (
        config.reward_preference_weight
        * bernoulli_pragmatic_cost(process.reward_probability)
        + config.approval_preference_weight
        * bernoulli_pragmatic_cost(process.approval_probability)
        + config.enabling_preference_weight
        * bernoulli_pragmatic_cost(enabling_probability)
    )
    reject_cost = (
        config.reward_preference_weight * bernoulli_pragmatic_cost(0.5)
        + config.approval_preference_weight * bernoulli_pragmatic_cost(0.5)
        + config.enabling_preference_weight * bernoulli_pragmatic_cost(0.5)
    )
    return float(reject_cost - accept_cost)


def _correlation(a: list[float], b: list[float]) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def evaluate_contexts(
    state: SemanticState, cases: list[ContextCase], config: RegimeConfig
) -> dict:
    actual = []
    signals = []
    policies = []
    for case in cases:
        _, target = actual_enabling(case)
        signal = efe_felt_signal(state, case, config)
        actual.append(target)
        signals.append(signal)
        policies.append(signal > 0.0)
    sign_consistent = [abs(value) > 0.01 for value in actual]
    accuracy = np.mean(
        [policies[i] == (actual[i] > 0.0) for i in range(len(actual)) if sign_consistent[i]]
    )
    return {
        "calibration": _correlation(signals, actual),
        "policy_accuracy": float(accuracy),
        "signals": np.asarray(signals),
        "actual": np.asarray(actual),
    }


def _make_process_sets(seed: int) -> dict[str, list[ContextCase]]:
    rng = np.random.default_rng(seed)
    baseline_train = [ContextCase(random_process(rng, i), DEPENDENT) for i in range(900)]
    baseline_test = [ContextCase(random_process(rng, 1000 + i), DEPENDENT) for i in range(300)]
    trap_pool: list[ContextCase] = []
    counter = 2000
    while len(trap_pool) < 500:
        candidate = random_process(rng, counter)
        counter += 1
        case = ContextCase(candidate, AUTONOMY_SENSITIVE)
        _, target = actual_enabling(case)
        if (
            candidate.approval_probability > 0.58
            and candidate.institutional_persistence_delta > 0.06
            and target < -0.012
        ):
            trap_pool.append(case)
    return {
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "diagnostic_train": trap_pool[:300],
        "diagnostic_test": trap_pool[300:],
    }


def _train_phase(
    state: SemanticState,
    cases: list[ContextCase],
    config: RegimeConfig,
    rng: np.random.Generator,
    evidence_precision: float,
    diagnosticity: float,
    observation_noise: float,
) -> None:
    for case in cases:
        process = case.process
        _, actual = actual_enabling(case)
        target = subjective_target(config, process, actual)
        observed = target + rng.normal(0.0, observation_noise)
        update_semantics(
            state,
            regime_features(case, config),
            observed,
            evidence_precision,
            diagnosticity,
            config,
        )


def run_regime(
    config: RegimeConfig,
    process_sets: dict[str, list[ContextCase]],
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    state = SemanticState(weights=np.zeros(len(FEATURE_NAMES)), centre=np.zeros(len(FEATURE_NAMES)))

    _train_phase(
        state,
        process_sets["baseline_train"],
        config,
        rng,
        evidence_precision=1.0,
        diagnosticity=0.25,
        observation_noise=0.035,
    )
    settled_weights = state.weights.copy()
    settled_centre = state.centre.copy()
    baseline = evaluate_contexts(state, process_sets["baseline_test"], config)
    trap_before = evaluate_contexts(state, process_sets["diagnostic_test"], config)

    # Low-precision, non-diagnostic noise should perturb fast meaning without
    # rewriting the slow centre.
    noise_processes = process_sets["baseline_train"][:120]
    _train_phase(
        state,
        noise_processes,
        config,
        rng,
        evidence_precision=0.04,
        diagnosticity=0.0,
        observation_noise=0.80,
    )
    noise_displacement = float(np.linalg.norm(state.weights - settled_weights))
    centre_noise_displacement = float(np.linalg.norm(state.centre - settled_centre))

    # Return to ordinary reliable evidence.
    _train_phase(
        state,
        process_sets["baseline_train"][:180],
        config,
        rng,
        evidence_precision=1.0,
        diagnosticity=0.15,
        observation_noise=0.035,
    )
    recovery_distance = float(np.linalg.norm(state.weights - settled_weights))
    recovery_ratio = recovery_distance / max(noise_displacement, 1e-12)
    pre_diagnostic_weights = state.weights.copy()
    pre_diagnostic_centre = state.centre.copy()
    pre_signals = evaluate_contexts(state, process_sets["diagnostic_test"], config)["signals"]

    # Repeated high-precision evidence that approved institution-preserving processes
    # damage constituent attainable metastability.
    _train_phase(
        state,
        process_sets["diagnostic_train"],
        config,
        rng,
        evidence_precision=1.5,
        diagnosticity=1.0,
        observation_noise=0.02,
    )
    trap_after = evaluate_contexts(state, process_sets["diagnostic_test"], config)
    post_signals = trap_after["signals"]
    reversals = int(np.sum((pre_signals > 0.0) != (post_signals > 0.0)))
    post_diagnostic_weights = state.weights.copy()
    post_diagnostic_centre = state.centre.copy()

    # Test whether the transformed meaning survives intervening ordinary contexts.
    # Fast local accommodation without slow revision should wash out here.
    _train_phase(
        state,
        process_sets["baseline_train"][300:600],
        config,
        rng,
        evidence_precision=1.0,
        diagnosticity=0.05,
        observation_noise=0.035,
    )
    trap_retained = evaluate_contexts(state, process_sets["diagnostic_test"], config)
    retained_signals = trap_retained["signals"]
    retained_reversals = int(
        np.sum((pre_signals > 0.0) != (retained_signals > 0.0))
    )

    return {
        "baseline_calibration": baseline["calibration"],
        "baseline_policy_accuracy": baseline["policy_accuracy"],
        "trap_calibration_before": trap_before["calibration"],
        "trap_calibration_after": trap_after["calibration"],
        "trap_policy_accuracy_before": trap_before["policy_accuracy"],
        "trap_policy_accuracy_after": trap_after["policy_accuracy"],
        "trap_calibration_retained": trap_retained["calibration"],
        "trap_policy_accuracy_retained": trap_retained["policy_accuracy"],
        "noise_displacement": noise_displacement,
        "centre_noise_displacement": centre_noise_displacement,
        "recovery_distance": recovery_distance,
        "recovery_ratio": recovery_ratio,
        "diagnostic_weight_change": float(np.linalg.norm(state.weights - pre_diagnostic_weights)),
        "diagnostic_centre_change": float(np.linalg.norm(state.centre - pre_diagnostic_centre)),
        "approval_weight_before": float(pre_diagnostic_weights[1]),
        "approval_weight_after": float(state.weights[1]),
        "policy_reversals": reversals,
        "retained_policy_reversals": retained_reversals,
        "post_diagnostic_washout": float(np.linalg.norm(state.weights - post_diagnostic_weights)),
        "centre_retention": float(np.linalg.norm(state.centre - post_diagnostic_centre)),
        "settled_weights": settled_weights.tolist(),
        "final_weights": state.weights.tolist(),
        "final_centre": state.centre.tolist(),
    }


def run_experiment(seed: int = 13000) -> dict:
    process_sets = _make_process_sets(seed)
    results = {
        name: run_regime(config, process_sets, seed + 100 * index)
        for index, (name, config) in enumerate(REGIMES.items())
    }
    calibrated = results["calibrated"]
    dogmatic = results["dogmatic"]
    unstable = results["unstable"]
    opportunistic = results["opportunistic"]
    gates = {
        "calibrated_baseline_meaning": calibrated["baseline_calibration"] > 0.75,
        "selective_noise_stability": (
            calibrated["noise_displacement"] < unstable["noise_displacement"] * 0.55
            and calibrated["centre_noise_displacement"] < 1e-10
        ),
        "calibrated_recovery": (
            calibrated["recovery_distance"] < 0.02
            and calibrated["recovery_distance"] < unstable["recovery_distance"] * 0.60
        ),
        "diagnostic_transformation": (
            calibrated["trap_calibration_after"]
            > calibrated["trap_calibration_before"] + 0.20
            and calibrated["trap_calibration_retained"]
            > calibrated["trap_calibration_before"] + 0.15
            and calibrated["diagnostic_centre_change"]
            > dogmatic["diagnostic_centre_change"] * 5.0
        ),
        "dogmatism_distinguished": (
            dogmatic["recovery_ratio"] < 0.9
            and calibrated["trap_calibration_retained"]
            > dogmatic["trap_calibration_retained"] + 0.12
        ),
        "opportunism_distinguished": (
            calibrated["trap_policy_accuracy_after"]
            > opportunistic["trap_policy_accuracy_after"] + 0.20
        ),
        "meaning_changes_policy": (
            calibrated["trap_policy_accuracy_retained"]
            > calibrated["trap_policy_accuracy_before"] + 0.08
        ),
    }
    return {
        "schema_version": "1.0",
        "study": "V2/V3 virtue attractor over meaning relative to goodness",
        "epistemic_status": "constructed dynamical simulation; no human or moral validation",
        "seed": seed,
        "feature_names": FEATURE_NAMES,
        "regime_parameters": {name: asdict(config) for name, config in REGIMES.items()},
        "results": results,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.seed)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
