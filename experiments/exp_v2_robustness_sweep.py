"""Multi-seed robustness sweep and mechanism ablations for V2/V3.

This script evaluates numeric parameter configurations without using their regime
names. A configuration passes only when it combines baseline calibration, noise
stability, recovery, retained diagnostic transformation, policy improvement, and
slow-centre revision across held-out contexts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp_v2_virtue_attractor import (  # noqa: E402
    REGIMES,
    RegimeConfig,
    _make_process_sets,
    run_regime,
)


GATE_NAMES = (
    "baseline_calibration",
    "noise_stability",
    "recovery",
    "retained_transformation",
    "retained_policy_improvement",
    "slow_centre_revision",
)


def signature_gates(result: dict) -> dict[str, bool]:
    return {
        "baseline_calibration": result["baseline_calibration"] > 0.75,
        "noise_stability": result["noise_displacement"] < 0.08,
        "recovery": result["recovery_distance"] < 0.04,
        "retained_transformation": (
            result["trap_calibration_retained"]
            > result["trap_calibration_before"] + 0.12
        ),
        "retained_policy_improvement": (
            result["trap_policy_accuracy_retained"]
            > result["trap_policy_accuracy_before"] + 0.05
        ),
        "slow_centre_revision": result["diagnostic_centre_change"] > 0.01,
    }


def sample_configurations(seed: int, count: int) -> list[RegimeConfig]:
    rng = np.random.default_rng(seed)
    configurations = []
    for index in range(count):
        configurations.append(
            RegimeConfig(
                name=f"candidate_{index:03d}",
                fast_learning_rate=float(rng.uniform(0.015, 0.065)),
                attractor_precision=float(np.exp(rng.uniform(np.log(0.02), np.log(0.30)))),
                slow_learning_rate=float(np.exp(rng.uniform(np.log(0.004), np.log(0.07)))),
                minimum_evidence_precision=float(rng.uniform(0.0, 0.20)),
                diagnostic_gain=float(rng.uniform(0.40, 1.30)),
                reward_preference_weight=0.12,
                approval_preference_weight=0.08,
                enabling_preference_weight=1.0,
            )
        )
    return configurations


def _parameter_summary(configurations: list[RegimeConfig]) -> dict:
    parameters = (
        "fast_learning_rate",
        "attractor_precision",
        "slow_learning_rate",
        "minimum_evidence_precision",
        "diagnostic_gain",
    )
    if not configurations:
        return {parameter: None for parameter in parameters}
    return {
        parameter: {
            "min": float(min(getattr(config, parameter) for config in configurations)),
            "median": float(
                np.median([getattr(config, parameter) for config in configurations])
            ),
            "max": float(max(getattr(config, parameter) for config in configurations)),
        }
        for parameter in parameters
    }


def evaluate_configuration(
    config: RegimeConfig,
    process_sets_by_seed: dict[int, dict],
) -> dict:
    seed_results = []
    for index, (seed, process_sets) in enumerate(process_sets_by_seed.items()):
        metrics = run_regime(config, process_sets, seed + 700 + index)
        gates = signature_gates(metrics)
        seed_results.append(
            {
                "seed": seed,
                "gates": gates,
                "all_gates_pass": all(gates.values()),
                "metrics": {
                    key: metrics[key]
                    for key in (
                        "baseline_calibration",
                        "noise_displacement",
                        "recovery_distance",
                        "trap_calibration_before",
                        "trap_calibration_retained",
                        "trap_policy_accuracy_before",
                        "trap_policy_accuracy_retained",
                        "diagnostic_centre_change",
                    )
                },
            }
        )
    pass_count = sum(result["all_gates_pass"] for result in seed_results)
    return {
        "config": asdict(config),
        "seed_results": seed_results,
        "seed_pass_count": pass_count,
        "robust_pass": pass_count == len(seed_results),
    }


def ablation_configurations() -> dict[str, RegimeConfig]:
    base = REGIMES["calibrated"]
    return {
        "full_model": base,
        "no_slow_centre_learning": replace(base, name="no_slow", slow_learning_rate=0.0),
        "no_precision_gating": replace(
            base, name="no_precision", minimum_evidence_precision=1.0
        ),
        "no_attractor": replace(base, name="no_attractor", attractor_precision=0.0),
        "no_context_interactions": replace(
            base, name="no_context", feature_ablation="no_context_interactions"
        ),
        "no_goodness_grounding": replace(
            base,
            name="no_grounding",
            grounding_weight=0.0,
            subjective_reward_weight=0.65,
            subjective_approval_weight=0.75,
        ),
        "no_efe_policy_mapping": replace(base, name="no_efe", use_efe_policy=False),
    }


def run_sweep(
    sweep_seed: int = 16000,
    configuration_count: int = 48,
    environment_seeds: tuple[int, ...] = (16100, 16200, 16300),
) -> dict:
    process_sets = {seed: _make_process_sets(seed) for seed in environment_seeds}
    configurations = sample_configurations(sweep_seed, configuration_count)
    evaluations = [
        evaluate_configuration(config, process_sets) for config in configurations
    ]
    robust_configs = [
        config
        for config, evaluation in zip(configurations, evaluations)
        if evaluation["robust_pass"]
    ]
    per_gate_rates = {
        gate: float(
            np.mean(
                [
                    seed_result["gates"][gate]
                    for evaluation in evaluations
                    for seed_result in evaluation["seed_results"]
                ]
            )
        )
        for gate in GATE_NAMES
    }

    ablations = {
        name: evaluate_configuration(config, process_sets)
        for name, config in ablation_configurations().items()
    }
    ablation_summary = {
        name: {
            "seed_pass_count": evaluation["seed_pass_count"],
            "robust_pass": evaluation["robust_pass"],
            "gate_pass_rates": {
                gate: float(
                    np.mean(
                        [result["gates"][gate] for result in evaluation["seed_results"]]
                    )
                )
                for gate in GATE_NAMES
            },
        }
        for name, evaluation in ablations.items()
    }

    gates = {
        "nonzero_robust_region": len(robust_configs) >= 3,
        # This guards against a vacuous sweep without imposing an arbitrary
        # upper bound on how broad a genuine robust region is allowed to be.
        "not_everything_passes": len(robust_configs) < configuration_count,
        "full_model_replicates": ablation_summary["full_model"]["robust_pass"],
        "slow_centre_is_necessary": not ablation_summary["no_slow_centre_learning"][
            "robust_pass"
        ],
        "precision_gating_is_necessary": not ablation_summary["no_precision_gating"][
            "robust_pass"
        ],
        "context_interactions_are_necessary": not ablation_summary[
            "no_context_interactions"
        ]["robust_pass"],
        "goodness_grounding_is_necessary": not ablation_summary["no_goodness_grounding"][
            "robust_pass"
        ],
    }

    return {
        "schema_version": "1.0",
        "study": "V2/V3 robustness sweep and mechanism ablations",
        "epistemic_status": "constructed parameter sweep; no human or moral validation",
        "sweep_seed": sweep_seed,
        "configuration_count": configuration_count,
        "environment_seeds": environment_seeds,
        "robust_configuration_count": len(robust_configs),
        "robust_configuration_fraction": len(robust_configs) / configuration_count,
        "robust_parameter_region": _parameter_summary(robust_configs),
        "per_gate_pass_rates": per_gate_rates,
        "ablations": ablation_summary,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "configuration_evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-seed", type=int, default=16000)
    parser.add_argument("--configurations", type=int, default=48)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_sweep(args.sweep_seed, args.configurations)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    # Console output omits the large per-configuration block.
    concise = {key: value for key, value in result.items() if key != "configuration_evaluations"}
    print(json.dumps(concise, indent=2, sort_keys=True))
    if not result["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
