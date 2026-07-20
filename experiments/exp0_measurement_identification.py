"""V0 measurement-identification simulation for the virtue programme.

This is a design-recovery study, not evidence about human moral cognition. It asks
whether a temporally ordered protocol could distinguish immediate felt goodness from
comfort, approval, an explicit consequence forecast, and downstream justification.

Four data-generating scenarios are included:

* ``nested``: feeling tracks perceived enabling consequences at several scales;
* ``proxy_only``: feeling tracks reward and approval but not enablingness;
* ``confounded``: enabling predictors covary with reward and approval;
* ``prompt_contaminated``: the feeling response is elicited after justification and
  is partly assimilated to it.

All confirmatory comparisons hold out entire scenario families.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


MODEL_COLUMNS = {
    "comfort_approval": ("reward", "approval"),
    "individual": ("reward", "approval", "perceived_individual"),
    "reported_forecast": (
        "reward",
        "approval",
        "perceived_individual",
        "reported_forecast",
    ),
    "nested_enabling": (
        "reward",
        "approval",
        "perceived_individual",
        "reported_forecast",
        "perceived_relational",
        "perceived_institutional",
        "perceived_ecological",
    ),
    # This deliberately invalid model quantifies post-treatment leakage.
    "justification_leakage": (
        "reward",
        "approval",
        "perceived_individual",
        "reported_forecast",
        "perceived_relational",
        "perceived_institutional",
        "perceived_ecological",
        "justification",
    ),
}


@dataclass(frozen=True)
class SimulationConfig:
    participants: int = 180
    context_families: int = 12
    trials_per_family: int = 2
    folds: int = 4
    noise_sd: float = 0.65


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    r2_comfort_approval: float
    r2_individual: float
    r2_reported_forecast: float
    r2_nested_enabling: float
    r2_justification_leakage: float
    nested_delta_r2: float
    revision_base_r2: float
    revision_evidence_r2: float
    revision_delta_r2: float
    design_condition_number: float
    feeling_justification_correlation: float
    nested_coefficient_signs: int


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale


def _ols_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    x_train_s, x_test_s = _standardize_train_test(x_train, x_test)
    train_design = np.column_stack([np.ones(len(x_train_s)), x_train_s])
    test_design = np.column_stack([np.ones(len(x_test_s)), x_test_s])
    coefficients, *_ = np.linalg.lstsq(train_design, y_train, rcond=None)
    return test_design @ coefficients


def grouped_cross_validated_r2(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
) -> float:
    """R-squared with whole context families held out."""
    unique_groups = np.unique(groups)
    fold_ids = np.arange(len(unique_groups)) % folds
    predictions = np.empty_like(y, dtype=float)
    for fold in range(folds):
        held_out = unique_groups[fold_ids == fold]
        test = np.isin(groups, held_out)
        train = ~test
        predictions[test] = _ols_predict(x[train], y[train], x[test])
    residual = float(np.sum((y - predictions) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - residual / max(total, 1e-12)


def _balanced_binary(rng: np.random.Generator, size: int) -> np.ndarray:
    values = np.tile(np.array([-1.0, 1.0]), int(np.ceil(size / 2)))[:size]
    return values[rng.permutation(size)]


def simulate_measurements(
    seed: int,
    scenario: str = "nested",
    config: SimulationConfig = SimulationConfig(),
) -> dict[str, np.ndarray]:
    if scenario not in {"nested", "proxy_only", "confounded", "prompt_contaminated"}:
        raise ValueError(f"unknown scenario: {scenario}")

    rng = np.random.default_rng(seed)
    trials_per_participant = config.context_families * config.trials_per_family
    n = config.participants * trials_per_participant
    participant = np.repeat(np.arange(config.participants), trials_per_participant)
    family_one = np.repeat(
        np.arange(config.context_families), config.trials_per_family
    )
    family = np.tile(family_one, config.participants)

    reward = _balanced_binary(rng, n)
    approval = _balanced_binary(rng, n)
    individual = _balanced_binary(rng, n)
    relational = _balanced_binary(rng, n)
    institutional = _balanced_binary(rng, n)
    ecological = _balanced_binary(rng, n)

    if scenario == "confounded":
        # This deliberately bad design makes scale variables nearly aliases of the
        # two familiar proxies. Small noise avoids an exactly singular matrix.
        individual = reward + rng.normal(0.0, 0.04, n)
        relational = approval + rng.normal(0.0, 0.04, n)
        institutional = reward + rng.normal(0.0, 0.04, n)
        ecological = approval + rng.normal(0.0, 0.04, n)

    calibration = np.clip(rng.normal(0.72, 0.14, config.participants), 0.2, 1.1)
    approval_capture = np.clip(rng.normal(0.16, 0.08, config.participants), 0.0, 0.4)
    person_intercept = rng.normal(0.0, 0.3, config.participants)

    def perceived(actual: np.ndarray, reliability: float) -> np.ndarray:
        return reliability * actual + rng.normal(0.0, 0.55, n)

    p_individual = perceived(individual, 0.86)
    p_relational = perceived(relational, 0.72)
    p_institutional = perceived(institutional, 0.60)
    p_ecological = perceived(ecological, 0.50)
    if scenario == "confounded":
        p_individual = reward + rng.normal(0.0, 0.025, n)
        p_relational = approval + rng.normal(0.0, 0.025, n)
        p_institutional = reward + rng.normal(0.0, 0.025, n)
        p_ecological = approval + rng.normal(0.0, 0.025, n)

    perceived_enabling = (
        0.35 * p_individual
        + 0.30 * p_relational
        + 0.20 * p_institutional
        + 0.15 * p_ecological
    )
    actual_enabling = (
        0.35 * individual
        + 0.30 * relational
        + 0.20 * institutional
        + 0.15 * ecological
    )

    # A verbal forecast is a noisy, focal-biased report rather than direct access to
    # the latent process. It therefore cannot be treated as identical to meaning.
    reported_forecast = (
        0.58 * p_individual
        + 0.18 * p_relational
        + 0.10 * p_institutional
        + 0.04 * p_ecological
        + rng.normal(0.0, 0.65, n)
    )

    enabling_weight = 0.0 if scenario == "proxy_only" else calibration[participant]
    felt_pre = (
        person_intercept[participant]
        + 0.22 * reward
        + approval_capture[participant] * approval
        + enabling_weight * perceived_enabling
        + rng.normal(0.0, config.noise_sd, n)
    )

    # Justification is elicited after feeling. It contains some genuine consequence
    # information but also rationalizes feeling and conforms to approval.
    justification = (
        0.44 * felt_pre
        + 0.28 * reported_forecast
        + 0.22 * approval
        + rng.normal(0.0, 0.65, n)
    )

    observed_felt_pre = felt_pre.copy()
    if scenario == "prompt_contaminated":
        observed_felt_pre = (
            0.55 * felt_pre + 0.45 * justification + rng.normal(0.0, 0.25, n)
        )

    # After a full consequence reveal, a calibrated agent should update toward the
    # actual multi-scale relation. Proxy-only agents retain the proxy process.
    post_enabling_weight = 0.0 if scenario == "proxy_only" else calibration[participant]
    felt_post = (
        person_intercept[participant]
        + 0.12 * reward
        + 0.05 * approval
        + post_enabling_weight * actual_enabling
        + rng.normal(0.0, config.noise_sd, n)
    )

    return {
        "participant": participant,
        "family": family,
        "reward": reward,
        "approval": approval,
        "actual_individual": individual,
        "actual_relational": relational,
        "actual_institutional": institutional,
        "actual_ecological": ecological,
        "perceived_individual": p_individual,
        "perceived_relational": p_relational,
        "perceived_institutional": p_institutional,
        "perceived_ecological": p_ecological,
        "reported_forecast": reported_forecast,
        "justification": justification,
        "felt_pre": observed_felt_pre,
        "felt_post": felt_post,
    }


def _matrix(data: dict[str, np.ndarray], columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([data[column] for column in columns])


def evaluate_scenario(
    seed: int,
    scenario: str,
    config: SimulationConfig = SimulationConfig(),
) -> ScenarioResult:
    data = simulate_measurements(seed, scenario, config)
    y = data["felt_pre"]
    groups = data["family"]
    model_r2 = {
        name: grouped_cross_validated_r2(
            _matrix(data, columns), y, groups, config.folds
        )
        for name, columns in MODEL_COLUMNS.items()
    }

    revision = data["felt_post"] - data["felt_pre"]
    revision_base_columns = ("felt_pre", "reward", "approval")
    revision_evidence_columns = revision_base_columns + (
        "actual_individual",
        "actual_relational",
        "actual_institutional",
        "actual_ecological",
        "perceived_individual",
        "perceived_relational",
        "perceived_institutional",
        "perceived_ecological",
    )
    revision_base_r2 = grouped_cross_validated_r2(
        _matrix(data, revision_base_columns), revision, groups, config.folds
    )
    revision_evidence_r2 = grouped_cross_validated_r2(
        _matrix(data, revision_evidence_columns), revision, groups, config.folds
    )

    nested_x = _matrix(data, MODEL_COLUMNS["nested_enabling"])
    nested_x_s, _ = _standardize_train_test(nested_x, nested_x)
    design_condition = float(np.linalg.cond(nested_x_s))
    design = np.column_stack([np.ones(len(nested_x_s)), nested_x_s])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    # Indices after the intercept: perceived individual=3, relational=5,
    # institutional=6, ecological=7 in MODEL_COLUMNS order.
    signs = int(np.sum(coefficients[[3, 5, 6, 7]] > 0.0))

    return ScenarioResult(
        scenario=scenario,
        r2_comfort_approval=model_r2["comfort_approval"],
        r2_individual=model_r2["individual"],
        r2_reported_forecast=model_r2["reported_forecast"],
        r2_nested_enabling=model_r2["nested_enabling"],
        r2_justification_leakage=model_r2["justification_leakage"],
        nested_delta_r2=(
            model_r2["nested_enabling"] - model_r2["reported_forecast"]
        ),
        revision_base_r2=revision_base_r2,
        revision_evidence_r2=revision_evidence_r2,
        revision_delta_r2=revision_evidence_r2 - revision_base_r2,
        design_condition_number=design_condition,
        feeling_justification_correlation=float(
            np.corrcoef(data["felt_pre"], data["justification"])[0, 1]
        ),
        nested_coefficient_signs=signs,
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run_recovery_study(
    replicates: int = 50,
    base_seed: int = 7000,
    config: SimulationConfig = SimulationConfig(),
) -> dict:
    scenarios = ("nested", "proxy_only", "confounded", "prompt_contaminated")
    raw: dict[str, list[ScenarioResult]] = {scenario: [] for scenario in scenarios}
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(replicates):
            seed = base_seed + scenario_index * 10000 + replicate
            raw[scenario].append(evaluate_scenario(seed, scenario, config))

    summaries = {}
    fields = (
        "r2_comfort_approval",
        "r2_individual",
        "r2_reported_forecast",
        "r2_nested_enabling",
        "r2_justification_leakage",
        "nested_delta_r2",
        "revision_base_r2",
        "revision_evidence_r2",
        "revision_delta_r2",
        "design_condition_number",
        "feeling_justification_correlation",
        "nested_coefficient_signs",
    )
    for scenario, results in raw.items():
        summaries[scenario] = {
            field: _summary([float(getattr(result, field)) for result in results])
            for field in fields
        }

    gates = {
        "nested_increment_detected": (
            summaries["nested"]["nested_delta_r2"]["p05"] > 0.03
        ),
        "evidence_revision_detected": (
            summaries["nested"]["revision_delta_r2"]["p05"] > 0.08
        ),
        "proxy_false_positive_controlled": (
            summaries["proxy_only"]["nested_delta_r2"]["p95"] < 0.01
        ),
        "confounding_flagged": (
            summaries["confounded"]["design_condition_number"]["p05"] > 20.0
        ),
        "prompt_contamination_detected": (
            summaries["prompt_contaminated"]["feeling_justification_correlation"]["p05"]
            > summaries["nested"]["feeling_justification_correlation"]["p95"]
        ),
        "nested_signs_recovered": (
            summaries["nested"]["nested_coefficient_signs"]["p05"] >= 4.0
        ),
    }

    return {
        "schema_version": "1.0",
        "study": "V0 measurement-identification simulation",
        "epistemic_status": "design recovery only; not human evidence",
        "config": asdict(config),
        "replicates": replicates,
        "base_seed": base_seed,
        "summaries": summaries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_recovery_study(args.replicates, args.base_seed)
    encoded = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not results["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
