"""Acceptance and adversarial checks for the V1 finite simulation."""

from __future__ import annotations

import json
from math import isclose
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from exp_v1_multiscale_enabling import (  # noqa: E402
    AUTONOMY_SENSITIVE,
    CALIBRATED,
    DEPENDENT,
    Process,
    bayes_sign_posterior,
    binary_variational_free_energy,
    enabling_contribution,
    evaluate_process,
    population_study,
)


def run() -> None:
    posterior = bayes_sign_posterior(0.5, True, 0.72, 1.0)
    posterior_vfe = binary_variational_free_energy(posterior, 0.5, True, 0.72, 1.0)
    assert posterior_vfe < binary_variational_free_energy(0.5, 0.5, True, 0.72, 1.0)
    assert posterior_vfe < binary_variational_free_energy(0.9, 0.5, True, 0.72, 1.0)

    neutral = Process(
        "neutral", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0
    )
    enabling, baseline, full = enabling_contribution(DEPENDENT, neutral)
    assert np.allclose(enabling, 0.0)
    assert np.allclose(baseline, full)

    # Institutional persistence is not part of constituent attainable metastability.
    extractive = Process(
        "extractive", -0.08, -0.10, -0.12, -0.14, -0.08, -0.15, 0.8, 0.85, 0.25
    )
    extractive_enabling, _, _ = enabling_contribution(DEPENDENT, extractive)
    assert extractive.institutional_persistence_delta > 0.0
    assert float(extractive_enabling.mean()) < 0.0

    # The same process can make a different counterfactual contribution to distinct
    # phenotypes; the phenotype index is not cosmetic.
    scaffold = Process(
        "centralized_scaffold", 0.03, 0.14, 0.18, 0.22, 0.05, -0.30, 0.55, 0.65, 0.20
    )
    dep = evaluate_process(DEPENDENT, scaffold, np.ones(4), CALIBRATED)
    auto = evaluate_process(AUTONOMY_SENSITIVE, scaffold, np.ones(4), CALIBRATED)
    assert dep.enabling_index > 0.0
    assert auto.enabling_index < 0.0

    result = population_study(seed=11000, scenarios=1200)
    assert result["all_gates_pass"], result["gates"]

    ledger = json.loads(
        (ROOT / "benchmarks" / "v1_multiscale_enabling_results.json").read_text("utf-8")
    )
    assert result["gates"] == {
        key: value for key, value in ledger["gates"].items() if key != "all_gates_pass"
    }
    for mode, statistics in ledger["summaries"].items():
        for statistic, expected in statistics.items():
            actual = result["summaries"][mode][statistic]
            assert isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    for phenotype, statistics in ledger["phenotype_reversal"].items():
        for statistic, expected in statistics.items():
            actual = result["phenotype_reversal"][phenotype][statistic]
            if isinstance(expected, list):
                assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)
            else:
                assert isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)


if __name__ == "__main__":
    run()
    print("PASS: V1 multi-scale enablingness and EFE calibration checks")
