"""Acceptance checks for virtue-attractor selective stability simulation."""

from __future__ import annotations

import json
from math import isclose
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from exp_v2_virtue_attractor import (  # noqa: E402
    ContextCase,
    FEATURE_NAMES,
    REGIMES,
    SemanticState,
    semantic_features,
    update_semantics,
    variational_free_energy,
    run_experiment,
)
from exp_v1_multiscale_enabling import DEPENDENT, Process  # noqa: E402


def run() -> None:
    process = Process("check", 0.05, 0.04, 0.03, 0.02, 0.01, 0.05, 0.5, 0.5, 0.0)
    features = semantic_features(ContextCase(process, DEPENDENT))
    assert len(features) == len(FEATURE_NAMES)

    config = REGIMES["calibrated"]
    state = SemanticState(np.zeros(len(features)), np.zeros(len(features)))
    before = variational_free_energy(state, features, 0.1, 1.0, config)
    update_semantics(state, features, 0.1, 1.0, 1.0, config)
    after = variational_free_energy(state, features, 0.1, 1.0, config)
    assert after < before

    result = run_experiment(seed=13000)
    assert result["all_gates_pass"], result["gates"]
    ledger = json.loads(
        (ROOT / "benchmarks" / "v2_virtue_attractor_results.json").read_text("utf-8")
    )
    assert result["gates"] == {
        key: value for key, value in ledger["gates"].items() if key != "all_gates_pass"
    }
    for regime, statistics in ledger["primary_results"].items():
        for statistic, expected in statistics.items():
            actual = result["results"][regime][statistic]
            assert isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)


if __name__ == "__main__":
    run()
    print("PASS: V2/V3 virtue-attractor selective-stability checks")
