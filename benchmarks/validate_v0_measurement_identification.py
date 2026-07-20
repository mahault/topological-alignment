"""Deterministic acceptance checks for the V0 design-recovery simulation."""

from __future__ import annotations

import json
from math import isclose
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from exp0_measurement_identification import (  # noqa: E402
    SimulationConfig,
    run_recovery_study,
)


def run() -> None:
    # Kept smaller than the archived 50-replicate analysis so the static validation
    # suite remains quick while exercising every adversarial scenario.
    result = run_recovery_study(
        replicates=12,
        base_seed=9100,
        config=SimulationConfig(participants=120),
    )
    assert result["all_gates_pass"], result["gates"]
    assert all(result["gates"].values())

    # Reproduce the archived primary summaries exactly from its declared run.
    ledger = json.loads(
        (ROOT / "benchmarks" / "v0_measurement_identification_results.json").read_text(
            "utf-8"
        )
    )
    archived = run_recovery_study(
        replicates=ledger["run"]["replicates"],
        base_seed=ledger["run"]["base_seed"],
        config=SimulationConfig(
            participants=ledger["run"]["participants"],
            context_families=ledger["run"]["context_families"],
            trials_per_family=ledger["run"]["trials_per_family"],
            folds=ledger["run"]["folds"],
        ),
    )
    assert archived["gates"] == {
        key: value for key, value in ledger["gates"].items() if key != "all_gates_pass"
    }
    for scenario, metrics in ledger["primary_results"].items():
        for metric, statistics in metrics.items():
            for statistic, expected in statistics.items():
                actual = archived["summaries"][scenario][metric][statistic]
                assert isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (
                    scenario,
                    metric,
                    statistic,
                    actual,
                    expected,
                )


if __name__ == "__main__":
    run()
    print("PASS: V0 measurement-identification and adversarial design checks")
