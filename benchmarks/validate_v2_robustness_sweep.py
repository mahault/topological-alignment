"""Deterministic validation for the V2/V3 robustness sweep."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from exp_v2_robustness_sweep import run_sweep  # noqa: E402


def run() -> None:
    result = run_sweep(
        sweep_seed=17000,
        configuration_count=12,
        environment_seeds=(17100, 17200),
    )
    assert result["all_gates_pass"]
    assert result["robust_configuration_count"] > 0
    assert result["ablations"]["full_model"]["seed_pass_count"] == 2
    assert not result["ablations"]["no_slow_centre_learning"]["robust_pass"]
    assert not result["ablations"]["no_precision_gating"]["robust_pass"]
    assert not result["ablations"]["no_context_interactions"]["robust_pass"]
    assert not result["ablations"]["no_goodness_grounding"]["robust_pass"]
    # These passing negative results are part of the claim boundary: this test
    # does not identify either explicit term as necessary.
    assert result["ablations"]["no_attractor"]["robust_pass"]
    assert result["ablations"]["no_efe_policy_mapping"]["robust_pass"]


if __name__ == "__main__":
    run()
    print("PASS: V2/V3 robustness sweep and ablation checks")
