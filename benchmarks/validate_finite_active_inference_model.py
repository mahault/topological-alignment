"""Validate the finite probabilistic active-inference counterexample."""

from __future__ import annotations

from math import isclose

from finite_active_inference_model import run_counterexample


def close(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate() -> list[str]:
    result = run_counterexample()
    posterior = result["posterior_after_reward"]
    evaluations = result["evaluations"]
    unconstrained = result["unconstrained_policy_posterior"]
    constrained = result["constrained_policy_posterior"]
    failures: list[str] = []

    if not all(close(a, b) for a, b in zip(posterior, (0.8181818181818181, 0.18181818181818182))):
        failures.append(f"unexpected Bayesian posterior: {posterior}")

    cooperate, externalize = evaluations
    if not close(cooperate.expected_free_energy, externalize.expected_free_energy):
        failures.append("focal EFE is not matched")
    if cooperate.predicted_states != externalize.predicted_states:
        failures.append("focal predicted states are not matched")
    if cooperate.predicted_observations != externalize.predicted_observations:
        failures.append("focal predicted observations are not matched")
    if not cooperate.admissible or externalize.admissible:
        failures.append("capability-floor classification is incorrect")
    if not all(close(value, 0.5) for value in unconstrained):
        failures.append(f"unconstrained posterior should be indifferent: {unconstrained}")
    if not (close(constrained[0], 1.0) and close(constrained[1], 0.0)):
        failures.append(f"constrained posterior should exclude externalization: {constrained}")

    return failures


if __name__ == "__main__":
    errors = validate()
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    result = run_counterexample()
    cooperate, externalize = result["evaluations"]
    print(f"posterior_after_reward={result['posterior_after_reward']}")
    print(f"matched_focal_EFE={cooperate.expected_free_energy:.12f}")
    print(f"risk={cooperate.risk:.12f}")
    print(f"ambiguity={cooperate.ambiguity:.12f}")
    print(f"pragmatic_cost={cooperate.pragmatic_cost:.12f}")
    print(f"information_gain={cooperate.information_gain:.12f}")
    print(f"affected_viability=({cooperate.affected_viability}, {externalize.affected_viability})")
    print(f"unconstrained={result['unconstrained_policy_posterior']}")
    print(f"constrained={result['constrained_policy_posterior']}")
    print("PASS: explicit finite active-inference externalization counterexample")
