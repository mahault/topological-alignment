"""Validate uncertainty-aware policy evaluation in the finite counterexample."""

from __future__ import annotations

from math import isclose

from uncertain_standing_model import run_uncertain_counterexample


def close(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate() -> list[str]:
    result = run_uncertain_counterexample()
    hypotheses = result["normative_hypotheses"]
    violation = result["violation_probabilities"]
    unconstrained = result["unconstrained_policy_posterior"]
    weighted = result["uncertainty_weighted_policy_posterior"]
    precautionary = result["precautionary_policy_posterior"]
    failures: list[str] = []

    if not close(sum(hypothesis.probability for hypothesis in hypotheses), 1.0):
        failures.append("normative hypothesis distribution is not normalized")
    if not (close(violation[0], 0.0) and close(violation[1], 0.42)):
        failures.append(f"unexpected violation probabilities: {violation}")
    if not all(close(value, 0.5) for value in unconstrained):
        failures.append(f"unconstrained posterior should remain indifferent: {unconstrained}")
    if not (close(weighted[0], 1.0 / 1.58) and close(weighted[1], 0.58 / 1.58)):
        failures.append(f"unexpected uncertainty-weighted posterior: {weighted}")
    if not (close(precautionary[0], 1.0) and close(precautionary[1], 0.0)):
        failures.append(f"precautionary posterior should exclude externalization: {precautionary}")
    return failures


if __name__ == "__main__":
    errors = validate()
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    result = run_uncertain_counterexample()
    print(f"violation_probabilities={result['violation_probabilities']}")
    print(f"unconstrained={result['unconstrained_policy_posterior']}")
    print(f"uncertainty_weighted={result['uncertainty_weighted_policy_posterior']}")
    print(f"precautionary={result['precautionary_policy_posterior']}")
    print("PASS: standing and capability-floor uncertainty model")
