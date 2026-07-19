"""Exhaustive validation for the executable finite moral-proxy systems."""

from __future__ import annotations

import json

from finite_causal_systems import FINITE_PAIRS, evaluate_pair


def validate() -> tuple[list[dict[str, object]], list[str]]:
    reports = [evaluate_pair(pair) for pair in FINITE_PAIRS]
    failures: list[str] = []

    ids = [report["pair_id"] for report in reports]
    if len(ids) != len(set(ids)):
        failures.append("pair identifiers are not unique")

    for report in reports:
        pair_id = str(report["pair_id"])
        if not report["proxy_matched"]:
            failures.append(f"{pair_id}: base proxy is not exactly matched")
        if report["expected_a"] == report["expected_b"]:
            failures.append(f"{pair_id}: expected labels are not opposed")
        if report["decorated_a"] != report["expected_a"]:
            failures.append(f"{pair_id}: decorated evaluator misclassifies case A")
        if report["decorated_b"] != report["expected_b"]:
            failures.append(f"{pair_id}: decorated evaluator misclassifies case B")

    return reports, failures


if __name__ == "__main__":
    results, errors = validate()
    print(json.dumps(results, indent=2, sort_keys=True))
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"PASS: {len(results)} finite matched-proxy systems")
