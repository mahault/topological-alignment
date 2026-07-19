"""Validate the structure and adversarial logic of the moral-proxy benchmark."""

from __future__ import annotations

import json
from pathlib import Path


BENCHMARK = Path(__file__).with_name("moral_proxy_pairs.json")
REQUIRED_PAIR_KEYS = {
    "id",
    "category",
    "proxy",
    "case_a",
    "case_b",
    "decisive_decorations",
    "audit_question",
}


def validate() -> list[str]:
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    errors: list[str] = []
    pairs = data.get("pairs", [])
    ids: set[str] = set()

    if not pairs:
        return ["benchmark contains no pairs"]

    for index, pair in enumerate(pairs):
        location = f"pairs[{index}]"
        missing = REQUIRED_PAIR_KEYS - pair.keys()
        if missing:
            errors.append(f"{location}: missing keys {sorted(missing)}")
            continue

        pair_id = pair["id"]
        if pair_id in ids:
            errors.append(f"{location}: duplicate id {pair_id!r}")
        ids.add(pair_id)

        proxy = pair["proxy"]
        if not {"name", "a", "b"} <= proxy.keys():
            errors.append(f"{pair_id}: proxy requires name, a, and b")
        elif proxy["a"] != proxy["b"]:
            errors.append(f"{pair_id}: adversarial proxy values are not matched")

        for case_key in ("case_a", "case_b"):
            case = pair[case_key]
            if not {"label", "description", "morally_admissible"} <= case.keys():
                errors.append(f"{pair_id}: {case_key} is incomplete")

        if pair["case_a"].get("morally_admissible") == pair["case_b"].get(
            "morally_admissible"
        ):
            errors.append(f"{pair_id}: cases do not have opposed moral labels")

        decorations = pair["decisive_decorations"]
        if not isinstance(decorations, list) or len(decorations) < 2:
            errors.append(f"{pair_id}: at least two decisive decorations required")

        if not str(pair["audit_question"]).endswith("?"):
            errors.append(f"{pair_id}: audit question must be explicit")

    return errors


if __name__ == "__main__":
    failures = validate()
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    count = len(json.loads(BENCHMARK.read_text(encoding="utf-8"))["pairs"])
    print(f"PASS: {count} matched-proxy, opposed-moral-status pairs")
