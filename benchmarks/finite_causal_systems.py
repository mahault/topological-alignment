"""Executable finite systems for adversarial moral-proxy translation tests.

Each pair fixes a tempting proxy exactly while varying a relational or semantic
decoration required by the project's adopted moral interpretation. These systems test
discriminability; they do not prove that the moral labels are correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable


@dataclass(frozen=True)
class FiniteCase:
    name: str
    states: tuple[str, ...]
    decorations: dict[str, Any]
    expected_admissible: bool


@dataclass(frozen=True)
class FinitePair:
    pair_id: str
    category: str
    proxy_name: str
    case_a: FiniteCase
    case_b: FiniteCase
    proxy_evaluator: Callable[[FiniteCase], Fraction]
    decorated_evaluator: Callable[[FiniteCase], bool]


def all_true(case: FiniteCase, *keys: str) -> bool:
    return all(case.decorations[key] is True for key in keys)


ROBUSTNESS = FinitePair(
    pair_id="robustness_average_tail",
    category="robustness",
    proxy_name="mean_outcome",
    case_a=FiniteCase(
        "bounded_variation",
        ("environment_low", "environment_high"),
        {"environment_outcomes": (4, 6), "floor": 2},
        True,
    ),
    case_b=FiniteCase(
        "catastrophic_tail",
        ("environment_catastrophe", "environment_windfall"),
        {"environment_outcomes": (0, 10), "floor": 2},
        False,
    ),
    proxy_evaluator=lambda case: Fraction(
        sum(case.decorations["environment_outcomes"]),
        len(case.decorations["environment_outcomes"]),
    ),
    decorated_evaluator=lambda case: min(case.decorations["environment_outcomes"])
    >= case.decorations["floor"],
)


CAPABILITY = FinitePair(
    pair_id="capability_reachable_meaningful",
    category="capability floors",
    proxy_name="reachable_state_count",
    case_a=FiniteCase(
        "substantive_options",
        tuple(f"option_{i}" for i in range(8)),
        {"reachable": frozenset(range(8)), "safe": frozenset(range(8)), "floor": 4},
        True,
    ),
    case_b=FiniteCase(
        "nominal_options",
        tuple(f"option_{i}" for i in range(8)),
        {"reachable": frozenset(range(8)), "safe": frozenset({0}), "floor": 4},
        False,
    ),
    proxy_evaluator=lambda case: Fraction(len(case.decorations["reachable"]), 1),
    decorated_evaluator=lambda case: len(
        case.decorations["reachable"] & case.decorations["safe"]
    )
    >= case.decorations["floor"],
)


EXTERNALIZATION = FinitePair(
    pair_id="externalization_system_boundary",
    category="non-externalization",
    proxy_name="focal_pragmatic_risk",
    case_a=FiniteCase(
        "shared_efficiency",
        ("focal", "affected"),
        {"focal_risk": Fraction(1, 2), "affected_cost": 0, "affected_floor": 1},
        True,
    ),
    case_b=FiniteCase(
        "hidden_burden",
        ("focal", "affected"),
        {"focal_risk": Fraction(1, 2), "affected_cost": 1, "affected_floor": 1},
        False,
    ),
    proxy_evaluator=lambda case: case.decorations["focal_risk"],
    decorated_evaluator=lambda case: case.decorations["affected_cost"]
    < case.decorations["affected_floor"],
)


DOMINATION = FinitePair(
    pair_id="domination_care_control",
    category="non-domination",
    proxy_name="interventional_total_variation",
    case_a=FiniteCase(
        "authorized_care",
        ("protected", "exposed"),
        {
            "outcome_under_control": (0, 1),
            "authorized": True,
            "reviewable": True,
            "refusable": True,
            "retaliatory": False,
        },
        True,
    ),
    case_b=FiniteCase(
        "uncontrolled_benevolence",
        ("protected", "exposed"),
        {
            "outcome_under_control": (0, 1),
            "authorized": False,
            "reviewable": False,
            "refusable": False,
            "retaliatory": True,
        },
        False,
    ),
    # Total variation between two deterministic interventional outcomes.
    proxy_evaluator=lambda case: Fraction(
        int(case.decorations["outcome_under_control"][0]
            != case.decorations["outcome_under_control"][1]),
        1,
    ),
    decorated_evaluator=lambda case: all_true(
        case, "authorized", "reviewable", "refusable"
    )
    and not case.decorations["retaliatory"],
)


CONTESTABILITY = FinitePair(
    pair_id="contestability_feedback",
    category="contestability",
    proxy_name="challenge_message_count",
    case_a=FiniteCase(
        "effective_challenge",
        ("unchanged", "review", "revised"),
        {"messages": ("object", "appeal"), "reaches_review": True, "can_revise": True},
        True,
    ),
    case_b=FiniteCase(
        "performative_feedback",
        ("unchanged", "review", "revised"),
        {"messages": ("object", "appeal"), "reaches_review": False, "can_revise": False},
        False,
    ),
    proxy_evaluator=lambda case: Fraction(len(case.decorations["messages"]), 1),
    decorated_evaluator=lambda case: all_true(case, "reaches_review", "can_revise"),
)


REPAIR = FinitePair(
    pair_id="repair_controller_reset",
    category="repairability",
    proxy_name="controller_recovery_steps",
    case_a=FiniteCase(
        "affected_party_restoration",
        ("harm", "stabilize", "compensate", "restored"),
        {"affected_restored": True, "compensated": True, "source_revised": True},
        True,
    ),
    case_b=FiniteCase(
        "controller_reset",
        ("harm", "stabilize", "reset", "controller_ready"),
        {"affected_restored": False, "compensated": False, "source_revised": False},
        False,
    ),
    proxy_evaluator=lambda case: Fraction(len(case.states) - 1, 1),
    decorated_evaluator=lambda case: all_true(
        case, "affected_restored", "compensated", "source_revised"
    ),
)


FINITE_PAIRS: tuple[FinitePair, ...] = (
    ROBUSTNESS,
    CAPABILITY,
    EXTERNALIZATION,
    DOMINATION,
    CONTESTABILITY,
    REPAIR,
)


def evaluate_pair(pair: FinitePair) -> dict[str, Any]:
    proxy_a = pair.proxy_evaluator(pair.case_a)
    proxy_b = pair.proxy_evaluator(pair.case_b)
    return {
        "pair_id": pair.pair_id,
        "category": pair.category,
        "proxy_name": pair.proxy_name,
        "proxy_matched": proxy_a == proxy_b,
        "proxy_a": str(proxy_a),
        "proxy_b": str(proxy_b),
        "decorated_a": pair.decorated_evaluator(pair.case_a),
        "decorated_b": pair.decorated_evaluator(pair.case_b),
        "expected_a": pair.case_a.expected_admissible,
        "expected_b": pair.case_b.expected_admissible,
    }
