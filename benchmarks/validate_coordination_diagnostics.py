"""Adversarial checks for the supported coordination diagnostics."""

from __future__ import annotations

import numpy as np

from coordination_diagnostics import (
    coefficient_of_determination,
    directed_processability,
    interventional_js_bits,
    matched_coupling_contrasts,
    reciprocal_readability,
    total_correlation_bits,
    whole_minus_parts_interaction_bits,
)


def run() -> None:
    # Perfect deterministic coordination need not have statistical dependence.
    deterministic = {(0, 0): 1.0}
    assert np.isclose(total_correlation_bits(deterministic), 0.0)

    # Redundant common randomness has positive dependence without establishing
    # cooperation or complementary contribution.
    redundant_common = {(0, 0): 0.5, (1, 1): 0.5}
    assert np.isclose(total_correlation_bits(redundant_common), 1.0)

    # A stable relabeling is mutually processable even when raw categories disagree.
    source = [[0.95, 0.05], [0.05, 0.95]] * 4
    relabeled_target = [[0.05, 0.95], [0.95, 0.05]] * 4
    processability = directed_processability(source, relabeled_target)
    assert processability["reliable"] is True
    assert float(processability["score"]) > 0.99

    # A constant target contains no temporal structure to reconstruct.
    constant_target = [[0.5, 0.5]] * 8
    vacuous = directed_processability(source, constant_target)
    assert vacuous["reliable"] is False

    # The signed whole-minus-parts contrast separates XOR complementarity from a
    # redundant copy, while remaining explicitly weaker than a full PID.
    xor_joint = {
        (0, 0, 0): 0.25,
        (0, 1, 1): 0.25,
        (1, 0, 1): 0.25,
        (1, 1, 0): 0.25,
    }
    redundant_joint = {(0, 0, 0): 0.5, (1, 1, 1): 0.5}
    assert np.isclose(whole_minus_parts_interaction_bits(xor_joint), 1.0)
    assert np.isclose(whole_minus_parts_interaction_bits(redundant_joint), -1.0)

    # Correlation would ignore scale error; coefficient of determination does not.
    actual = [0.0, 1.0, 2.0, 3.0]
    scaled = [0.0, 2.0, 4.0, 6.0]
    assert np.isclose(np.corrcoef(actual, scaled)[0, 1], 1.0)
    assert coefficient_of_determination(actual, scaled) < 0.0

    # Mutual readability is bottlenecked by the worse observer direction.
    reciprocal = reciprocal_readability(
        actual,
        actual,
        actual,
        [3.0, 2.0, 1.0, 0.0],
    )
    assert reciprocal < 0.0

    # An effect diagnostic is zero for identical interventional distributions and
    # positive when interventions change the outcome distribution.
    assert np.isclose(interventional_js_bits([0.8, 0.2], [0.8, 0.2]), 0.0)
    assert interventional_js_bits([0.9, 0.1], [0.1, 0.9]) > 0.5

    # Coupling can benefit one agent while harming another. Keeping the vector blocks
    # a scalar aggregate from hiding this conflict.
    contrasts = matched_coupling_contrasts(
        coupled_efe=[3.0, 1.0], decoupled_efe=[4.0, 0.5]
    )
    assert np.allclose(contrasts, [1.0, -0.5])


if __name__ == "__main__":
    run()
    print("PASS: coordination diagnostics and adversarial non-equivalences")
