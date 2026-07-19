"""Finite diagnostics for semantic compatibility and joint coordination.

These quantities are deliberately not combined into a scalar definition of
cooperation. Each measures a different property and carries its own failure modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log2

import numpy as np


EPS = 1e-12


def _normalized(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(array < 0):
        raise ValueError("probabilities must be non-negative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return array / total


def kl_bits(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in bits for finite categorical distributions."""

    p_array = _normalized(p)
    q_array = _normalized(q)
    mask = p_array > 0
    if np.any(q_array[mask] <= 0):
        return float("inf")
    return float(np.sum(p_array[mask] * np.log2(p_array[mask] / q_array[mask])))


def entropy_bits(probabilities: Sequence[float]) -> float:
    p = _normalized(probabilities)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def _validate_joint(joint: Mapping[tuple[int, ...], float]) -> int:
    if not joint:
        raise ValueError("joint distribution cannot be empty")
    width = len(next(iter(joint)))
    if width == 0 or any(len(key) != width for key in joint):
        raise ValueError("joint keys must be equal-length non-empty tuples")
    if any(value < 0 for value in joint.values()):
        raise ValueError("joint probabilities must be non-negative")
    if sum(joint.values()) <= 0:
        raise ValueError("joint distribution must have positive mass")
    return width


def marginal(
    joint: Mapping[tuple[int, ...], float], positions: Sequence[int]
) -> dict[tuple[int, ...], float]:
    """Marginalize a finite joint distribution onto selected positions."""

    width = _validate_joint(joint)
    if not positions or any(position < 0 or position >= width for position in positions):
        raise ValueError("positions must be valid and non-empty")
    total = float(sum(joint.values()))
    result: dict[tuple[int, ...], float] = {}
    for state, mass in joint.items():
        key = tuple(state[position] for position in positions)
        result[key] = result.get(key, 0.0) + mass / total
    return result


def mutual_information_bits(
    joint: Mapping[tuple[int, ...], float],
    x_positions: Sequence[int],
    y_positions: Sequence[int],
) -> float:
    """Mutual information between two subvectors of a finite joint distribution."""

    xy_positions = tuple(x_positions) + tuple(y_positions)
    p_x = marginal(joint, x_positions)
    p_y = marginal(joint, y_positions)
    p_xy = marginal(joint, xy_positions)
    information = 0.0
    x_width = len(x_positions)
    for xy_state, mass in p_xy.items():
        if mass <= 0:
            continue
        x_state = xy_state[:x_width]
        y_state = xy_state[x_width:]
        information += mass * log2(mass / (p_x[x_state] * p_y[y_state]))
    return information


def total_correlation_bits(joint: Mapping[tuple[int, ...], float]) -> float:
    """Dependence baseline: sum marginal entropies minus joint entropy."""

    width = _validate_joint(joint)
    joint_entropy = entropy_bits(list(joint.values()))
    marginal_entropies = sum(
        entropy_bits(list(marginal(joint, [position]).values()))
        for position in range(width)
    )
    return marginal_entropies - joint_entropy


def whole_minus_parts_interaction_bits(
    joint_aby: Mapping[tuple[int, int, int], float]
) -> float:
    """Signed interaction contrast I(A,B;Y)-I(A;Y)-I(B;Y).

    This separates canonical XOR synergy from redundant copying, but is not a full
    partial-information decomposition and must not be reported as one.
    """

    if _validate_joint(joint_aby) != 3:
        raise ValueError("expected a three-variable joint distribution (A, B, Y)")
    return (
        mutual_information_bits(joint_aby, [0, 1], [2])
        - mutual_information_bits(joint_aby, [0], [2])
        - mutual_information_bits(joint_aby, [1], [2])
    )


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def temporal_information_bits(protention: Sequence[Sequence[float]]) -> float:
    """Mean KL from each horizon step to the horizon-average prediction."""

    stream = np.asarray(protention, dtype=float)
    if stream.ndim != 2 or stream.shape[0] < 2:
        raise ValueError("protention must be a two-dimensional horizon stream")
    stream = np.vstack([_normalized(row) for row in stream])
    average = stream.mean(axis=0)
    return float(np.mean([kl_bits(row, average) for row in stream]))


def fit_alignment_channel(
    source: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    iterations: int = 3000,
    learning_rate: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Fit one row-stochastic channel across an entire protention horizon.

    Adapted from ``shared-protention-alignment/core/morphism.py``. The objective is
    mean_t KL(target_t || source_t T). A single channel prevents per-step relabeling
    from manufacturing compatibility.
    """

    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("source and target must be two-dimensional")
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("source and target horizons must match")
    source_array = np.vstack([_normalized(row) for row in source_array])
    target_array = np.vstack([_normalized(row) for row in target_array])
    horizon, source_width = source_array.shape
    target_width = target_array.shape[1]
    logits = np.zeros((source_width, target_width))
    for _ in range(iterations):
        channel = _softmax_rows(logits)
        mapped = np.clip(source_array @ channel, EPS, None)
        channel_gradient = -(source_array.T @ (target_array / mapped)) / horizon
        logits_gradient = channel * (
            channel_gradient
            - (channel_gradient * channel).sum(axis=1, keepdims=True)
        )
        logits -= learning_rate * logits_gradient
    channel = _softmax_rows(logits)
    mapped = np.clip(source_array @ channel, EPS, None)
    residual = float(
        np.mean([kl_bits(target_row, mapped_row) for target_row, mapped_row in zip(target_array, mapped)])
    )
    return channel, residual


def directed_processability(
    source: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    minimum_information_bits: float = 0.05,
) -> dict[str, float | bool | np.ndarray]:
    """Held-out frame alignment normalized by target temporal information."""

    channel, residual = fit_alignment_channel(source, target)
    target_information = temporal_information_bits(target)
    reliable = target_information >= minimum_information_bits
    score = (
        max(0.0, 1.0 - residual / target_information)
        if target_information > EPS
        else float("nan")
    )
    return {
        "score": score,
        "residual_bits": residual,
        "target_information_bits": target_information,
        "reliable": reliable,
        "channel": channel,
    }


def coefficient_of_determination(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    """Readability score that penalizes bias and scale mismatch."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape or actual_array.size < 2:
        raise ValueError("actual and predicted must have the same non-trivial shape")
    denominator = float(np.sum((actual_array - actual_array.mean()) ** 2))
    if denominator <= EPS:
        raise ValueError("actual sequence must have non-zero variance")
    residual = float(np.sum((actual_array - predicted_array) ** 2))
    return 1.0 - residual / denominator


def reciprocal_readability(
    a_actual: Sequence[float],
    a_inferred_by_b: Sequence[float],
    b_actual: Sequence[float],
    b_inferred_by_a: Sequence[float],
) -> float:
    """Mutual readability is limited by the weaker directed observer model."""

    return min(
        coefficient_of_determination(a_actual, a_inferred_by_b),
        coefficient_of_determination(b_actual, b_inferred_by_a),
    )


def interventional_js_bits(
    outcome_under_do_0: Sequence[float], outcome_under_do_1: Sequence[float]
) -> float:
    """Jensen-Shannon separation between two supplied intervention distributions.

    This is a bounded causal-effect diagnostic only when its inputs come from valid
    interventions in an identified causal model or randomized experiment.
    """

    p_0 = _normalized(outcome_under_do_0)
    p_1 = _normalized(outcome_under_do_1)
    midpoint = 0.5 * (p_0 + p_1)
    return 0.5 * kl_bits(p_0, midpoint) + 0.5 * kl_bits(p_1, midpoint)


def matched_coupling_contrasts(
    coupled_efe: Sequence[float], decoupled_efe: Sequence[float]
) -> np.ndarray:
    """Return per-agent G[do(coupling=0)] - G[coupled], without aggregation."""

    coupled = np.asarray(coupled_efe, dtype=float)
    decoupled = np.asarray(decoupled_efe, dtype=float)
    if coupled.shape != decoupled.shape or coupled.ndim != 1:
        raise ValueError("EFE vectors must be one-dimensional with equal shape")
    return decoupled - coupled
