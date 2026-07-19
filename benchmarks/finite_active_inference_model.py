"""Finite probabilistic active-inference counterexample.

Two policies have exactly the same focal-agent generative model and expected free
energy, but different causal effects on an affected agent. An unconstrained policy
posterior cannot distinguish them; an explicit capability-floor predicate can.

This is a small exact model of the claimed logical separation. It is not an empirical
model of human moral cognition and does not derive the capability floor from EFE.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, log
from typing import Mapping


Distribution = tuple[float, ...]
Matrix = tuple[Distribution, ...]


def normalize(weights: Distribution) -> Distribution:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("normalization requires positive total mass")
    return tuple(weight / total for weight in weights)


def entropy(distribution: Distribution) -> float:
    return -sum(probability * log(probability) for probability in distribution if probability)


def kl_divergence(q: Distribution, p: Distribution) -> float:
    if len(q) != len(p):
        raise ValueError("distributions must have equal dimension")
    return sum(qi * log(qi / pi) for qi, pi in zip(q, p) if qi)


def bayes_posterior(
    prior: Distribution, likelihood: Matrix, observation_index: int
) -> Distribution:
    return normalize(
        tuple(
            prior[state] * likelihood[state][observation_index]
            for state in range(len(prior))
        )
    )


def predict_states(posterior: Distribution, transition: Matrix) -> Distribution:
    state_count = len(posterior)
    return tuple(
        sum(posterior[current] * transition[current][future] for current in range(state_count))
        for future in range(state_count)
    )


def predict_observations(predicted_states: Distribution, likelihood: Matrix) -> Distribution:
    observation_count = len(likelihood[0])
    return tuple(
        sum(
            predicted_states[state] * likelihood[state][observation]
            for state in range(len(predicted_states))
        )
        for observation in range(observation_count)
    )


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: str
    predicted_states: Distribution
    predicted_observations: Distribution
    risk: float
    ambiguity: float
    expected_free_energy: float
    pragmatic_cost: float
    information_gain: float
    affected_viability: float
    admissible: bool


@dataclass(frozen=True)
class FiniteActiveInferenceModel:
    state_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    policies: tuple[str, ...]
    prior_states: Distribution
    likelihood: Matrix
    focal_transitions: Mapping[str, Matrix]
    preferred_observations: Distribution
    policy_prior: Distribution
    affected_viability_under_intervention: Mapping[str, float]
    affected_capability_floor: float
    precision: float = 1.0

    def infer_states(self, observation: str) -> Distribution:
        return bayes_posterior(
            self.prior_states, self.likelihood, self.observation_names.index(observation)
        )

    def evaluate_policy(self, policy: str, posterior: Distribution) -> PolicyEvaluation:
        predicted_states = predict_states(posterior, self.focal_transitions[policy])
        predicted_observations = predict_observations(predicted_states, self.likelihood)

        risk = kl_divergence(predicted_observations, self.preferred_observations)
        ambiguity = sum(
            predicted_states[state] * entropy(self.likelihood[state])
            for state in range(len(predicted_states))
        )
        pragmatic_cost = -sum(
            predicted_observations[index] * log(self.preferred_observations[index])
            for index in range(len(predicted_observations))
        )
        information_gain = entropy(predicted_observations) - ambiguity
        expected_free_energy = risk + ambiguity
        affected_viability = self.affected_viability_under_intervention[policy]

        return PolicyEvaluation(
            policy=policy,
            predicted_states=predicted_states,
            predicted_observations=predicted_observations,
            risk=risk,
            ambiguity=ambiguity,
            expected_free_energy=expected_free_energy,
            pragmatic_cost=pragmatic_cost,
            information_gain=information_gain,
            affected_viability=affected_viability,
            admissible=affected_viability >= self.affected_capability_floor,
        )

    def policy_posterior(
        self, evaluations: tuple[PolicyEvaluation, ...], constrained: bool
    ) -> Distribution:
        weights = []
        for index, evaluation in enumerate(evaluations):
            allowed = evaluation.admissible or not constrained
            weights.append(
                self.policy_prior[index]
                * exp(-self.precision * evaluation.expected_free_energy)
                * float(allowed)
            )
        return normalize(tuple(weights))


def build_counterexample() -> FiniteActiveInferenceModel:
    # Rows are current hidden states; columns are future hidden states.
    focal_transition: Matrix = (
        (0.9, 0.1),  # viable -> viable/non-viable
        (0.4, 0.6),  # non-viable -> viable/non-viable
    )
    return FiniteActiveInferenceModel(
        state_names=("viable", "non_viable"),
        observation_names=("reward", "loss"),
        policies=("cooperate", "externalize"),
        prior_states=(0.5, 0.5),
        # Rows P(observation | state).
        likelihood=((0.9, 0.1), (0.2, 0.8)),
        # The focal model is intentionally identical under both policies.
        focal_transitions={
            "cooperate": focal_transition,
            "externalize": focal_transition,
        },
        preferred_observations=(0.8, 0.2),
        policy_prior=(0.5, 0.5),
        # These are interventional effects on the affected agent:
        # P(affected viable at t+1 | do(policy)).
        affected_viability_under_intervention={
            "cooperate": 0.9,
            "externalize": 0.1,
        },
        affected_capability_floor=0.5,
        precision=1.0,
    )


def run_counterexample() -> dict[str, object]:
    model = build_counterexample()
    posterior = model.infer_states("reward")
    evaluations = tuple(model.evaluate_policy(policy, posterior) for policy in model.policies)
    unconstrained = model.policy_posterior(evaluations, constrained=False)
    constrained = model.policy_posterior(evaluations, constrained=True)

    for evaluation in evaluations:
        if not isclose(
            evaluation.expected_free_energy,
            evaluation.pragmatic_cost - evaluation.information_gain,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError("EFE decompositions disagree")

    return {
        "posterior_after_reward": posterior,
        "evaluations": evaluations,
        "unconstrained_policy_posterior": unconstrained,
        "constrained_policy_posterior": constrained,
    }
