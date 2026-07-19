"""Standing and capability-floor uncertainty for the finite AIF counterexample."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from finite_active_inference_model import (
    FiniteActiveInferenceModel,
    PolicyEvaluation,
    build_counterexample,
    normalize,
)


@dataclass(frozen=True)
class NormativeHypothesis:
    """One joint hypothesis about standing and the applicable capability floor."""

    standing: bool
    capability_floor: float
    probability: float


@dataclass(frozen=True)
class UncertainNormativeModel:
    active_inference: FiniteActiveInferenceModel
    hypotheses: tuple[NormativeHypothesis, ...]
    precautionary_risk_limit: float

    def violation_probability(self, evaluation: PolicyEvaluation) -> float:
        """Posterior mass on hypotheses under which the policy violates a floor."""
        return sum(
            hypothesis.probability
            for hypothesis in self.hypotheses
            if hypothesis.standing
            and evaluation.affected_viability < hypothesis.capability_floor
        )

    def uncertainty_weighted_posterior(
        self, evaluations: tuple[PolicyEvaluation, ...]
    ) -> tuple[float, ...]:
        """Weight policies by posterior probability of normative admissibility.

        This is a comparison rule, not the default moral recommendation: it permits
        compensation between normative uncertainty and focal value.
        """
        weights = tuple(
            self.active_inference.policy_prior[index]
            * exp(-self.active_inference.precision * evaluation.expected_free_energy)
            * (1.0 - self.violation_probability(evaluation))
            for index, evaluation in enumerate(evaluations)
        )
        return normalize(weights)

    def precautionary_posterior(
        self, evaluations: tuple[PolicyEvaluation, ...]
    ) -> tuple[float, ...]:
        """Exclude policies whose posterior violation risk exceeds a declared limit."""
        weights = tuple(
            self.active_inference.policy_prior[index]
            * exp(-self.active_inference.precision * evaluation.expected_free_energy)
            * float(self.violation_probability(evaluation) <= self.precautionary_risk_limit)
            for index, evaluation in enumerate(evaluations)
        )
        return normalize(weights)


def build_uncertain_counterexample() -> UncertainNormativeModel:
    active_inference = build_counterexample()
    # P(standing)=0.6. Conditional floor hypotheses are strict=0.5 with probability
    # 0.7 and permissive=0.05 with probability 0.3. Their product forms a normalized
    # joint distribution. When standing is false, the floor has no normative effect.
    hypotheses = (
        NormativeHypothesis(True, 0.5, 0.6 * 0.7),
        NormativeHypothesis(True, 0.05, 0.6 * 0.3),
        NormativeHypothesis(False, 0.5, 0.4 * 0.7),
        NormativeHypothesis(False, 0.05, 0.4 * 0.3),
    )
    return UncertainNormativeModel(
        active_inference=active_inference,
        hypotheses=hypotheses,
        precautionary_risk_limit=0.1,
    )


def run_uncertain_counterexample() -> dict[str, object]:
    model = build_uncertain_counterexample()
    posterior_states = model.active_inference.infer_states("reward")
    evaluations = tuple(
        model.active_inference.evaluate_policy(policy, posterior_states)
        for policy in model.active_inference.policies
    )
    violation_probabilities = tuple(
        model.violation_probability(evaluation) for evaluation in evaluations
    )
    return {
        "normative_hypotheses": model.hypotheses,
        "violation_probabilities": violation_probabilities,
        "unconstrained_policy_posterior": model.active_inference.policy_posterior(
            evaluations, constrained=False
        ),
        "uncertainty_weighted_policy_posterior": model.uncertainty_weighted_posterior(
            evaluations
        ),
        "precautionary_policy_posterior": model.precautionary_posterior(evaluations),
    }
