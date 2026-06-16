from __future__ import annotations

from .models import ActionCandidate, ActionScore
from .physics import BayesianMassEstimator


class ExpectedFreeEnergyPlanner:
    def __init__(
        self,
        *,
        displacement_weight: float = 0.05,
        target_displacement: float = 0.0,
    ) -> None:
        if displacement_weight < 0:
            raise ValueError("Displacement weight must be non-negative")
        self.displacement_weight = displacement_weight
        self.target_displacement = target_displacement

    def score(
        self,
        action: ActionCandidate,
        mass_estimator: BayesianMassEstimator,
    ) -> ActionScore:
        expected_entropy = mass_estimator.expected_entropy(action.force)
        expected_acceleration = action.force / mass_estimator.mean
        expected_displacement = (
            0.5 * expected_acceleration * action.duration * action.duration
        )
        pragmatic_cost = self.displacement_weight * (
            expected_displacement - self.target_displacement
        ) ** 2
        return ActionScore(
            action=action,
            expected_free_energy=expected_entropy + pragmatic_cost,
            epistemic_cost=expected_entropy,
            pragmatic_cost=pragmatic_cost,
        )

    def select(
        self,
        actions: list[ActionCandidate],
        mass_estimator: BayesianMassEstimator,
    ) -> ActionScore:
        if not actions:
            raise ValueError("At least one action candidate is required")
        return min(
            (self.score(action, mass_estimator) for action in actions),
            key=lambda item: item.expected_free_energy,
        )

