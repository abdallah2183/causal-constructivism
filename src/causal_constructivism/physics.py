from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .models import ActionCandidate


@dataclass(slots=True)
class Body1D:
    mass: float
    friction: float = 0.0
    position: float = 0.0
    velocity: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Mass must be finite and positive")
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("Friction must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PhysicsObservation:
    force: float
    acceleration: float
    displacement: float
    duration: float


class PhysicsWorld1D:
    """Deterministic-step 1D rigid-body world with optional sensor noise."""

    def __init__(
        self,
        body: Body1D,
        *,
        sensor_noise_std: float = 0.05,
        seed: int = 7,
    ) -> None:
        if sensor_noise_std < 0:
            raise ValueError("Sensor noise must be non-negative")
        self.body = body
        self.sensor_noise_std = sensor_noise_std
        self._random = random.Random(seed)

    def execute(self, action: ActionCandidate) -> PhysicsObservation:
        if action.duration <= 0:
            raise ValueError("Action duration must be positive")
        friction_force = self.body.friction * self.body.mass * 9.81
        if self.body.velocity == 0 and abs(action.force) <= friction_force:
            acceleration = 0.0
        else:
            direction = 1.0 if action.force >= 0 else -1.0
            net_force = action.force - direction * friction_force
            acceleration = net_force / self.body.mass

        initial_velocity = self.body.velocity
        displacement = (
            initial_velocity * action.duration
            + 0.5 * acceleration * action.duration * action.duration
        )
        self.body.position += displacement
        self.body.velocity += acceleration * action.duration
        measured_acceleration = acceleration + self._random.gauss(
            0.0, self.sensor_noise_std
        )
        return PhysicsObservation(
            force=action.force,
            acceleration=measured_acceleration,
            displacement=displacement,
            duration=action.duration,
        )

    def reset_motion(self) -> None:
        self.body.velocity = 0.0


class BayesianMassEstimator:
    """Grid posterior for mass under the measurement model a = F / m."""

    def __init__(
        self,
        *,
        minimum_mass: float = 0.1,
        maximum_mass: float = 10.0,
        bins: int = 400,
        sensor_noise_std: float = 0.05,
    ) -> None:
        if minimum_mass <= 0 or maximum_mass <= minimum_mass:
            raise ValueError("Invalid mass interval")
        if bins < 2:
            raise ValueError("At least two mass bins are required")
        if sensor_noise_std <= 0:
            raise ValueError("Sensor noise must be positive")
        step = (maximum_mass - minimum_mass) / (bins - 1)
        self.masses = [minimum_mass + index * step for index in range(bins)]
        self.probabilities = [1.0 / bins] * bins
        self.sensor_variance = sensor_noise_std * sensor_noise_std

    def update(self, force: float, measured_acceleration: float) -> None:
        log_weights = []
        for mass, prior_probability in zip(
            self.masses, self.probabilities, strict=True
        ):
            error = measured_acceleration - force / mass
            log_likelihood = -0.5 * error * error / self.sensor_variance
            log_weights.append(math.log(max(prior_probability, 1e-300)) + log_likelihood)
        maximum = max(log_weights)
        weights = [math.exp(item - maximum) for item in log_weights]
        normalizer = sum(weights)
        self.probabilities = [item / normalizer for item in weights]

    @property
    def mean(self) -> float:
        return sum(
            mass * probability
            for mass, probability in zip(
                self.masses, self.probabilities, strict=True
            )
        )

    @property
    def variance(self) -> float:
        mean = self.mean
        return max(
            sum(
                probability * (mass - mean) ** 2
                for mass, probability in zip(
                    self.masses, self.probabilities, strict=True
                )
            ),
            1e-9,
        )

    @property
    def entropy(self) -> float:
        return -sum(
            probability * math.log(max(probability, 1e-300))
            for probability in self.probabilities
        )

    def expected_entropy(self, force: float) -> float:
        """Approximate posterior entropy at the current posterior mean outcome."""
        predicted_acceleration = force / self.mean
        clone = object.__new__(BayesianMassEstimator)
        clone.masses = list(self.masses)
        clone.probabilities = list(self.probabilities)
        clone.sensor_variance = self.sensor_variance
        clone.update(force, predicted_acceleration)
        return clone.entropy

