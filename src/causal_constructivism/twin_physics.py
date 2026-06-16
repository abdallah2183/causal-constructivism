from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from .models import (
    GaussianBelief,
    ObjectAction,
    VectorGaussianBelief,
)


@dataclass(slots=True)
class RigidBody1D:
    object_id: str
    mass: float
    radius: float
    restitution: float = 1.0
    friction: float = 0.0
    position: float = 0.0
    velocity: float = 0.0

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("Object ID is required")
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Mass must be finite and positive")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Radius must be finite and positive")
        if not 0 <= self.restitution <= 1:
            raise ValueError("Restitution must be in [0, 1]")
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("Friction must be finite and non-negative")
        if not math.isfinite(self.position) or not math.isfinite(self.velocity):
            raise ValueError("Position and velocity must be finite")


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    object_a: str
    object_b: str
    time: float
    impulse: float

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.object_a, self.object_b)))


@dataclass(frozen=True, slots=True)
class MultiObjectSnapshot:
    time: float
    states: dict[str, tuple[float, float]]
    collisions: tuple[CollisionEvent, ...]


class MultiObjectWorld1D:
    """Fixed-step rigid-body sandbox with impulse-based pair collisions."""

    def __init__(
        self,
        bodies: list[RigidBody1D],
        *,
        timestep: float = 0.005,
        bounds: tuple[float, float] | None = None,
    ) -> None:
        if not bodies:
            raise ValueError("At least one body is required")
        if timestep <= 0:
            raise ValueError("Timestep must be positive")
        if len({body.object_id for body in bodies}) != len(bodies):
            raise ValueError("Body IDs must be unique")
        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError("World bounds must be increasing")
        self.bodies = {body.object_id: replace(body) for body in bodies}
        self.timestep = timestep
        self.bounds = bounds
        self.time = 0.0

    def clone(self) -> MultiObjectWorld1D:
        clone = MultiObjectWorld1D(
            list(self.bodies.values()),
            timestep=self.timestep,
            bounds=self.bounds,
        )
        clone.time = self.time
        return clone

    def simulate(
        self,
        action: ObjectAction,
        *,
        horizon: float,
    ) -> MultiObjectSnapshot:
        if action.object_id not in self.bodies:
            raise KeyError(f"Unknown action target: {action.object_id}")
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("Simulation horizon must be finite and positive")
        elapsed = 0.0
        events: list[CollisionEvent] = []
        while elapsed < horizon - 1e-12:
            step = min(self.timestep, horizon - elapsed)
            force = action.force if elapsed < action.duration else 0.0
            self._integrate(step, action.object_id, force)
            events.extend(self._resolve_collisions(self.time + step))
            self._resolve_boundaries()
            elapsed += step
            self.time += step
        return MultiObjectSnapshot(
            time=self.time,
            states={
                object_id: (body.position, body.velocity)
                for object_id, body in self.bodies.items()
            },
            collisions=tuple(events),
        )

    def _integrate(self, step: float, target_id: str, force: float) -> None:
        for object_id, body in self.bodies.items():
            applied_force = force if object_id == target_id else 0.0
            if body.velocity:
                applied_force -= (
                    math.copysign(body.friction * body.mass * 9.81, body.velocity)
                )
            acceleration = applied_force / body.mass
            body.position += body.velocity * step + 0.5 * acceleration * step * step
            body.velocity += acceleration * step

    def _resolve_collisions(self, event_time: float) -> list[CollisionEvent]:
        events: list[CollisionEvent] = []
        ordered = sorted(self.bodies.values(), key=lambda body: body.position)
        for left, right in zip(ordered, ordered[1:]):
            separation = right.position - left.position
            minimum_separation = left.radius + right.radius
            if separation > minimum_separation:
                continue
            relative_velocity = right.velocity - left.velocity
            impulse = 0.0
            if relative_velocity < 0:
                restitution = min(left.restitution, right.restitution)
                impulse = -(1 + restitution) * relative_velocity / (
                    1 / left.mass + 1 / right.mass
                )
                left.velocity -= impulse / left.mass
                right.velocity += impulse / right.mass
                events.append(
                    CollisionEvent(
                        object_a=left.object_id,
                        object_b=right.object_id,
                        time=event_time,
                        impulse=impulse,
                    )
                )

            overlap = minimum_separation - separation
            if overlap > 0:
                inverse_left = 1 / left.mass
                inverse_right = 1 / right.mass
                inverse_total = inverse_left + inverse_right
                left.position -= overlap * inverse_left / inverse_total
                right.position += overlap * inverse_right / inverse_total
        return events

    def _resolve_boundaries(self) -> None:
        if self.bounds is None:
            return
        lower, upper = self.bounds
        for body in self.bodies.values():
            if body.position - body.radius < lower:
                body.position = lower + body.radius
                body.velocity = abs(body.velocity) * body.restitution
            elif body.position + body.radius > upper:
                body.position = upper - body.radius
                body.velocity = -abs(body.velocity) * body.restitution


@dataclass(frozen=True, slots=True)
class ParticleBodyPrior:
    object_id: str
    mass: GaussianBelief
    position: GaussianBelief
    velocity: GaussianBelief
    radius: float
    restitution: float = 1.0
    friction: float = 0.0


@dataclass(frozen=True, slots=True)
class ParticleRollout:
    states: dict[str, VectorGaussianBelief]
    collision_probability: float
    pair_collision_probabilities: dict[tuple[str, str], float]

    def belief_for(self, variable: str) -> GaussianBelief:
        if variable == "collision":
            probability = self.collision_probability
            return GaussianBelief(
                probability,
                max(probability * (1 - probability), 1e-9),
            )
        if variable.startswith("collision."):
            parts = variable.split(".")
            if len(parts) != 3:
                raise KeyError(f"Invalid collision query: {variable}")
            probability = self.pair_collision_probabilities.get(
                tuple(sorted((parts[1], parts[2]))),
                0.0,
            )
            return GaussianBelief(
                probability,
                max(probability * (1 - probability), 1e-9),
            )
        try:
            object_id, field = variable.rsplit(".", 1)
            state = self.states[object_id]
        except (KeyError, ValueError) as exc:
            raise KeyError(f"Unknown rollout variable: {variable}") from exc
        if field == "position":
            return state.component(0)
        if field == "velocity":
            return state.component(1)
        raise KeyError(f"Unsupported rollout field: {field}")


class ParticleCollisionFactor:
    """Particle adapter for the discontinuous collision factor."""

    def __init__(
        self,
        *,
        particles: int = 200,
        timestep: float = 0.005,
        seed: int = 17,
    ) -> None:
        if particles < 10:
            raise ValueError("At least ten particles are required")
        self.particles = particles
        self.timestep = timestep
        self.seed = seed

    def rollout(
        self,
        priors: list[ParticleBodyPrior],
        action: ObjectAction,
        *,
        horizon: float,
        interventions: dict[str, float] | None = None,
    ) -> ParticleRollout:
        interventions = interventions or {}
        randomizer = random.Random(self.seed)
        samples: dict[str, list[tuple[float, float]]] = {
            prior.object_id: [] for prior in priors
        }
        collision_count = 0
        pair_counts: dict[tuple[str, str], int] = {}

        for _ in range(self.particles):
            bodies = [
                self._sample_body(prior, randomizer, interventions)
                for prior in priors
            ]
            effective_action = ObjectAction(
                object_id=action.object_id,
                force=interventions.get("action.force", action.force),
                duration=interventions.get("action.duration", action.duration),
            )
            world = MultiObjectWorld1D(bodies, timestep=self.timestep)
            snapshot = world.simulate(effective_action, horizon=horizon)
            for object_id, state in snapshot.states.items():
                samples[object_id].append(state)
            collided_pairs = {event.pair for event in snapshot.collisions}
            if collided_pairs:
                collision_count += 1
            for pair in collided_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return ParticleRollout(
            states={
                object_id: VectorGaussianBelief.from_samples(values)
                for object_id, values in samples.items()
            },
            collision_probability=collision_count / self.particles,
            pair_collision_probabilities={
                pair: count / self.particles for pair, count in pair_counts.items()
            },
        )

    @staticmethod
    def _sample_body(
        prior: ParticleBodyPrior,
        randomizer: random.Random,
        interventions: dict[str, float],
    ) -> RigidBody1D:
        def sample_positive(belief: GaussianBelief) -> float:
            return max(
                1e-4,
                randomizer.gauss(belief.mean, math.sqrt(belief.variance)),
            )

        prefix = prior.object_id
        return RigidBody1D(
            object_id=prior.object_id,
            mass=interventions.get(f"{prefix}.mass", sample_positive(prior.mass)),
            radius=interventions.get(f"{prefix}.radius", prior.radius),
            restitution=interventions.get(
                f"{prefix}.restitution", prior.restitution
            ),
            friction=interventions.get(f"{prefix}.friction", prior.friction),
            position=interventions.get(
                f"{prefix}.position",
                randomizer.gauss(
                    prior.position.mean, math.sqrt(prior.position.variance)
                ),
            ),
            velocity=interventions.get(
                f"{prefix}.velocity",
                randomizer.gauss(
                    prior.velocity.mean, math.sqrt(prior.velocity.variance)
                ),
            ),
        )
