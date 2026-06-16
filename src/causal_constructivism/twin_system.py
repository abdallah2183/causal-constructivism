from __future__ import annotations

from .contact_learning import ContactRelationLearner
from .counterfactual import CounterfactualEngine, EvidenceValue, build_twin_graph
from .models import CounterfactualResult, GaussianBelief, ObjectAction
from .twin_physics import (
    MultiObjectSnapshot,
    MultiObjectWorld1D,
    ParticleBodyPrior,
    ParticleCollisionFactor,
    RigidBody1D,
)


class TwinWorldSystem:
    """Phase 2 vertical slice for multi-object counterfactual physics."""

    def __init__(
        self,
        priors: list[ParticleBodyPrior],
        action: ObjectAction,
        *,
        particles: int = 200,
        timestep: float = 0.005,
        horizon: float = 2.0,
        seed: int = 17,
    ) -> None:
        self.priors = list(priors)
        self.action = action
        self.horizon = horizon
        self.graph = build_twin_graph(priors, action)
        self.particle_factor = ParticleCollisionFactor(
            particles=particles,
            timestep=timestep,
            seed=seed,
        )
        self.counterfactual_engine = CounterfactualEngine(
            self.graph,
            self.priors,
            self.action,
            particle_factor=self.particle_factor,
            horizon=horizon,
        )
        self.contact_learner = ContactRelationLearner(self.graph)
        self.timestep = timestep

    def observe_actual_world(self) -> MultiObjectSnapshot:
        bodies = [
            RigidBody1D(
                object_id=prior.object_id,
                mass=prior.mass.mean,
                radius=prior.radius,
                restitution=prior.restitution,
                friction=prior.friction,
                position=prior.position.mean,
                velocity=prior.velocity.mean,
            )
            for prior in self.priors
        ]
        snapshot = MultiObjectWorld1D(
            bodies,
            timestep=self.timestep,
        ).simulate(self.action, horizon=self.horizon)
        self.contact_learner.observe(
            snapshot,
            [prior.object_id for prior in self.priors],
        )
        return snapshot

    def learn_contact_relations(self, repetitions: int = 3) -> tuple[str, ...]:
        if repetitions <= 0:
            raise ValueError("Repetitions must be positive")
        for _ in range(repetitions):
            self.observe_actual_world()
        integrated: list[str] = []
        object_ids = [prior.object_id for prior in self.priors]
        for index, left in enumerate(object_ids):
            for right in object_ids[index + 1 :]:
                proposal = self.contact_learner.propose(left, right)
                if proposal is not None and proposal.score > 0:
                    relation_name = f"contact.{proposal.object_a}.{proposal.object_b}"
                    try:
                        self.graph.find_node_by_name(relation_name)
                    except KeyError:
                        integrated.append(self.contact_learner.integrate(proposal))
        return tuple(integrated)

    def counterfactual(
        self,
        intervention: dict[str, float],
        query_var: str,
        *,
        evidence: dict[str, EvidenceValue] | None = None,
        time_horizon: float | None = None,
    ) -> CounterfactualResult:
        return self.counterfactual_engine.query(
            intervention,
            query_var,
            evidence=evidence,
            time_horizon=time_horizon,
        )


def default_twin_world(*, particles: int = 200) -> TwinWorldSystem:
    priors = [
        ParticleBodyPrior(
            object_id="red",
            mass=GaussianBelief(2.0, 0.04),
            position=GaussianBelief(0.5, 0.0025),
            velocity=GaussianBelief(0.0, 0.0025),
            radius=0.5,
            restitution=0.9,
        ),
        ParticleBodyPrior(
            object_id="green",
            mass=GaussianBelief(1.0, 0.01),
            position=GaussianBelief(3.0, 0.0025),
            velocity=GaussianBelief(0.0, 0.0025),
            radius=0.5,
            restitution=0.9,
        ),
    ]
    return TwinWorldSystem(
        priors,
        ObjectAction(object_id="red", force=10.0, duration=0.5),
        particles=particles,
        horizon=2.0,
    )
