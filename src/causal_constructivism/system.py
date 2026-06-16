from __future__ import annotations

from dataclasses import dataclass

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .inference import GaussianBeliefPropagation
from .models import (
    ActionCandidate,
    ActionScore,
    EdgeType,
    GaussianBelief,
    GroundingAudit,
    InferenceResult,
    NodeType,
)
from .physics import BayesianMassEstimator, Body1D, PhysicsObservation, PhysicsWorld1D
from .planning import ExpectedFreeEnergyPlanner


@dataclass(frozen=True, slots=True)
class ExperimentStep:
    action_score: ActionScore
    observation: PhysicsObservation
    mass_mean: float
    mass_variance: float
    inference: InferenceResult
    audit: GroundingAudit


class CausalConstructivismSystem:
    """End-to-end active experiment loop for the Phase 1 prototype."""

    def __init__(
        self,
        *,
        true_mass: float = 2.5,
        sensor_noise_std: float = 0.05,
        seed: int = 7,
    ) -> None:
        self.graph = CausalGraph()
        self.world = PhysicsWorld1D(
            Body1D(mass=true_mass),
            sensor_noise_std=sensor_noise_std,
            seed=seed,
        )
        self.mass_estimator = BayesianMassEstimator(
            sensor_noise_std=max(sensor_noise_std, 1e-3)
        )
        self.planner = ExpectedFreeEnergyPlanner()
        self.inference = GaussianBeliefPropagation(self.graph)
        self.auditor = MetacognitiveAuditor(self.graph)
        self._sequence = 0

        self.law_node = self.graph.add_node(
            "newton_second_law",
            NodeType.LAW,
            GaussianBelief(1.0, 1e-6),
            is_axiom=True,
            metadata={"equation": "F = m * a"},
        )
        self.object_node = self.graph.add_node(
            "body_001",
            NodeType.OBJECT,
            GaussianBelief(1.0, 0.25),
        )
        self.graph.add_edge(
            self.law_node.id,
            self.object_node.id,
            EdgeType.INSTANCE_OF,
            noise_variance=0.1,
        )
        self.mass_node = self.graph.add_node(
            "body_001.mass",
            NodeType.PROPERTY,
            GaussianBelief(self.mass_estimator.mean, self.mass_estimator.variance),
            modality="physical",
        )
        self.graph.add_edge(
            self.object_node.id,
            self.mass_node.id,
            EdgeType.PART_OF,
            noise_variance=0.1,
        )

    def step(self, actions: list[ActionCandidate]) -> ExperimentStep:
        selected = self.planner.select(actions, self.mass_estimator)
        observation = self.world.execute(selected.action)
        self.world.reset_motion()
        self.mass_estimator.update(
            observation.force,
            observation.acceleration,
        )
        self._sequence += 1
        observation_node = self.graph.add_node(
            f"push_observation_{self._sequence:03d}",
            NodeType.OBSERVATION,
            GaussianBelief(self.mass_estimator.mean, self.mass_estimator.variance),
            modality="proprioceptive",
            evidence=GaussianBelief(
                self.mass_estimator.mean,
                self.mass_estimator.variance,
            ),
            metadata={
                "force": observation.force,
                "acceleration": observation.acceleration,
                "duration": observation.duration,
            },
        )
        self.graph.add_edge(
            observation_node.id,
            self.mass_node.id,
            EdgeType.OBSERVES,
            noise_variance=max(self.mass_estimator.variance, 1e-6),
            confidence=min(1.0, 1.0 / (1.0 + self.mass_estimator.variance)),
        )
        inference_result = self.inference.run([observation_node.id])
        audit = self.auditor.audit(self.mass_node.id)
        return ExperimentStep(
            action_score=selected,
            observation=observation,
            mass_mean=self.mass_estimator.mean,
            mass_variance=self.mass_estimator.variance,
            inference=inference_result,
            audit=audit,
        )

