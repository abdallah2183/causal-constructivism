from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .models import (
    CounterfactualResult,
    EdgeType,
    GaussianBelief,
    NodeType,
    ObjectAction,
)
from .twin_physics import ParticleBodyPrior, ParticleCollisionFactor


EvidenceValue = GaussianBelief | tuple[float, float]


class CounterfactualEngine:
    """Abduction-action-prediction over a particle-based twin world."""

    def __init__(
        self,
        graph: CausalGraph,
        priors: list[ParticleBodyPrior],
        action: ObjectAction,
        *,
        particle_factor: ParticleCollisionFactor | None = None,
        horizon: float = 2.0,
    ) -> None:
        self.graph = graph
        self.priors = list(priors)
        self.action = action
        self.particle_factor = particle_factor or ParticleCollisionFactor()
        self.horizon = horizon

    def query(
        self,
        intervention: Mapping[str, float],
        query_var: str,
        *,
        evidence: Mapping[str, EvidenceValue] | None = None,
        time_horizon: float | None = None,
    ) -> CounterfactualResult:
        horizon = time_horizon or self.horizon
        actual_graph = self.graph.clone()
        abducted_priors = self._abduct(actual_graph, evidence or {})
        actual_rollout = self.particle_factor.rollout(
            abducted_priors,
            self.action,
            horizon=horizon,
        )

        counterfactual_graph = actual_graph.clone()
        intervention_node_ids: list[str] = []
        for variable, value in intervention.items():
            node = counterfactual_graph.find_node_by_name(variable)
            counterfactual_graph.intervene(node.id, value)
            intervention_node_ids.append(node.id)

        counterfactual_rollout = self.particle_factor.rollout(
            abducted_priors,
            self.action,
            horizon=horizon,
            interventions=dict(intervention),
        )
        query_belief = counterfactual_rollout.belief_for(query_var)
        actual_belief = actual_rollout.belief_for(query_var)

        query_node = counterfactual_graph.add_node(
            f"counterfactual.{query_var}",
            NodeType.STATE,
            query_belief,
            modality="simulated",
            metadata={
                "counterfactual": True,
                "query_variable": query_var,
                "time_horizon": horizon,
                "collision_probability": (
                    counterfactual_rollout.collision_probability
                ),
            },
        )
        shared_exogenous_ids = self._connect_provenance(
            counterfactual_graph,
            query_node.id,
            intervention_node_ids,
        )
        audit = MetacognitiveAuditor(counterfactual_graph).audit(query_node.id)
        return CounterfactualResult(
            query_variable=query_var,
            belief=query_belief,
            actual_belief=actual_belief,
            collision_probability=counterfactual_rollout.collision_probability,
            actual_collision_probability=actual_rollout.collision_probability,
            grounding=audit,
            intervention_node_ids=tuple(intervention_node_ids),
            shared_exogenous_node_ids=shared_exogenous_ids,
        )

    def _abduct(
        self,
        graph: CausalGraph,
        evidence: Mapping[str, EvidenceValue],
    ) -> list[ParticleBodyPrior]:
        beliefs: dict[str, GaussianBelief] = {}
        for variable, raw_value in evidence.items():
            belief = (
                raw_value
                if isinstance(raw_value, GaussianBelief)
                else GaussianBelief(raw_value[0], raw_value[1])
            )
            variable_node = graph.find_node_by_name(variable)
            observation = graph.add_node(
                f"evidence.{variable}",
                NodeType.OBSERVATION,
                belief,
                evidence=belief,
                modality="counterfactual_evidence",
            )
            graph.add_edge(
                observation.id,
                variable_node.id,
                EdgeType.OBSERVES,
                noise_variance=belief.variance,
            )
            variable_node.belief = GaussianBelief.fuse(
                [variable_node.belief, belief]
            )
            beliefs[variable] = variable_node.belief

        abducted: list[ParticleBodyPrior] = []
        for prior in self.priors:
            prefix = prior.object_id
            abducted.append(
                replace(
                    prior,
                    mass=beliefs.get(f"{prefix}.mass", prior.mass),
                    position=beliefs.get(f"{prefix}.position", prior.position),
                    velocity=beliefs.get(f"{prefix}.velocity", prior.velocity),
                    radius=beliefs.get(
                        f"{prefix}.radius",
                        GaussianBelief(prior.radius, 1e-9),
                    ).mean,
                    restitution=beliefs.get(
                        f"{prefix}.restitution",
                        GaussianBelief(prior.restitution, 1e-9),
                    ).mean,
                    friction=beliefs.get(
                        f"{prefix}.friction",
                        GaussianBelief(max(prior.friction, 0.0), 1e-9),
                    ).mean,
                )
            )
        return abducted

    @staticmethod
    def _connect_provenance(
        graph: CausalGraph,
        query_node_id: str,
        intervention_node_ids: list[str],
    ) -> tuple[str, ...]:
        shared_ids: list[str] = []
        intervention_ids = set(intervention_node_ids)
        for node in graph.iter_active_nodes():
            if node.id == query_node_id:
                continue
            is_shared_exogenous = (
                node.name.endswith(
                    (
                        ".mass",
                        ".position",
                        ".velocity",
                        ".radius",
                        ".restitution",
                        ".friction",
                    )
                )
            )
            is_causal_context = (
                is_shared_exogenous
                or node.node_type in {NodeType.LAW, NodeType.ACTION}
            )
            if node.id in intervention_ids or is_causal_context:
                graph.add_edge(
                    node.id,
                    query_node_id,
                    EdgeType.AFFECTS,
                    noise_variance=1e-6,
                    metadata={"counterfactual_provenance": True},
                )
                if is_shared_exogenous and node.id not in intervention_ids:
                    shared_ids.append(node.id)
        return tuple(shared_ids)


def build_twin_graph(
    priors: list[ParticleBodyPrior],
    action: ObjectAction,
) -> CausalGraph:
    graph = CausalGraph()
    law = graph.add_node(
        "collision_dynamics",
        NodeType.LAW,
        GaussianBelief(1.0, 1e-9),
        is_axiom=True,
        metadata={"equations": ("F=ma", "impulse-momentum")},
    )
    for prior in priors:
        object_node = graph.add_node(
            prior.object_id,
            NodeType.OBJECT,
            GaussianBelief(1.0, 1e-6),
        )
        graph.add_edge(
            law.id,
            object_node.id,
            EdgeType.INSTANCE_OF,
            noise_variance=1e-6,
        )
        for field, belief in (
            ("mass", prior.mass),
            ("position", prior.position),
            ("velocity", prior.velocity),
            ("radius", GaussianBelief(prior.radius, 1e-9)),
            ("restitution", GaussianBelief(prior.restitution, 1e-9)),
            ("friction", GaussianBelief(max(prior.friction, 0.0), 1e-9)),
        ):
            property_node = graph.add_node(
                f"{prior.object_id}.{field}",
                (
                    NodeType.STATE
                    if field in {"position", "velocity"}
                    else NodeType.PROPERTY
                ),
                belief,
                modality="physical",
            )
            observation = graph.add_node(
                f"initial_observation.{prior.object_id}.{field}",
                NodeType.OBSERVATION,
                belief,
                evidence=belief,
                modality="simulated_sensor",
            )
            graph.add_edge(
                object_node.id,
                property_node.id,
                EdgeType.HAS_STATE if field != "mass" else EdgeType.PART_OF,
                noise_variance=1e-6,
            )
            graph.add_edge(
                observation.id,
                property_node.id,
                EdgeType.OBSERVES,
                noise_variance=belief.variance,
            )

    action_node = graph.add_node(
        "action.force",
        NodeType.ACTION,
        GaussianBelief(action.force, 1e-9),
        modality="motor",
    )
    action_observation = graph.add_node(
        "actual_action.force",
        NodeType.OBSERVATION,
        GaussianBelief(action.force, 1e-9),
        evidence=GaussianBelief(action.force, 1e-9),
        modality="motor_command",
    )
    graph.add_edge(
        action_observation.id,
        action_node.id,
        EdgeType.OBSERVES,
        noise_variance=1e-9,
    )
    duration_node = graph.add_node(
        "action.duration",
        NodeType.ACTION,
        GaussianBelief(action.duration, 1e-9),
        modality="motor",
    )
    duration_observation = graph.add_node(
        "actual_action.duration",
        NodeType.OBSERVATION,
        GaussianBelief(action.duration, 1e-9),
        evidence=GaussianBelief(action.duration, 1e-9),
        modality="motor_command",
    )
    graph.add_edge(
        duration_observation.id,
        duration_node.id,
        EdgeType.OBSERVES,
        noise_variance=1e-9,
    )
    return graph
