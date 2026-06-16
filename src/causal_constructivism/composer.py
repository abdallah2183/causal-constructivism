from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, GroundingAudit, NodeType


GRAVITY = 9.81


@dataclass(frozen=True, slots=True)
class CompoundObservation:
    object_id: str
    initial_velocity: float
    slide_duration: float
    impact_velocity: float
    rebound_velocity: float
    surface_id: str = "compound_ramp"
    wall_id: str = "compound_wall"

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("Object ID is required")
        for name, value in (
            ("initial_velocity", self.initial_velocity),
            ("slide_duration", self.slide_duration),
            ("impact_velocity", self.impact_velocity),
            ("rebound_velocity", self.rebound_velocity),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.initial_velocity <= 0:
            raise ValueError("Initial velocity must be positive")
        if self.slide_duration <= 0:
            raise ValueError("Slide duration must be positive")
        if self.impact_velocity < 0:
            raise ValueError("Impact velocity must be non-negative")
        if self.rebound_velocity > 0:
            raise ValueError("Rebound velocity should oppose the incoming direction")


@dataclass(frozen=True, slots=True)
class ConceptVocabularyEntry:
    name: str
    parameter_name: str
    law: str
    signature: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelFit:
    concepts: tuple[str, ...]
    parameters: dict[str, float]
    residual_error: float
    evidence_gain: float
    complexity_penalty: float

    @property
    def score(self) -> float:
        return self.evidence_gain - self.complexity_penalty


@dataclass(frozen=True, slots=True)
class DiscoveredConcept:
    name: str
    parameter_name: str
    parameter_value: float
    node_id: str
    audit: GroundingAudit


@dataclass(frozen=True, slots=True)
class ConceptInteraction:
    source_concept: str
    target_concept: str
    interaction_type: str
    node_id: str
    audit: GroundingAudit


@dataclass(frozen=True, slots=True)
class ComposerResult:
    selected_model: ModelFit
    candidate_models: tuple[ModelFit, ...]
    concepts: tuple[DiscoveredConcept, ...]
    interaction: ConceptInteraction | None
    graph_nodes: int
    graph_edges: int
    ungrounded_nodes: int


class ConceptGrammar:
    """Bounded physical vocabulary for the first multi-concept slice."""

    def __init__(self) -> None:
        self._entries = {
            "friction": ConceptVocabularyEntry(
                name="friction",
                parameter_name="coefficient",
                law="impact_velocity = max(0, initial_velocity - mu * g * slide_duration)",
                signature=("surface_contact", "sliding_deceleration"),
            ),
            "restitution": ConceptVocabularyEntry(
                name="restitution",
                parameter_name="coefficient",
                law="rebound_velocity = -e * impact_velocity",
                signature=("collision", "velocity_reversal"),
            ),
            "spring_constant": ConceptVocabularyEntry(
                name="spring_constant",
                parameter_name="k",
                law="restoring_force = -k * displacement",
                signature=("oscillation", "restoring_force"),
            ),
            "damping": ConceptVocabularyEntry(
                name="damping",
                parameter_name="c",
                law="damping_force = -c * velocity",
                signature=("amplitude_decay", "velocity_proportional_loss"),
            ),
            "center_of_mass": ConceptVocabularyEntry(
                name="center_of_mass",
                parameter_name="offset",
                law="torque = force x (application_point - center_of_mass)",
                signature=("tipping", "rotation_from_off_center_force"),
            ),
        }

    @property
    def entries(self) -> tuple[ConceptVocabularyEntry, ...]:
        return tuple(self._entries.values())

    def require(self, name: str) -> ConceptVocabularyEntry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"Unknown concept grammar entry: {name}") from exc


class CompoundModelSelector:
    def __init__(self, *, complexity_weight: float = 0.35) -> None:
        self.complexity_weight = complexity_weight

    def fit(self, observations: Iterable[CompoundObservation]) -> tuple[ModelFit, ...]:
        rows = tuple(observations)
        if not rows:
            raise ValueError("Model selection requires observations")
        baseline_error = _model_error(rows, friction=None, restitution=None)
        candidates = (
            self._fit_candidate(rows, baseline_error, ("friction",)),
            self._fit_candidate(rows, baseline_error, ("restitution",)),
            self._fit_candidate(rows, baseline_error, ("friction", "restitution")),
        )
        return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))

    def _fit_candidate(
        self,
        rows: tuple[CompoundObservation, ...],
        baseline_error: float,
        concepts: tuple[str, ...],
    ) -> ModelFit:
        friction = _fit_friction(rows) if "friction" in concepts else None
        restitution = _fit_restitution(rows, friction=friction) if "restitution" in concepts else None
        residual_error = _model_error(rows, friction=friction, restitution=restitution)
        evidence_gain = 0.5 * len(rows) * math.log(
            max(baseline_error, 1e-12) / max(residual_error, 1e-12)
        )
        complexity_penalty = self.complexity_weight * len(concepts) * math.log(len(rows) + 1)
        parameters = {}
        if friction is not None:
            parameters["friction.coefficient"] = friction
        if restitution is not None:
            parameters["restitution.coefficient"] = restitution
        return ModelFit(
            concepts=concepts,
            parameters=parameters,
            residual_error=residual_error,
            evidence_gain=evidence_gain,
            complexity_penalty=complexity_penalty,
        )


class ComposerSystem:
    def __init__(
        self,
        *,
        graph: CausalGraph | None = None,
        selector: CompoundModelSelector | None = None,
        grammar: ConceptGrammar | None = None,
    ) -> None:
        self.graph = graph or CausalGraph()
        self.selector = selector or CompoundModelSelector()
        self.grammar = grammar or ConceptGrammar()

    def discover(
        self,
        observations: Iterable[CompoundObservation],
        *,
        threshold: float = 2.0,
    ) -> ComposerResult:
        rows = tuple(observations)
        fits = self.selector.fit(rows)
        selected = fits[0]
        if selected.score <= threshold:
            raise ValueError("No compound model cleared the discovery threshold")
        object_node = _ensure_object(self.graph, rows[0].object_id)
        observation_node = _add_compound_observation_node(self.graph, rows)
        concepts = tuple(
            self._integrate_concept(
                concept_name,
                selected.parameters[f"{concept_name}.coefficient"],
                object_node_id=object_node.id,
                observation_node_id=observation_node.id,
                model=selected,
            )
            for concept_name in selected.concepts
        )
        interaction = None
        if selected.concepts == ("friction", "restitution"):
            interaction = self._integrate_interaction(concepts, selected)
        return ComposerResult(
            selected_model=selected,
            candidate_models=fits,
            concepts=concepts,
            interaction=interaction,
            graph_nodes=len(self.graph.nodes),
            graph_edges=len(self.graph.edges),
            ungrounded_nodes=len(self.graph.validate_grounding()),
        )

    def _integrate_concept(
        self,
        concept_name: str,
        value: float,
        *,
        object_node_id: str,
        observation_node_id: str,
        model: ModelFit,
    ) -> DiscoveredConcept:
        entry = self.grammar.require(concept_name)
        law_node = _ensure_law(self.graph, entry)
        node = self.graph.add_node(
            f"{concept_name}.{entry.parameter_name}",
            NodeType.CONCEPT,
            GaussianBelief(value, max(value * value * 0.0025, 1e-6)),
            modality="compound_discovery",
            metadata={
                "concept": concept_name,
                "parameter": entry.parameter_name,
                "model_score": model.score,
                "evidence_gain": model.evidence_gain,
                "residual_error": model.residual_error,
            },
        )
        variance = node.belief.variance
        self.graph.add_edge(
            object_node_id,
            node.id,
            EdgeType.PART_OF,
            noise_variance=variance,
            learned=True,
        )
        self.graph.add_edge(
            law_node.id,
            node.id,
            EdgeType.CAUSES,
            noise_variance=variance,
            learned=True,
        )
        self.graph.add_edge(
            observation_node_id,
            node.id,
            EdgeType.OBSERVES,
            noise_variance=variance,
            confidence=node.belief.confidence,
            learned=True,
        )
        return DiscoveredConcept(
            name=concept_name,
            parameter_name=entry.parameter_name,
            parameter_value=value,
            node_id=node.id,
            audit=MetacognitiveAuditor(self.graph).audit(node.id),
        )

    def _integrate_interaction(
        self,
        concepts: tuple[DiscoveredConcept, ...],
        model: ModelFit,
    ) -> ConceptInteraction:
        by_name = {concept.name: concept for concept in concepts}
        friction = by_name["friction"]
        restitution = by_name["restitution"]
        law = self.graph.add_node(
            "law.friction_restitution_energy_coupling",
            NodeType.LAW,
            GaussianBelief(1.0, 1e-9),
            is_axiom=True,
            metadata={
                "equation": "rebound_velocity = -e * max(0, v - mu * g * t)",
            },
        )
        interaction = self.graph.add_node(
            "interaction.friction.restitution",
            NodeType.RELATION,
            GaussianBelief(model.evidence_gain, 1e-4),
            modality="compound_interaction",
            metadata={
                "interaction_type": "friction_loss_modulates_restitution",
                "model_score": model.score,
                "residual_error": model.residual_error,
            },
        )
        self.graph.add_edge(
            friction.node_id,
            interaction.id,
            EdgeType.AFFECTS,
            noise_variance=1e-4,
            learned=True,
        )
        self.graph.add_edge(
            restitution.node_id,
            interaction.id,
            EdgeType.AFFECTS,
            noise_variance=1e-4,
            learned=True,
        )
        self.graph.add_edge(
            law.id,
            interaction.id,
            EdgeType.CAUSES,
            noise_variance=1e-4,
            learned=True,
        )
        return ConceptInteraction(
            source_concept="friction",
            target_concept="restitution",
            interaction_type="friction_loss_modulates_restitution",
            node_id=interaction.id,
            audit=MetacognitiveAuditor(self.graph).audit(interaction.id),
        )


def run_composer_benchmark(
    *,
    friction: float = 0.25,
    restitution: float = 0.65,
    count: int = 8,
) -> ComposerResult:
    observations = synthetic_compound_observations(
        friction=friction,
        restitution=restitution,
        count=count,
    )
    return ComposerSystem().discover(observations)


def synthetic_compound_observations(
    *,
    friction: float = 0.25,
    restitution: float = 0.65,
    count: int = 8,
) -> tuple[CompoundObservation, ...]:
    if friction < 0:
        raise ValueError("Friction must be non-negative")
    if not 0 <= restitution <= 1.5:
        raise ValueError("Restitution must be in [0, 1.5]")
    if count <= 0:
        raise ValueError("Observation count must be positive")
    rows = []
    for index in range(count):
        initial = 2.0 + 0.14 * index
        duration = 0.10 + 0.01 * (index % 4)
        impact = max(0.0, initial - friction * GRAVITY * duration)
        rebound = -restitution * impact
        rows.append(
            CompoundObservation(
                object_id="composer_block",
                initial_velocity=initial,
                slide_duration=duration,
                impact_velocity=impact,
                rebound_velocity=rebound,
            )
        )
    return tuple(rows)


def _fit_friction(rows: tuple[CompoundObservation, ...]) -> float:
    values = [
        (row.initial_velocity - row.impact_velocity) / (GRAVITY * row.slide_duration)
        for row in rows
        if row.slide_duration > 0
    ]
    return max(0.0, sum(values) / len(values))


def _fit_restitution(
    rows: tuple[CompoundObservation, ...],
    *,
    friction: float | None,
) -> float:
    values = []
    for row in rows:
        predicted_impact = _predict_impact(row, friction=friction)
        if predicted_impact > 1e-9:
            values.append(max(0.0, -row.rebound_velocity / predicted_impact))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _model_error(
    rows: tuple[CompoundObservation, ...],
    *,
    friction: float | None,
    restitution: float | None,
) -> float:
    error = 0.0
    for row in rows:
        predicted_impact = _predict_impact(row, friction=friction)
        predicted_rebound = -(restitution if restitution is not None else 1.0) * predicted_impact
        error += (row.impact_velocity - predicted_impact) ** 2
        error += (row.rebound_velocity - predicted_rebound) ** 2
    return error / len(rows)


def _predict_impact(row: CompoundObservation, *, friction: float | None) -> float:
    if friction is None:
        return row.initial_velocity
    return max(0.0, row.initial_velocity - friction * GRAVITY * row.slide_duration)


def _ensure_object(graph: CausalGraph, object_id: str):
    try:
        return graph.find_node_by_name(object_id)
    except KeyError:
        observation = graph.add_node(
            f"composer_observation.{object_id}",
            NodeType.OBSERVATION,
            GaussianBelief(1.0, 1e-6),
            evidence=GaussianBelief(1.0, 1e-6),
            modality="compound_scene",
        )
        node = graph.add_node(
            object_id,
            NodeType.OBJECT,
            GaussianBelief(1.0, 1e-6),
            modality="compound_identity",
            metadata={"object_id": object_id},
        )
        graph.add_edge(
            observation.id,
            node.id,
            EdgeType.OBSERVES,
            noise_variance=1e-6,
        )
        return node


def _add_compound_observation_node(
    graph: CausalGraph,
    rows: tuple[CompoundObservation, ...],
):
    mean_impact = sum(row.impact_velocity for row in rows) / len(rows)
    node = graph.add_node(
        "compound_observation.slide_collision",
        NodeType.OBSERVATION,
        GaussianBelief(mean_impact, 1e-6),
        evidence=GaussianBelief(mean_impact, 1e-6),
        modality="compound_scene",
        metadata={
            "records": len(rows),
            "surface_id": rows[0].surface_id,
            "wall_id": rows[0].wall_id,
        },
    )
    return node


def _ensure_law(graph: CausalGraph, entry: ConceptVocabularyEntry):
    name = f"law.{entry.name}"
    try:
        return graph.find_node_by_name(name)
    except KeyError:
        return graph.add_node(
            name,
            NodeType.LAW,
            GaussianBelief(1.0, 1e-9),
            is_axiom=True,
            metadata={"equation": entry.law, "signature": entry.signature},
        )
