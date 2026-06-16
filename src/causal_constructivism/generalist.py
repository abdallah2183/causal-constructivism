from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .audit import MetacognitiveAuditor
from .discovery import (
    ExperimentLog,
    ExperimentRecord,
    fit_friction_coefficient,
    predict_with_friction,
    synthetic_sliding_records,
)
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, GroundingAudit, NodeType


@dataclass(frozen=True, slots=True)
class Revision:
    environment: str
    concept: str
    modification: str
    old_value: float
    new_value: float
    evidence_gain: float
    timestamp: str


@dataclass(slots=True)
class Concept:
    name: str
    parameter_name: str
    parameter_value: float
    applicability_signature: dict[str, object]
    confidence: float
    revision_history: list[Revision] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["revision_history"] = [
            asdict(revision) for revision in self.revision_history
        ]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> Concept:
        revisions = [
            Revision(**revision)
            for revision in payload.get("revision_history", [])
        ]
        return cls(
            name=str(payload["name"]),
            parameter_name=str(payload["parameter_name"]),
            parameter_value=float(payload["parameter_value"]),
            applicability_signature=dict(payload["applicability_signature"]),
            confidence=float(payload["confidence"]),
            revision_history=revisions,
        )


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    concept: Concept
    score: float


@dataclass(frozen=True, slots=True)
class TransferResult:
    concept: Concept
    property_node_id: str
    audit: GroundingAudit


@dataclass(frozen=True, slots=True)
class RevisionResult:
    revised_concept: Concept
    property_node_id: str
    revision: Revision
    audit: GroundingAudit


@dataclass(frozen=True, slots=True)
class GeneralistBenchmarkResult:
    learned_concept: Concept
    transferred_node_id: str
    revised_concept: Concept
    revision: Revision
    transfer_audit: GroundingAudit
    revision_audit: GroundingAudit
    pre_revision_error: float
    post_revision_error: float
    graph_nodes: int
    graph_edges: int


class ConceptLibrary:
    def __init__(self, concepts: list[Concept] | None = None) -> None:
        self._concepts: dict[str, Concept] = {}
        for concept in concepts or []:
            self.add(concept)

    @property
    def concepts(self) -> tuple[Concept, ...]:
        return tuple(self._concepts.values())

    def add(self, concept: Concept) -> None:
        if not concept.name:
            raise ValueError("Concept name is required")
        if not 0 <= concept.confidence <= 1:
            raise ValueError("Concept confidence must be in [0, 1]")
        self._concepts[concept.name] = concept

    def require(self, name: str) -> Concept:
        try:
            return self._concepts[name]
        except KeyError as exc:
            raise KeyError(f"Unknown concept: {name}") from exc

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [concept.to_json() for concept in self.concepts]
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ConceptLibrary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Concept.from_json(item) for item in payload])


class TransferEngine:
    def __init__(self, library: ConceptLibrary, *, threshold: float = 0.8) -> None:
        self.library = library
        self.threshold = threshold

    def environment_signature(self, log: ExperimentLog) -> dict[str, object]:
        records = log.records
        if not records:
            raise ValueError("Environment signature requires records")
        stop_fraction = sum(
            1
            for record in records
            if abs(record.final_velocity) < 0.75 * abs(record.initial_velocity)
        ) / len(records)
        mean_deceleration = sum(
            max(
                0.0,
                -math.copysign(1.0, record.initial_velocity or 1.0)
                * record.observed_acceleration,
            )
            for record in records
        ) / len(records)
        return {
            "motion_type": "sliding",
            "surface_interaction": True,
            "deceleration": mean_deceleration > 0.01 or stop_fraction > 0.25,
            "surface_id": records[0].surface_id,
        }

    def match(self, signature: dict[str, object]) -> tuple[ConceptMatch, ...]:
        matches = []
        for concept in self.library.concepts:
            score = _signature_overlap(signature, concept.applicability_signature)
            if score >= self.threshold:
                matches.append(ConceptMatch(concept, score))
        return tuple(sorted(matches, key=lambda item: item.score, reverse=True))

    def transfer(
        self,
        graph: CausalGraph,
        concept: Concept,
        *,
        target_object_id: str,
        environment_name: str,
        prior_scale: float = 4.0,
    ) -> TransferResult:
        if concept.name != "friction":
            raise ValueError(f"Unsupported transfer concept: {concept.name}")
        object_node = _ensure_object(graph, target_object_id)
        law_node = _ensure_friction_law(graph)
        variance = max((abs(concept.parameter_value) * prior_scale) ** 2, 1e-4)
        property_node = graph.add_node(
            f"{target_object_id}.friction.transferred.{environment_name}",
            NodeType.PROPERTY,
            GaussianBelief(concept.parameter_value, variance),
            modality="transferred_concept",
            metadata={
                "concept": concept.name,
                "parameter": concept.parameter_name,
                "environment": environment_name,
                "transferred": True,
                "source_confidence": concept.confidence,
            },
        )
        library_observation = graph.add_node(
            f"concept_library.{concept.name}.{environment_name}",
            NodeType.OBSERVATION,
            GaussianBelief(concept.confidence, 1e-6),
            evidence=GaussianBelief(concept.confidence, 1e-6),
            modality="concept_library",
            metadata={
                "concept": concept.name,
                "environment": environment_name,
                "transfer": True,
            },
        )
        graph.add_edge(
            object_node.id,
            property_node.id,
            EdgeType.PART_OF,
            noise_variance=variance,
            learned=True,
        )
        graph.add_edge(
            law_node.id,
            property_node.id,
            EdgeType.CAUSES,
            noise_variance=variance,
            learned=True,
        )
        graph.add_edge(
            library_observation.id,
            property_node.id,
            EdgeType.OBSERVES,
            noise_variance=variance,
            confidence=concept.confidence,
            learned=True,
        )
        return TransferResult(
            concept=concept,
            property_node_id=property_node.id,
            audit=MetacognitiveAuditor(graph).audit(property_node.id),
        )


class RevisionEngine:
    def __init__(
        self,
        *,
        contradiction_threshold: float = 0.05,
        evidence_threshold: float = 1.0,
    ) -> None:
        self.contradiction_threshold = contradiction_threshold
        self.evidence_threshold = evidence_threshold

    def prediction_error(
        self,
        records: tuple[ExperimentRecord, ...],
        coefficient: float,
    ) -> float:
        if not records:
            raise ValueError("Prediction error requires records")
        return sum(
            (record.final_velocity - predict_with_friction(record, coefficient)) ** 2
            for record in records
        ) / len(records)

    def revise(
        self,
        graph: CausalGraph,
        concept: Concept,
        *,
        property_node_id: str,
        log: ExperimentLog,
        environment_name: str,
    ) -> RevisionResult | None:
        records = log.records
        old_error = self.prediction_error(records, concept.parameter_value)
        if old_error < self.contradiction_threshold:
            return None
        new_value = fit_friction_coefficient(records)
        new_error = self.prediction_error(records, new_value)
        evidence_gain = 0.5 * len(records) * math.log(
            max(old_error, 1e-12) / max(new_error, 1e-12)
        )
        if evidence_gain <= self.evidence_threshold:
            return None
        node = graph.require_node(property_node_id)
        replacement = graph.version_node(
            node.id,
            prior=GaussianBelief(new_value, max(0.01 * new_value * new_value, 1e-6)),
            metadata_updates={
                "revised": True,
                "old_value": concept.parameter_value,
                "new_value": new_value,
                "environment": environment_name,
                "evidence_gain": evidence_gain,
            },
        )
        observation = graph.add_node(
            f"revision_observation.{concept.name}.{environment_name}",
            NodeType.OBSERVATION,
            replacement.belief,
            evidence=replacement.belief,
            modality="revision_experiment",
            metadata={
                "concept": concept.name,
                "environment": environment_name,
                "records": len(records),
                "old_error": old_error,
                "new_error": new_error,
            },
        )
        graph.add_edge(
            observation.id,
            replacement.id,
            EdgeType.OBSERVES,
            noise_variance=replacement.belief.variance,
            confidence=replacement.belief.confidence,
            learned=True,
        )
        revision = Revision(
            environment=environment_name,
            concept=concept.name,
            modification="parameter_refinement",
            old_value=concept.parameter_value,
            new_value=new_value,
            evidence_gain=evidence_gain,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        revised = Concept(
            name=concept.name,
            parameter_name=concept.parameter_name,
            parameter_value=new_value,
            applicability_signature={
                **concept.applicability_signature,
                "surface_id": records[0].surface_id,
            },
            confidence=min(1.0, concept.confidence + 0.05),
            revision_history=[*concept.revision_history, revision],
        )
        return RevisionResult(
            revised_concept=revised,
            property_node_id=replacement.id,
            revision=revision,
            audit=MetacognitiveAuditor(graph).audit(replacement.id),
        )


class GeneralistSystem:
    def __init__(self, library: ConceptLibrary | None = None) -> None:
        self.library = library or ConceptLibrary()
        self.graph = CausalGraph()
        self.transfer_engine = TransferEngine(self.library)
        self.revision_engine = RevisionEngine()

    def learn_friction_concept(
        self,
        *,
        coefficient: float,
        surface_id: str = "wood",
    ) -> Concept:
        concept = Concept(
            name="friction",
            parameter_name="coefficient",
            parameter_value=coefficient,
            applicability_signature={
                "motion_type": "sliding",
                "surface_interaction": True,
                "deceleration": True,
                "surface_id": surface_id,
            },
            confidence=0.95,
        )
        self.library.add(concept)
        return concept

    def transfer_and_revise(
        self,
        *,
        target_object_id: str,
        environment_name: str,
        log: ExperimentLog,
    ) -> GeneralistBenchmarkResult:
        signature = self.transfer_engine.environment_signature(log)
        matches = self.transfer_engine.match(signature)
        if not matches:
            raise RuntimeError("No concept matched the target environment")
        transfer = self.transfer_engine.transfer(
            self.graph,
            matches[0].concept,
            target_object_id=target_object_id,
            environment_name=environment_name,
        )
        pre_error = self.revision_engine.prediction_error(
            log.records,
            matches[0].concept.parameter_value,
        )
        revision = self.revision_engine.revise(
            self.graph,
            matches[0].concept,
            property_node_id=transfer.property_node_id,
            log=log,
            environment_name=environment_name,
        )
        if revision is None:
            raise RuntimeError("Transferred concept did not require revision")
        self.library.add(revision.revised_concept)
        post_error = self.revision_engine.prediction_error(
            log.records,
            revision.revised_concept.parameter_value,
        )
        return GeneralistBenchmarkResult(
            learned_concept=matches[0].concept,
            transferred_node_id=transfer.property_node_id,
            revised_concept=revision.revised_concept,
            revision=revision.revision,
            transfer_audit=transfer.audit,
            revision_audit=revision.audit,
            pre_revision_error=pre_error,
            post_revision_error=post_error,
            graph_nodes=len(self.graph.nodes),
            graph_edges=len(self.graph.edges),
        )


def run_generalist_benchmark(
    *,
    source_friction: float = 0.30,
    target_friction: float = 0.05,
) -> GeneralistBenchmarkResult:
    system = GeneralistSystem()
    system.learn_friction_concept(coefficient=source_friction, surface_id="wood")
    target_records = tuple(
        ExperimentRecord(
            object_id=record.object_id,
            initial_velocity=record.initial_velocity,
            final_velocity=record.final_velocity,
            duration=record.duration,
            predicted_final_velocity=record.predicted_final_velocity,
            surface_id="ice",
        )
        for record in synthetic_sliding_records(
            object_id="block_ice",
            friction=target_friction,
            count=8,
        )
    )
    return system.transfer_and_revise(
        target_object_id="block_ice",
        environment_name="ice",
        log=ExperimentLog(target_records),
    )


def _signature_overlap(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    matched = 0
    relevant = 0
    for key in keys:
        if key == "surface_id":
            continue
        relevant += 1
        if left.get(key) == right.get(key):
            matched += 1
    return matched / max(relevant, 1)


def _ensure_object(graph: CausalGraph, object_id: str):
    try:
        return graph.find_node_by_name(object_id)
    except KeyError:
        node = graph.add_node(
            object_id,
            NodeType.OBJECT,
            GaussianBelief(1.0, 1e-6),
            modality="transferred_identity",
            metadata={"object_id": object_id},
        )
        observation = graph.add_node(
            f"transfer_observation.{object_id}",
            NodeType.OBSERVATION,
            GaussianBelief(1.0, 1e-6),
            evidence=GaussianBelief(1.0, 1e-6),
            modality="environment_signature",
        )
        graph.add_edge(
            observation.id,
            node.id,
            EdgeType.OBSERVES,
            noise_variance=1e-6,
        )
        return node


def _ensure_friction_law(graph: CausalGraph):
    try:
        return graph.find_node_by_name("law.friction_force")
    except KeyError:
        return graph.add_node(
            "law.friction_force",
            NodeType.LAW,
            GaussianBelief(1.0, 1e-9),
            is_axiom=True,
            metadata={"equation": "a = -mu * g * sign(v)"},
        )
