from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


MIN_VARIANCE = 1e-9


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class NodeType(StrEnum):
    OBJECT = "object"
    PROPERTY = "property"
    STATE = "state"
    COLLISION = "collision"
    CONTACT_RELATION = "contact_relation"
    RELATION = "relation"
    LAW = "law"
    ACTION = "action"
    OBSERVATION = "observation"
    GOAL = "goal"
    CONCEPT = "concept"
    EXPERIMENT = "experiment"
    POLICY = "policy"


class EdgeType(StrEnum):
    CAUSES = "causes"
    ENABLES = "enables"
    INHIBITS = "inhibits"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    OBSERVES = "observes"
    PREDICTS = "predicts"
    HAS_STATE = "has_state"
    EVOLVES_TO = "evolves_to"
    COLLIDES = "collides"
    AFFECTS = "affects"
    TOUCHES = "touches"


class GroundingStatus(StrEnum):
    CONFIDENT = "confident"
    SPECULATIVE = "speculative"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True, slots=True)
class GaussianBelief:
    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("Belief mean must be finite")
        if not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("Belief variance must be finite and positive")

    @property
    def precision(self) -> float:
        return 1.0 / max(self.variance, MIN_VARIANCE)

    @property
    def confidence(self) -> float:
        return 1.0 / (1.0 + math.sqrt(self.variance))

    @classmethod
    def fuse(cls, beliefs: list[GaussianBelief]) -> GaussianBelief:
        if not beliefs:
            raise ValueError("At least one belief is required")
        precision = sum(item.precision for item in beliefs)
        weighted_mean = sum(item.mean * item.precision for item in beliefs)
        return cls(weighted_mean / precision, max(1.0 / precision, MIN_VARIANCE))

    @classmethod
    def from_samples(
        cls,
        samples: list[float],
        weights: list[float] | None = None,
    ) -> GaussianBelief:
        if not samples:
            raise ValueError("At least one sample is required")
        if weights is None:
            weights = [1.0 / len(samples)] * len(samples)
        if len(samples) != len(weights):
            raise ValueError("Samples and weights must have equal length")
        total_weight = sum(weights)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Particle weights must have positive finite mass")
        normalized = [weight / total_weight for weight in weights]
        mean = sum(
            sample * weight
            for sample, weight in zip(samples, normalized, strict=True)
        )
        variance = sum(
            weight * (sample - mean) ** 2
            for sample, weight in zip(samples, normalized, strict=True)
        )
        return cls(mean, max(variance, MIN_VARIANCE))

    def kl_divergence(self, prior: GaussianBelief) -> float:
        ratio = self.variance / prior.variance
        mean_delta = self.mean - prior.mean
        return 0.5 * (
            ratio
            + (mean_delta * mean_delta) / prior.variance
            - 1.0
            - math.log(ratio)
        )


@dataclass(frozen=True, slots=True)
class VectorGaussianBelief:
    means: tuple[float, ...]
    variances: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.means or len(self.means) != len(self.variances):
            raise ValueError("Vector belief dimensions must be non-empty and equal")
        if not all(math.isfinite(item) for item in self.means):
            raise ValueError("Vector belief means must be finite")
        if not all(math.isfinite(item) and item > 0 for item in self.variances):
            raise ValueError("Vector belief variances must be finite and positive")

    @classmethod
    def from_samples(
        cls,
        samples: list[tuple[float, ...]],
    ) -> VectorGaussianBelief:
        if not samples:
            raise ValueError("At least one vector sample is required")
        dimensions = len(samples[0])
        if dimensions == 0 or any(len(sample) != dimensions for sample in samples):
            raise ValueError("Vector samples must have equal non-zero dimensions")
        columns = [
            [sample[index] for sample in samples] for index in range(dimensions)
        ]
        beliefs = [GaussianBelief.from_samples(column) for column in columns]
        return cls(
            means=tuple(item.mean for item in beliefs),
            variances=tuple(item.variance for item in beliefs),
        )

    def component(self, index: int) -> GaussianBelief:
        return GaussianBelief(self.means[index], self.variances[index])


@dataclass(slots=True)
class Node:
    id: str
    name: str
    node_type: NodeType
    prior: GaussianBelief
    belief: GaussianBelief
    modality: str = "logical"
    is_axiom: bool = False
    evidence: GaussianBelief | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    deprecated_at: datetime | None = None
    superseded_by: str | None = None

    @classmethod
    def create(
        cls,
        name: str,
        node_type: NodeType,
        prior: GaussianBelief,
        *,
        node_id: str | None = None,
        modality: str = "logical",
        is_axiom: bool = False,
        evidence: GaussianBelief | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        return cls(
            id=node_id or str(uuid4()),
            name=name,
            node_type=node_type,
            prior=prior,
            belief=GaussianBelief.fuse([prior, evidence]) if evidence else prior,
            modality=modality,
            is_axiom=is_axiom,
            evidence=evidence,
            metadata=dict(metadata or {}),
        )

    @property
    def is_grounding_root(self) -> bool:
        return bool(self.metadata.get("intervened")) or self.is_axiom or (
            self.node_type is NodeType.OBSERVATION and self.evidence is not None
        )


@dataclass(slots=True)
class Edge:
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    bias: float = 0.0
    noise_variance: float = 1.0
    confidence: float = 1.0
    learned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        *,
        weight: float = 1.0,
        bias: float = 0.0,
        noise_variance: float = 1.0,
        confidence: float = 1.0,
        learned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        if source_id == target_id:
            raise ValueError("Self-edges are not supported")
        if not math.isfinite(weight) or weight == 0:
            raise ValueError("Edge weight must be finite and non-zero")
        if not math.isfinite(bias):
            raise ValueError("Edge bias must be finite")
        if not math.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("Edge noise variance must be finite and positive")
        if not 0 <= confidence <= 1:
            raise ValueError("Edge confidence must be in [0, 1]")
        return cls(
            id=str(uuid4()),
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            bias=bias,
            noise_variance=noise_variance,
            confidence=confidence,
            learned=learned,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class InferenceResult:
    converged: bool
    iterations: int
    max_residual: float
    free_energy: float
    changed_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundingAudit:
    node_id: str
    status: GroundingStatus
    confidence: float
    root_ids: tuple[str, ...]
    trace_edge_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructureProposal:
    source_id: str
    target_id: str
    weight: float
    bias: float
    noise_variance: float
    evidence_gain: float
    complexity_penalty: float

    @property
    def score(self) -> float:
        return self.evidence_gain - self.complexity_penalty


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    force: float
    duration: float


@dataclass(frozen=True, slots=True)
class ActionScore:
    action: ActionCandidate
    expected_free_energy: float
    epistemic_cost: float
    pragmatic_cost: float


@dataclass(frozen=True, slots=True)
class ObjectAction:
    object_id: str
    force: float
    duration: float

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("Action object ID is required")
        if not math.isfinite(self.force):
            raise ValueError("Action force must be finite")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Action duration must be finite and positive")


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    query_variable: str
    belief: GaussianBelief
    actual_belief: GaussianBelief
    collision_probability: float
    actual_collision_probability: float
    grounding: GroundingAudit
    intervention_node_ids: tuple[str, ...]
    shared_exogenous_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContactRelationProposal:
    object_a: str
    object_b: str
    probability: float
    evidence_gain: float
    complexity_penalty: float
    observations: int

    @property
    def score(self) -> float:
        return self.evidence_gain - self.complexity_penalty
