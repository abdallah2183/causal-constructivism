from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, GroundingAudit, NodeType


GRAVITY = 9.81


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    object_id: str
    initial_velocity: float
    final_velocity: float
    duration: float
    predicted_final_velocity: float
    surface_id: str = "default_surface"

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("Object ID is required")
        for name, value in (
            ("initial_velocity", self.initial_velocity),
            ("final_velocity", self.final_velocity),
            ("duration", self.duration),
            ("predicted_final_velocity", self.predicted_final_velocity),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")

    @property
    def prediction_error(self) -> float:
        return self.final_velocity - self.predicted_final_velocity

    @property
    def observed_acceleration(self) -> float:
        return (self.final_velocity - self.initial_velocity) / self.duration


class ExperimentLog:
    def __init__(self, records: Iterable[ExperimentRecord] = ()) -> None:
        self._records = list(records)

    @property
    def records(self) -> tuple[ExperimentRecord, ...]:
        return tuple(self._records)

    def append(self, record: ExperimentRecord) -> None:
        self._records.append(record)

    def extend(self, records: Iterable[ExperimentRecord]) -> None:
        for record in records:
            self.append(record)

    def for_object(self, object_id: str) -> tuple[ExperimentRecord, ...]:
        return tuple(record for record in self._records if record.object_id == object_id)

    def save_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> ExperimentLog:
        records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(ExperimentRecord(**json.loads(line)))
        return cls(records)


@dataclass(frozen=True, slots=True)
class Anomaly:
    object_id: str
    records: tuple[ExperimentRecord, ...]
    features: tuple[float, ...]
    local_free_energy: float
    mean_prediction_error: float


class AnomalyDetector:
    """Finds persistent failures of the constant-velocity dynamics model."""

    def __init__(
        self,
        *,
        minimum_records: int = 4,
        error_threshold: float = 0.05,
    ) -> None:
        self.minimum_records = minimum_records
        self.error_threshold = error_threshold

    def detect(self, log: ExperimentLog, object_id: str) -> Anomaly | None:
        records = log.for_object(object_id)
        if len(records) < self.minimum_records:
            return None
        features = friction_features(records)
        mean_abs_error = features[0]
        mean_deceleration = features[1]
        stop_fraction = features[4]
        if (
            mean_abs_error < self.error_threshold
            or mean_deceleration <= 0
            or stop_fraction < 0.25
        ):
            return None
        local_free_energy = sum(
            record.prediction_error * record.prediction_error
            for record in records
        )
        mean_error = sum(record.prediction_error for record in records) / len(records)
        return Anomaly(
            object_id=object_id,
            records=records,
            features=features,
            local_free_energy=local_free_energy,
            mean_prediction_error=mean_error,
        )


def friction_features(records: Iterable[ExperimentRecord]) -> tuple[float, ...]:
    rows = tuple(records)
    if not rows:
        raise ValueError("Feature extraction requires at least one record")
    abs_errors = [abs(record.prediction_error) for record in rows]
    decelerations = [
        max(0.0, -math.copysign(1.0, record.initial_velocity or 1.0) * record.observed_acceleration)
        for record in rows
    ]
    initial_speeds = [abs(record.initial_velocity) for record in rows]
    final_speeds = [abs(record.final_velocity) for record in rows]
    durations = [record.duration for record in rows]
    stop_fraction = sum(
        1 for initial, final in zip(initial_speeds, final_speeds, strict=True)
        if final < 0.75 * initial
    ) / len(rows)
    return (
        sum(abs_errors) / len(abs_errors),
        sum(decelerations) / len(decelerations),
        max(initial_speeds) - min(initial_speeds),
        sum(durations) / len(durations),
        stop_fraction,
    )


@dataclass(frozen=True, slots=True)
class StructureCandidate:
    concept: str
    target_object_id: str
    confidence: float
    parameters: dict[str, float]


@dataclass(frozen=True, slots=True)
class ScoredProposal:
    candidate: StructureCandidate
    evidence_gain: float
    complexity_penalty: float
    neural_bonus: float

    @property
    def score(self) -> float:
        return self.evidence_gain + self.neural_bonus - self.complexity_penalty


class NeuralStructureProposer:
    """Small trainable neural classifier for constrained graph proposals.

    This is deliberately not an open-ended ontology generator. It is the first
    neural proposal contract: a hidden-layer network maps anomaly features to a
    grammar-valid proposal type.
    """

    def __init__(
        self,
        *,
        input_size: int = 5,
        hidden_size: int = 6,
        seed: int = 13,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = random.Random(seed)
        self.w1 = [
            [rng.uniform(-0.25, 0.25) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [rng.uniform(-0.25, 0.25) for _ in range(hidden_size)]
        self.b2 = 0.0
        self.feature_means = [0.0 for _ in range(input_size)]
        self.feature_scales = [1.0 for _ in range(input_size)]
        self.trained = False

    def train(
        self,
        examples: list[tuple[tuple[float, ...], int]],
        *,
        epochs: int = 700,
        learning_rate: float = 0.08,
    ) -> None:
        if not examples:
            raise ValueError("Training requires examples")
        self._fit_normalizer([features for features, _ in examples])
        for _ in range(epochs):
            for raw_features, label in examples:
                features = self._normalize(raw_features)
                hidden_raw = [
                    sum(weight * value for weight, value in zip(row, features, strict=True))
                    + bias
                    for row, bias in zip(self.w1, self.b1, strict=True)
                ]
                hidden = [_tanh(value) for value in hidden_raw]
                logit = (
                    sum(weight * value for weight, value in zip(self.w2, hidden, strict=True))
                    + self.b2
                )
                probability = _sigmoid(logit)
                output_error = probability - label
                old_w2 = list(self.w2)
                for index in range(self.hidden_size):
                    self.w2[index] -= learning_rate * output_error * hidden[index]
                self.b2 -= learning_rate * output_error
                for hidden_index in range(self.hidden_size):
                    hidden_grad = (
                        output_error
                        * old_w2[hidden_index]
                        * (1 - hidden[hidden_index] * hidden[hidden_index])
                    )
                    for feature_index in range(self.input_size):
                        self.w1[hidden_index][feature_index] -= (
                            learning_rate * hidden_grad * features[feature_index]
                        )
                    self.b1[hidden_index] -= learning_rate * hidden_grad
        self.trained = True

    def probability(self, features: tuple[float, ...]) -> float:
        if len(features) != self.input_size:
            raise ValueError("Unexpected feature dimension")
        normalized = self._normalize(features)
        hidden = [
            _tanh(sum(weight * value for weight, value in zip(row, normalized, strict=True)) + bias)
            for row, bias in zip(self.w1, self.b1, strict=True)
        ]
        return _sigmoid(
            sum(weight * value for weight, value in zip(self.w2, hidden, strict=True))
            + self.b2
        )

    def propose(self, anomaly: Anomaly) -> StructureCandidate | None:
        if not self.trained:
            raise RuntimeError("NeuralStructureProposer must be trained first")
        confidence = self.probability(anomaly.features)
        if confidence < 0.65:
            return None
        friction = fit_friction_coefficient(anomaly.records)
        return StructureCandidate(
            concept="friction",
            target_object_id=anomaly.object_id,
            confidence=confidence,
            parameters={"coefficient": friction},
        )

    def _fit_normalizer(self, rows: list[tuple[float, ...]]) -> None:
        for index in range(self.input_size):
            values = [row[index] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.feature_means[index] = mean
            self.feature_scales[index] = max(math.sqrt(variance), 1e-6)

    def _normalize(self, features: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                features,
                self.feature_means,
                self.feature_scales,
                strict=True,
            )
        )


class ProposalScorer:
    def __init__(
        self,
        *,
        complexity_penalty: float = 1.5,
        neural_weight: float = 1.0,
    ) -> None:
        self.complexity_penalty = complexity_penalty
        self.neural_weight = neural_weight

    def score(
        self,
        anomaly: Anomaly,
        candidate: StructureCandidate,
    ) -> ScoredProposal:
        if candidate.concept != "friction":
            raise ValueError(f"Unsupported concept: {candidate.concept}")
        baseline_sse = sum(
            record.prediction_error * record.prediction_error
            for record in anomaly.records
        )
        coefficient = candidate.parameters["coefficient"]
        residual_sse = sum(
            (
                record.final_velocity
                - predict_with_friction(record, coefficient)
            )
            ** 2
            for record in anomaly.records
        )
        evidence_gain = 0.5 * len(anomaly.records) * math.log(
            max(baseline_sse, 1e-12) / max(residual_sse, 1e-12)
        )
        return ScoredProposal(
            candidate=candidate,
            evidence_gain=evidence_gain,
            complexity_penalty=self.complexity_penalty * math.log(len(anomaly.records) + 1),
            neural_bonus=self.neural_weight * math.log(max(candidate.confidence, 1e-9) / 0.5),
        )


class DiscoveryIntegrator:
    def __init__(self, graph: CausalGraph) -> None:
        self.graph = graph

    def integrate(
        self,
        proposal: ScoredProposal,
        anomaly: Anomaly,
        *,
        threshold: float = 1.0,
    ) -> tuple[str, GroundingAudit]:
        if proposal.score <= threshold:
            raise ValueError("Proposal did not clear the integration threshold")
        candidate = proposal.candidate
        object_node = self._ensure_object(candidate.target_object_id, anomaly)
        law_node = self._ensure_friction_law()
        coefficient = candidate.parameters["coefficient"]
        property_node = self.graph.add_node(
            f"{candidate.target_object_id}.friction",
            NodeType.PROPERTY,
            GaussianBelief(coefficient, max(0.01 * coefficient * coefficient, 1e-6)),
            modality="discovered_physical",
            metadata={
                "discovered": True,
                "concept": "friction",
                "proposal_score": proposal.score,
                "neural_confidence": candidate.confidence,
                "evidence_gain": proposal.evidence_gain,
            },
        )
        observation = self.graph.add_node(
            f"discovery_observation.{candidate.target_object_id}.friction",
            NodeType.OBSERVATION,
            property_node.belief,
            evidence=property_node.belief,
            modality="experiment_log",
            metadata={
                "records": len(anomaly.records),
                "local_free_energy": anomaly.local_free_energy,
            },
        )
        self.graph.add_edge(
            object_node.id,
            property_node.id,
            EdgeType.PART_OF,
            noise_variance=property_node.belief.variance,
            learned=True,
        )
        self.graph.add_edge(
            law_node.id,
            property_node.id,
            EdgeType.CAUSES,
            noise_variance=property_node.belief.variance,
            learned=True,
        )
        self.graph.add_edge(
            observation.id,
            property_node.id,
            EdgeType.OBSERVES,
            noise_variance=property_node.belief.variance,
            confidence=candidate.confidence,
            learned=True,
        )
        return property_node.id, MetacognitiveAuditor(self.graph).audit(property_node.id)

    def _ensure_object(self, object_id: str, anomaly: Anomaly):
        try:
            return self.graph.find_node_by_name(object_id)
        except KeyError:
            node = self.graph.add_node(
                object_id,
                NodeType.OBJECT,
                GaussianBelief(1.0, 1e-6),
                modality="discovery_identity",
                metadata={"object_id": object_id},
            )
            evidence = GaussianBelief(abs(anomaly.records[0].initial_velocity), 1e-6)
            observation = self.graph.add_node(
                f"discovery_observation.{object_id}",
                NodeType.OBSERVATION,
                evidence,
                evidence=evidence,
                modality="experiment_log",
            )
            self.graph.add_edge(
                observation.id,
                node.id,
                EdgeType.OBSERVES,
                noise_variance=1e-6,
            )
            return node

    def _ensure_friction_law(self):
        try:
            return self.graph.find_node_by_name("law.friction_force")
        except KeyError:
            return self.graph.add_node(
                "law.friction_force",
                NodeType.LAW,
                GaussianBelief(1.0, 1e-9),
                is_axiom=True,
                metadata={"equation": "a = -mu * g * sign(v)"},
            )


class DiscoverySystem:
    def __init__(self, *, graph: CausalGraph | None = None) -> None:
        self.graph = graph or CausalGraph()
        self.detector = AnomalyDetector()
        self.proposer = NeuralStructureProposer()
        self.scorer = ProposalScorer()
        self.integrator = DiscoveryIntegrator(self.graph)

    def train_default_curriculum(self) -> None:
        self.proposer.train(synthetic_discovery_curriculum())

    def discover_friction(
        self,
        log: ExperimentLog,
        object_id: str,
    ) -> tuple[ScoredProposal, str, GroundingAudit] | None:
        anomaly = self.detector.detect(log, object_id)
        if anomaly is None:
            return None
        if not self.proposer.trained:
            self.train_default_curriculum()
        candidate = self.proposer.propose(anomaly)
        if candidate is None:
            return None
        scored = self.scorer.score(anomaly, candidate)
        if scored.score <= 1.0:
            return None
        node_id, audit = self.integrator.integrate(scored, anomaly)
        return scored, node_id, audit


def fit_friction_coefficient(records: Iterable[ExperimentRecord]) -> float:
    coefficients = []
    for record in records:
        if abs(record.initial_velocity) < 1e-9:
            continue
        deceleration = -math.copysign(1.0, record.initial_velocity) * record.observed_acceleration
        if deceleration > 0:
            coefficients.append(deceleration / GRAVITY)
    if not coefficients:
        return 0.0
    return max(0.0, sum(coefficients) / len(coefficients))


def predict_with_friction(record: ExperimentRecord, coefficient: float) -> float:
    speed = abs(record.initial_velocity)
    reduced = max(0.0, speed - coefficient * GRAVITY * record.duration)
    return math.copysign(reduced, record.initial_velocity)


def synthetic_sliding_records(
    *,
    object_id: str = "block_001",
    friction: float = 0.25,
    count: int = 8,
) -> tuple[ExperimentRecord, ...]:
    records = []
    for index in range(count):
        initial = 0.8 + 0.12 * index
        duration = 0.18 + 0.01 * (index % 3)
        final = max(0.0, initial - friction * GRAVITY * duration)
        records.append(
            ExperimentRecord(
                object_id=object_id,
                initial_velocity=initial,
                final_velocity=final,
                duration=duration,
                predicted_final_velocity=initial,
            )
        )
    return tuple(records)


def synthetic_constant_velocity_records(
    *,
    object_id: str = "block_001",
    count: int = 8,
) -> tuple[ExperimentRecord, ...]:
    return tuple(
        ExperimentRecord(
            object_id=object_id,
            initial_velocity=1.0 + 0.05 * index,
            final_velocity=1.0 + 0.05 * index,
            duration=0.2,
            predicted_final_velocity=1.0 + 0.05 * index,
        )
        for index in range(count)
    )


def synthetic_discovery_curriculum() -> list[tuple[tuple[float, ...], int]]:
    examples: list[tuple[tuple[float, ...], int]] = []
    for friction in (0.08, 0.12, 0.18, 0.25, 0.35, 0.5):
        examples.append((friction_features(synthetic_sliding_records(friction=friction)), 1))
    for scale in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        records = tuple(
            ExperimentRecord(
                object_id="block_001",
                initial_velocity=1.0 + 0.05 * index,
                final_velocity=1.0 + 0.05 * index - scale,
                duration=0.2,
                predicted_final_velocity=1.0 + 0.05 * index,
            )
            for index in range(8)
        )
        examples.append((friction_features(records), 0))
    return examples


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _tanh(value: float) -> float:
    return math.tanh(value)
