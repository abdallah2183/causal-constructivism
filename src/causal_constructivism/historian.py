from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .audit import MetacognitiveAuditor
from .discovery import (
    GRAVITY,
    AnomalyDetector,
    ExperimentLog,
    ExperimentRecord,
    fit_friction_coefficient,
    predict_with_friction,
)
from .generalist import Concept, ConceptLibrary
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, GroundingAudit, GroundingStatus, NodeType


@dataclass(frozen=True, slots=True)
class SharedExogenousTrial:
    trial_id: str
    initial_velocity: float
    duration: float
    true_mass: float
    hidden_friction: float

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("Trial ID is required")
        for name, value in (
            ("initial_velocity", self.initial_velocity),
            ("duration", self.duration),
            ("true_mass", self.true_mass),
            ("hidden_friction", self.hidden_friction),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.true_mass <= 0:
            raise ValueError("True mass must be positive")
        if self.hidden_friction < 0:
            raise ValueError("Hidden friction must be non-negative")

    @property
    def final_velocity(self) -> float:
        speed = max(0.0, abs(self.initial_velocity) - self.hidden_friction * GRAVITY * self.duration)
        return math.copysign(speed, self.initial_velocity)


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    action_force: float = 40.0
    action_duration: float = 0.2
    epistemic_weight: float = 1.0
    anomaly_threshold: float = 0.05
    discovery_min_records: int = 4
    proposal_confidence_threshold: float = 0.65
    evidence_gain_threshold: float = 1.0
    transfer_match_threshold: float = 0.8
    revision_contradiction_threshold: float = 0.05
    revision_evidence_threshold: float = 1.0
    initial_friction_coefficient: float | None = None
    use_concepts_for_mass_inference: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Policy name is required")
        for name, value in (
            ("action_force", self.action_force),
            ("action_duration", self.action_duration),
            ("epistemic_weight", self.epistemic_weight),
            ("anomaly_threshold", self.anomaly_threshold),
            ("proposal_confidence_threshold", self.proposal_confidence_threshold),
            ("evidence_gain_threshold", self.evidence_gain_threshold),
            ("transfer_match_threshold", self.transfer_match_threshold),
            ("revision_contradiction_threshold", self.revision_contradiction_threshold),
            ("revision_evidence_threshold", self.revision_evidence_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.action_force <= 0:
            raise ValueError("Action force must be positive")
        if self.action_duration <= 0:
            raise ValueError("Action duration must be positive")
        if self.discovery_min_records <= 0:
            raise ValueError("Discovery minimum must be positive")
        if self.initial_friction_coefficient is not None and self.initial_friction_coefficient < 0:
            raise ValueError("Initial friction coefficient must be non-negative")


@dataclass(frozen=True, slots=True)
class PolicyIntervention:
    name: str
    discover_friction_before_mass: bool = False
    action_force_multiplier: float = 1.0
    discovery_min_records: int | None = None

    def apply(self, policy: Policy, *, hidden_friction: float) -> Policy:
        if self.action_force_multiplier <= 0 or not math.isfinite(self.action_force_multiplier):
            raise ValueError("Action force multiplier must be positive and finite")
        return Policy(
            name=f"{policy.name}.{self.name}",
            action_force=policy.action_force * self.action_force_multiplier,
            action_duration=policy.action_duration,
            epistemic_weight=policy.epistemic_weight,
            anomaly_threshold=policy.anomaly_threshold,
            discovery_min_records=self.discovery_min_records or policy.discovery_min_records,
            proposal_confidence_threshold=policy.proposal_confidence_threshold,
            evidence_gain_threshold=policy.evidence_gain_threshold,
            transfer_match_threshold=policy.transfer_match_threshold,
            revision_contradiction_threshold=policy.revision_contradiction_threshold,
            revision_evidence_threshold=policy.revision_evidence_threshold,
            initial_friction_coefficient=(
                hidden_friction
                if self.discover_friction_before_mass
                else policy.initial_friction_coefficient
            ),
            use_concepts_for_mass_inference=policy.use_concepts_for_mass_inference,
        )


@dataclass(frozen=True, slots=True)
class HistoricalExperiment:
    experiment_id: str
    index: int
    policy_name: str
    trial_id: str
    record: ExperimentRecord
    action_force: float
    mass_estimate: float
    mass_error: float
    friction_known_before_mass: bool
    discovered_concepts: tuple[str, ...]
    experiment_node_id: str
    mass_node_id: str
    library_snapshot_node_id: str


@dataclass(frozen=True, slots=True)
class HistoryMetrics:
    mean_mass_error: float
    final_mass_error: float
    mean_prediction_error: float
    experiments_to_friction: int | None
    confident_fraction: float
    ungrounded_nodes: int
    final_mass_grounding: GroundingAudit


@dataclass(frozen=True, slots=True)
class HistoryReplay:
    label: str
    policy: Policy
    experiments: tuple[HistoricalExperiment, ...]
    concept_library: ConceptLibrary
    graph: CausalGraph
    metrics: HistoryMetrics


@dataclass(frozen=True, slots=True)
class HistoryCounterfactualResult:
    actual: HistoryReplay
    counterfactual: HistoryReplay
    intervention: PolicyIntervention
    shared_exogenous_ids: tuple[str, ...]

    @property
    def mean_mass_error_reduction(self) -> float:
        return (
            self.actual.metrics.mean_mass_error
            - self.counterfactual.metrics.mean_mass_error
        )

    @property
    def prediction_error_reduction(self) -> float:
        return (
            self.actual.metrics.mean_prediction_error
            - self.counterfactual.metrics.mean_prediction_error
        )

    @property
    def experiments_saved_to_friction(self) -> int:
        actual = self.actual.metrics.experiments_to_friction
        counterfactual = self.counterfactual.metrics.experiments_to_friction
        if actual is None:
            actual = len(self.actual.experiments) + 1
        if counterfactual is None:
            counterfactual = len(self.counterfactual.experiments) + 1
        return actual - counterfactual


class HistoryLog:
    def __init__(self, trials: Iterable[SharedExogenousTrial]) -> None:
        self._trials = tuple(trials)
        if not self._trials:
            raise ValueError("History requires at least one trial")

    @property
    def trials(self) -> tuple[SharedExogenousTrial, ...]:
        return self._trials

    def save_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for trial in self._trials:
                handle.write(json.dumps(asdict(trial), sort_keys=True) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> HistoryLog:
        trials = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trials.append(SharedExogenousTrial(**json.loads(line)))
        return cls(trials)


class HistorianSystem:
    def replay(
        self,
        trials: Iterable[SharedExogenousTrial],
        policy: Policy,
        *,
        label: str = "actual",
    ) -> HistoryReplay:
        shared_trials = tuple(trials)
        if not shared_trials:
            raise ValueError("Replay requires at least one trial")
        graph = CausalGraph()
        library = ConceptLibrary()
        policy_node = _add_policy_node(graph, policy, label)
        previous_experiment_node_id: str | None = None
        previous_library_node_id = _add_initial_library_snapshot(graph, label)
        known_friction = policy.initial_friction_coefficient
        if known_friction is not None:
            library.add(_friction_concept(known_friction, confidence=0.99))
        experiments: list[HistoricalExperiment] = []
        records: list[ExperimentRecord] = []
        experiments_to_friction = 0 if known_friction is not None else None
        detector = AnomalyDetector(
            minimum_records=policy.discovery_min_records,
            error_threshold=policy.anomaly_threshold,
        )

        for index, trial in enumerate(shared_trials, start=1):
            friction_known_before_mass = known_friction is not None
            predicted_final = (
                predict_with_friction(
                    _record_for_trial(trial, predicted_final_velocity=trial.initial_velocity),
                    known_friction,
                )
                if known_friction is not None
                else trial.initial_velocity
            )
            record = _record_for_trial(trial, predicted_final_velocity=predicted_final)
            records.append(record)
            mass_estimate = _infer_mass(
                trial,
                force=policy.action_force,
                known_friction=(
                    known_friction
                    if policy.use_concepts_for_mass_inference
                    else None
                ),
            )
            mass_error = abs(mass_estimate - trial.true_mass) / trial.true_mass
            experiment_node = _add_experiment_node(
                graph,
                label=label,
                index=index,
                policy_node_id=policy_node.id,
                trial=trial,
                previous_experiment_node_id=previous_experiment_node_id,
            )
            mass_node = graph.add_node(
                f"{label}.mass_estimate.{index}",
                NodeType.PROPERTY,
                GaussianBelief(mass_estimate, 1e-4),
                modality="history_replay",
                metadata={
                    "true_mass": trial.true_mass,
                    "relative_error": mass_error,
                    "known_friction_used": known_friction is not None,
                },
            )
            graph.add_edge(
                experiment_node.id,
                mass_node.id,
                EdgeType.CAUSES,
                noise_variance=1e-4,
            )
            discovered = []
            anomaly = detector.detect(ExperimentLog(records), "block_001")
            if known_friction is None and anomaly is not None:
                known_friction = fit_friction_coefficient(records)
                concept = _friction_concept(known_friction, confidence=0.95)
                library.add(concept)
                experiments_to_friction = index
                discovered.append("friction")
                _add_discovered_concept_node(
                    graph,
                    label=label,
                    experiment_node_id=experiment_node.id,
                    concept=concept,
                    records=len(records),
                )
            library_snapshot = _add_library_snapshot(
                graph,
                label=label,
                index=index,
                concept_count=len(library.concepts),
                previous_library_node_id=previous_library_node_id,
                experiment_node_id=experiment_node.id,
            )
            experiments.append(
                HistoricalExperiment(
                    experiment_id=f"{label}.experiment.{index}",
                    index=index,
                    policy_name=policy.name,
                    trial_id=trial.trial_id,
                    record=record,
                    action_force=policy.action_force,
                    mass_estimate=mass_estimate,
                    mass_error=mass_error,
                    friction_known_before_mass=friction_known_before_mass,
                    discovered_concepts=tuple(discovered),
                    experiment_node_id=experiment_node.id,
                    mass_node_id=mass_node.id,
                    library_snapshot_node_id=library_snapshot.id,
                )
            )
            previous_experiment_node_id = experiment_node.id
            previous_library_node_id = library_snapshot.id

        metrics = _history_metrics(
            graph,
            tuple(experiments),
            experiments_to_friction=experiments_to_friction,
        )
        return HistoryReplay(
            label=label,
            policy=policy,
            experiments=tuple(experiments),
            concept_library=library,
            graph=graph,
            metrics=metrics,
        )

    def counterfactual(
        self,
        trials: Iterable[SharedExogenousTrial],
        policy: Policy,
        intervention: PolicyIntervention,
    ) -> HistoryCounterfactualResult:
        shared_trials = tuple(trials)
        if not shared_trials:
            raise ValueError("Counterfactual requires at least one trial")
        actual = self.replay(shared_trials, policy, label="actual")
        counterfactual_policy = intervention.apply(
            policy,
            hidden_friction=shared_trials[0].hidden_friction,
        )
        counterfactual = self.replay(
            shared_trials,
            counterfactual_policy,
            label="counterfactual",
        )
        return HistoryCounterfactualResult(
            actual=actual,
            counterfactual=counterfactual,
            intervention=intervention,
            shared_exogenous_ids=tuple(trial.trial_id for trial in shared_trials),
        )


def run_historian_benchmark(
    *,
    experiments: int = 6,
    true_mass: float = 2.5,
    hidden_friction: float = 0.25,
) -> HistoryCounterfactualResult:
    trials = synthetic_history_trials(
        count=experiments,
        true_mass=true_mass,
        hidden_friction=hidden_friction,
    )
    policy = Policy(name="baseline")
    intervention = PolicyIntervention(
        name="discover_friction_before_mass",
        discover_friction_before_mass=True,
    )
    return HistorianSystem().counterfactual(trials, policy, intervention)


def synthetic_history_trials(
    *,
    count: int = 6,
    true_mass: float = 2.5,
    hidden_friction: float = 0.25,
) -> tuple[SharedExogenousTrial, ...]:
    if count <= 0:
        raise ValueError("Trial count must be positive")
    return tuple(
        SharedExogenousTrial(
            trial_id=f"trial_{index + 1:03d}",
            initial_velocity=0.8 + 0.12 * index,
            duration=0.18 + 0.01 * (index % 3),
            true_mass=true_mass,
            hidden_friction=hidden_friction,
        )
        for index in range(count)
    )


def _record_for_trial(
    trial: SharedExogenousTrial,
    *,
    predicted_final_velocity: float,
) -> ExperimentRecord:
    return ExperimentRecord(
        object_id="block_001",
        initial_velocity=trial.initial_velocity,
        final_velocity=trial.final_velocity,
        duration=trial.duration,
        predicted_final_velocity=predicted_final_velocity,
        surface_id="history_surface",
    )


def _infer_mass(
    trial: SharedExogenousTrial,
    *,
    force: float,
    known_friction: float | None,
) -> float:
    observed_acceleration = force / trial.true_mass - trial.hidden_friction * GRAVITY
    if observed_acceleration <= 0:
        raise ValueError("Action force is too small to identify mass")
    corrected_acceleration = observed_acceleration
    if known_friction is not None:
        corrected_acceleration += known_friction * GRAVITY
    return force / corrected_acceleration


def _friction_concept(coefficient: float, *, confidence: float) -> Concept:
    return Concept(
        name="friction",
        parameter_name="coefficient",
        parameter_value=coefficient,
        applicability_signature={
            "motion_type": "sliding",
            "surface_interaction": True,
            "deceleration": True,
            "surface_id": "history_surface",
        },
        confidence=confidence,
    )


def _add_policy_node(graph: CausalGraph, policy: Policy, label: str):
    return graph.add_node(
        f"{label}.policy.{policy.name}",
        NodeType.POLICY,
        GaussianBelief(policy.epistemic_weight, 1e-6),
        is_axiom=True,
        modality="policy_design",
        metadata={"policy": asdict(policy)},
    )


def _add_initial_library_snapshot(graph: CausalGraph, label: str) -> str:
    observation = graph.add_node(
        f"{label}.library_snapshot.initial.observation",
        NodeType.OBSERVATION,
        GaussianBelief(0.0, 1e-6),
        evidence=GaussianBelief(0.0, 1e-6),
        modality="concept_library_snapshot",
    )
    snapshot = graph.add_node(
        f"{label}.library_snapshot.initial",
        NodeType.CONCEPT,
        GaussianBelief(0.0, 1e-6),
        modality="concept_library_snapshot",
        metadata={"concept_count": 0},
    )
    graph.add_edge(
        observation.id,
        snapshot.id,
        EdgeType.OBSERVES,
        noise_variance=1e-6,
    )
    return snapshot.id


def _add_experiment_node(
    graph: CausalGraph,
    *,
    label: str,
    index: int,
    policy_node_id: str,
    trial: SharedExogenousTrial,
    previous_experiment_node_id: str | None,
):
    observation = graph.add_node(
        f"{label}.experiment.{index}.exogenous",
        NodeType.OBSERVATION,
        GaussianBelief(trial.initial_velocity, 1e-6),
        evidence=GaussianBelief(trial.initial_velocity, 1e-6),
        modality="shared_exogenous",
        metadata={
            "trial_id": trial.trial_id,
            "duration": trial.duration,
            "hidden_friction": trial.hidden_friction,
        },
    )
    node = graph.add_node(
        f"{label}.experiment.{index}",
        NodeType.EXPERIMENT,
        GaussianBelief(float(index), 1e-6),
        modality="history_replay",
        metadata={"trial_id": trial.trial_id},
    )
    graph.add_edge(
        policy_node_id,
        node.id,
        EdgeType.CAUSES,
        noise_variance=1e-6,
    )
    graph.add_edge(
        observation.id,
        node.id,
        EdgeType.OBSERVES,
        noise_variance=1e-6,
    )
    if previous_experiment_node_id is not None:
        graph.add_edge(
            previous_experiment_node_id,
            node.id,
            EdgeType.EVOLVES_TO,
            noise_variance=1e-6,
        )
    return node


def _add_discovered_concept_node(
    graph: CausalGraph,
    *,
    label: str,
    experiment_node_id: str,
    concept: Concept,
    records: int,
) -> None:
    law = _ensure_friction_law(graph)
    observation = graph.add_node(
        f"{label}.concept.{concept.name}.observation",
        NodeType.OBSERVATION,
        GaussianBelief(concept.parameter_value, 1e-6),
        evidence=GaussianBelief(concept.parameter_value, 1e-6),
        modality="history_discovery",
        metadata={"records": records},
    )
    concept_node = graph.add_node(
        f"{label}.concept.{concept.name}",
        NodeType.CONCEPT,
        GaussianBelief(concept.parameter_value, 1e-4),
        modality="history_discovery",
        metadata={"concept": concept.name, "confidence": concept.confidence},
    )
    graph.add_edge(
        observation.id,
        concept_node.id,
        EdgeType.OBSERVES,
        noise_variance=1e-4,
        confidence=concept.confidence,
    )
    graph.add_edge(
        law.id,
        concept_node.id,
        EdgeType.CAUSES,
        noise_variance=1e-4,
    )
    graph.add_edge(
        experiment_node_id,
        concept_node.id,
        EdgeType.CAUSES,
        noise_variance=1e-4,
    )


def _add_library_snapshot(
    graph: CausalGraph,
    *,
    label: str,
    index: int,
    concept_count: int,
    previous_library_node_id: str,
    experiment_node_id: str,
):
    snapshot = graph.add_node(
        f"{label}.library_snapshot.{index}",
        NodeType.CONCEPT,
        GaussianBelief(float(concept_count), 1e-6),
        modality="concept_library_snapshot",
        metadata={"concept_count": concept_count},
    )
    graph.add_edge(
        previous_library_node_id,
        snapshot.id,
        EdgeType.EVOLVES_TO,
        noise_variance=1e-6,
    )
    graph.add_edge(
        experiment_node_id,
        snapshot.id,
        EdgeType.CAUSES,
        noise_variance=1e-6,
    )
    return snapshot


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


def _history_metrics(
    graph: CausalGraph,
    experiments: tuple[HistoricalExperiment, ...],
    *,
    experiments_to_friction: int | None,
) -> HistoryMetrics:
    if not experiments:
        raise ValueError("Metrics require experiments")
    mean_mass_error = sum(item.mass_error for item in experiments) / len(experiments)
    final_mass_error = experiments[-1].mass_error
    mean_prediction_error = sum(
        item.record.prediction_error * item.record.prediction_error
        for item in experiments
    ) / len(experiments)
    auditor = MetacognitiveAuditor(graph)
    audits = [
        auditor.audit(node.id)
        for node in graph.iter_active_nodes()
        if not node.is_grounding_root
    ]
    confident = sum(1 for audit in audits if audit.status is GroundingStatus.CONFIDENT)
    confident_fraction = confident / len(audits) if audits else 1.0
    final_mass_audit = auditor.audit(experiments[-1].mass_node_id)
    return HistoryMetrics(
        mean_mass_error=mean_mass_error,
        final_mass_error=final_mass_error,
        mean_prediction_error=mean_prediction_error,
        experiments_to_friction=experiments_to_friction,
        confident_fraction=confident_fraction,
        ungrounded_nodes=len(graph.validate_grounding()),
        final_mass_grounding=final_mass_audit,
    )
