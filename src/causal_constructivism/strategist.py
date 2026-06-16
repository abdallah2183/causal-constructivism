from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .historian import HistorianSystem, Policy, SharedExogenousTrial, synthetic_history_trials
from .models import EdgeType, GaussianBelief, GroundingAudit, NodeType


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    policy_id: str
    version: int
    epistemic_weight: float = 1.0
    action_force: float = 40.0
    anomaly_threshold: float = 0.05
    discovery_min_records: int = 4
    evidence_gain_threshold: float = 1.0
    complexity_penalty_weight: float = 1.0
    improvement_threshold: float = 0.2
    evaluation_horizon: int = 6

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("Policy ID is required")
        if self.version <= 0:
            raise ValueError("Policy version must be positive")
        for name, value in (
            ("epistemic_weight", self.epistemic_weight),
            ("action_force", self.action_force),
            ("anomaly_threshold", self.anomaly_threshold),
            ("evidence_gain_threshold", self.evidence_gain_threshold),
            ("complexity_penalty_weight", self.complexity_penalty_weight),
            ("improvement_threshold", self.improvement_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.action_force <= 0:
            raise ValueError("Action force must be positive")
        if self.anomaly_threshold < 0:
            raise ValueError("Anomaly threshold must be non-negative")
        if self.discovery_min_records <= 0:
            raise ValueError("Discovery minimum must be positive")
        if self.evaluation_horizon <= 0:
            raise ValueError("Evaluation horizon must be positive")

    def to_historian_policy(self) -> Policy:
        return Policy(
            name=self.policy_id,
            action_force=self.action_force,
            epistemic_weight=self.epistemic_weight,
            anomaly_threshold=self.anomaly_threshold,
            discovery_min_records=self.discovery_min_records,
            evidence_gain_threshold=self.evidence_gain_threshold,
        )


@dataclass(frozen=True, slots=True)
class PolicyScore:
    policy: DiscoveryPolicy
    experiments_to_friction: int | None
    mean_mass_error: float
    confident_fraction: float
    efficiency_gain: float
    accuracy_gain: float
    grounding_gain: float
    composite_score: float


@dataclass(frozen=True, slots=True)
class StrategyResult:
    baseline: PolicyScore
    candidates: tuple[PolicyScore, ...]
    selected: PolicyScore
    adopted: bool
    audit: GroundingAudit
    graph_nodes: int
    graph_edges: int
    ungrounded_nodes: int


class PolicyGenerator:
    def generate(self, base: DiscoveryPolicy) -> tuple[DiscoveryPolicy, ...]:
        version = base.version + 1
        return (
            DiscoveryPolicy(
                policy_id=f"{base.policy_id}.sensitive_anomaly",
                version=version,
                epistemic_weight=base.epistemic_weight,
                action_force=base.action_force,
                anomaly_threshold=max(0.005, base.anomaly_threshold * 0.5),
                discovery_min_records=max(2, base.discovery_min_records - 1),
                evidence_gain_threshold=base.evidence_gain_threshold,
                complexity_penalty_weight=base.complexity_penalty_weight,
                improvement_threshold=base.improvement_threshold,
                evaluation_horizon=base.evaluation_horizon,
            ),
            DiscoveryPolicy(
                policy_id=f"{base.policy_id}.stronger_actions",
                version=version,
                epistemic_weight=base.epistemic_weight * 1.1,
                action_force=base.action_force * 1.5,
                anomaly_threshold=base.anomaly_threshold,
                discovery_min_records=base.discovery_min_records,
                evidence_gain_threshold=base.evidence_gain_threshold,
                complexity_penalty_weight=base.complexity_penalty_weight,
                improvement_threshold=base.improvement_threshold,
                evaluation_horizon=base.evaluation_horizon,
            ),
            DiscoveryPolicy(
                policy_id=f"{base.policy_id}.fast_discovery",
                version=version,
                epistemic_weight=base.epistemic_weight * 1.2,
                action_force=base.action_force * 1.5,
                anomaly_threshold=max(0.005, base.anomaly_threshold * 0.5),
                discovery_min_records=max(2, base.discovery_min_records - 1),
                evidence_gain_threshold=max(0.1, base.evidence_gain_threshold * 0.8),
                complexity_penalty_weight=base.complexity_penalty_weight,
                improvement_threshold=base.improvement_threshold,
                evaluation_horizon=base.evaluation_horizon,
            ),
        )


class StrategistSystem:
    def __init__(
        self,
        *,
        graph: CausalGraph | None = None,
        generator: PolicyGenerator | None = None,
    ) -> None:
        self.graph = graph or CausalGraph()
        self.generator = generator or PolicyGenerator()

    def optimize(
        self,
        trials: Iterable[SharedExogenousTrial],
        base_policy: DiscoveryPolicy,
    ) -> StrategyResult:
        shared_trials = tuple(trials)
        if not shared_trials:
            raise ValueError("Strategist optimization requires trials")
        evaluation_trials = shared_trials[: base_policy.evaluation_horizon]
        baseline = self.evaluate_policy(base_policy, evaluation_trials, baseline=None)
        candidates = tuple(
            self.evaluate_policy(candidate, evaluation_trials, baseline=baseline)
            for candidate in self.generator.generate(base_policy)
        )
        selected = max(candidates, key=lambda score: score.composite_score)
        adopted = (
            selected.composite_score > baseline.composite_score
            and selected.efficiency_gain >= (1.0 + base_policy.improvement_threshold)
        )
        selected_score = selected if adopted else baseline
        audit = self._integrate_decision(
            baseline,
            candidates,
            selected_score,
            adopted=adopted,
            trial_count=len(evaluation_trials),
        )
        return StrategyResult(
            baseline=baseline,
            candidates=candidates,
            selected=selected_score,
            adopted=adopted,
            audit=audit,
            graph_nodes=len(self.graph.nodes),
            graph_edges=len(self.graph.edges),
            ungrounded_nodes=len(self.graph.validate_grounding()),
        )

    def evaluate_policy(
        self,
        policy: DiscoveryPolicy,
        trials: tuple[SharedExogenousTrial, ...],
        *,
        baseline: PolicyScore | None,
    ) -> PolicyScore:
        replay = HistorianSystem().replay(
            trials,
            policy.to_historian_policy(),
            label=policy.policy_id.replace(".", "_"),
        )
        experiments_to_friction = replay.metrics.experiments_to_friction
        if experiments_to_friction is None:
            experiments_to_friction = len(trials) + 1
        efficiency = 1.0 / experiments_to_friction
        accuracy = 1.0 / (1.0 + replay.metrics.mean_mass_error)
        grounding = replay.metrics.confident_fraction
        if baseline is None:
            efficiency_gain = 1.0
            accuracy_gain = 1.0
            grounding_gain = 1.0
        else:
            baseline_experiments = baseline.experiments_to_friction or (len(trials) + 1)
            efficiency_gain = baseline_experiments / experiments_to_friction
            accuracy_gain = (1.0 + baseline.mean_mass_error) / (
                1.0 + replay.metrics.mean_mass_error
            )
            grounding_gain = grounding / max(baseline.confident_fraction, 1e-9)
        composite = 0.55 * efficiency + 0.30 * accuracy + 0.15 * grounding
        return PolicyScore(
            policy=policy,
            experiments_to_friction=replay.metrics.experiments_to_friction,
            mean_mass_error=replay.metrics.mean_mass_error,
            confident_fraction=grounding,
            efficiency_gain=efficiency_gain,
            accuracy_gain=accuracy_gain,
            grounding_gain=grounding_gain,
            composite_score=composite,
        )

    def _integrate_decision(
        self,
        baseline: PolicyScore,
        candidates: tuple[PolicyScore, ...],
        selected: PolicyScore,
        *,
        adopted: bool,
        trial_count: int,
    ) -> GroundingAudit:
        observation = self.graph.add_node(
            "strategist_observation.policy_replay",
            NodeType.OBSERVATION,
            GaussianBelief(selected.composite_score, 1e-6),
            evidence=GaussianBelief(selected.composite_score, 1e-6),
            modality="counterfactual_policy_evaluation",
            metadata={
                "trial_count": trial_count,
                "baseline_policy": baseline.policy.policy_id,
                "selected_policy": selected.policy.policy_id,
                "adopted": adopted,
            },
        )
        baseline_node = self._add_policy_score_node("baseline", baseline)
        selected_node = self._add_policy_score_node("selected", selected)
        decision = self.graph.add_node(
            f"strategy_decision.{selected.policy.policy_id}",
            NodeType.POLICY,
            GaussianBelief(selected.composite_score, 1e-4),
            modality="meta_policy_selection",
            metadata={
                "adopted": adopted,
                "selected_policy": asdict(selected.policy),
                "baseline_policy": asdict(baseline.policy),
                "candidate_count": len(candidates),
                "efficiency_gain": selected.efficiency_gain,
                "accuracy_gain": selected.accuracy_gain,
                "grounding_gain": selected.grounding_gain,
            },
        )
        self.graph.add_edge(
            observation.id,
            decision.id,
            EdgeType.OBSERVES,
            noise_variance=1e-4,
            confidence=selected.confident_fraction,
            learned=True,
        )
        self.graph.add_edge(
            baseline_node.id,
            decision.id,
            EdgeType.ENABLES,
            noise_variance=1e-4,
            learned=True,
        )
        self.graph.add_edge(
            selected_node.id,
            decision.id,
            EdgeType.CAUSES,
            noise_variance=1e-4,
            learned=True,
        )
        return MetacognitiveAuditor(self.graph).audit(decision.id)

    def _add_policy_score_node(self, role: str, score: PolicyScore):
        node = self.graph.add_node(
            f"policy_score.{role}.{score.policy.policy_id}",
            NodeType.POLICY,
            GaussianBelief(score.composite_score, 1e-4),
            modality="policy_score",
            metadata={
                "policy": asdict(score.policy),
                "experiments_to_friction": score.experiments_to_friction,
                "mean_mass_error": score.mean_mass_error,
                "confident_fraction": score.confident_fraction,
                "composite_score": score.composite_score,
            },
        )
        evidence = self.graph.add_node(
            f"policy_score_observation.{role}.{score.policy.policy_id}",
            NodeType.OBSERVATION,
            GaussianBelief(score.composite_score, 1e-6),
            evidence=GaussianBelief(score.composite_score, 1e-6),
            modality="policy_score_evidence",
        )
        self.graph.add_edge(
            evidence.id,
            node.id,
            EdgeType.OBSERVES,
            noise_variance=1e-4,
            confidence=score.confident_fraction,
            learned=True,
        )
        return node


def run_strategist_benchmark(
    *,
    experiments: int = 6,
    true_mass: float = 2.5,
    hidden_friction: float = 0.25,
) -> StrategyResult:
    trials = synthetic_history_trials(
        count=experiments,
        true_mass=true_mass,
        hidden_friction=hidden_friction,
    )
    base_policy = DiscoveryPolicy(policy_id="baseline", version=1)
    return StrategistSystem().optimize(trials, base_policy)
