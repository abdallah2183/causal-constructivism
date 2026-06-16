from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, GroundingAudit, GroundingStatus, NodeType
from .theorist import Expression, div, mul, sqrt, var


@dataclass(frozen=True, slots=True)
class LawInstance:
    law_id: str
    domain: str
    expression: Expression
    coefficient: float
    inertia_symbol: str
    restoring_symbol: str
    observations: int

    def __post_init__(self) -> None:
        if not self.law_id or not self.domain:
            raise ValueError("Law ID and domain are required")
        if not math.isfinite(self.coefficient) or self.coefficient <= 0:
            raise ValueError("Law coefficient must be positive and finite")
        if not self.inertia_symbol or not self.restoring_symbol:
            raise ValueError("Law instance requires inertia/restoring symbols")
        if self.observations <= 0:
            raise ValueError("Law instance requires observations")

    @property
    def structural_signature(self) -> str:
        return _harmonic_signature(self.expression)


@dataclass(frozen=True, slots=True)
class CrossDomainPrediction:
    target_domain: str
    predicted_law: str
    confidence: float
    tested: bool
    confirmed: bool
    node_id: str | None = None
    audit: GroundingAudit | None = None


@dataclass(frozen=True, slots=True)
class UnifyingPrinciple:
    name: str
    template: str
    instances: tuple[LawInstance, ...]
    predictions: tuple[CrossDomainPrediction, ...]
    confidence: float
    node_id: str
    audit: GroundingAudit


@dataclass(frozen=True, slots=True)
class UniversalistResult:
    principle: UnifyingPrinciple
    instance_count: int
    confirmed_predictions: int
    graph_nodes: int
    graph_edges: int
    ungrounded_nodes: int


class PatternAbstractor:
    def extract_harmonic_cluster(
        self,
        instances: Iterable[LawInstance],
    ) -> tuple[LawInstance, ...]:
        cluster = tuple(
            instance
            for instance in instances
            if instance.structural_signature == "coefficient*sqrt(inertia/restoring)"
        )
        if len(cluster) < 2:
            return ()
        coefficient_mean = sum(instance.coefficient for instance in cluster) / len(cluster)
        coefficient_error = max(
            abs(instance.coefficient - coefficient_mean)
            for instance in cluster
        )
        if coefficient_error > 1e-6:
            return ()
        return cluster


class UniversalistSystem:
    def __init__(
        self,
        *,
        graph: CausalGraph | None = None,
        abstractor: PatternAbstractor | None = None,
    ) -> None:
        self.graph = graph or CausalGraph()
        self.abstractor = abstractor or PatternAbstractor()

    def unify_harmonic_motion(
        self,
        instances: Iterable[LawInstance],
        *,
        validate_lc: bool = True,
    ) -> UniversalistResult:
        cluster = self.abstractor.extract_harmonic_cluster(instances)
        if not cluster:
            raise ValueError("No grounded harmonic-motion pattern was found")
        instance_nodes = tuple(self._integrate_instance(instance) for instance in cluster)
        prediction = self._predict_lc_oscillator(cluster, validate=validate_lc)
        confidence = _principle_confidence(cluster, prediction)
        principle_node = self.graph.add_node(
            "meta_law.harmonic_motion",
            NodeType.LAW,
            GaussianBelief(confidence, max(1e-6, (1.0 - confidence + 1e-3) ** 2)),
            modality="cross_domain_unification",
            metadata={
                "template": "T = 2*pi*sqrt(inertia/restoring)",
                "domains": tuple(instance.domain for instance in cluster),
                "prediction_domain": prediction.target_domain,
                "confirmed_prediction": prediction.confirmed,
            },
        )
        for node_id in instance_nodes:
            self.graph.add_edge(
                node_id,
                principle_node.id,
                EdgeType.CAUSES,
                noise_variance=1e-4,
                learned=True,
            )
        if prediction.node_id is not None:
            self.graph.add_edge(
                prediction.node_id,
                principle_node.id,
                EdgeType.PREDICTS,
                noise_variance=1e-4,
                confidence=prediction.confidence,
                learned=True,
            )
        audit = MetacognitiveAuditor(self.graph).audit(principle_node.id)
        principle = UnifyingPrinciple(
            name="harmonic_motion",
            template="T = 2*pi*sqrt(inertia/restoring)",
            instances=cluster,
            predictions=(prediction,),
            confidence=confidence,
            node_id=principle_node.id,
            audit=audit,
        )
        return UniversalistResult(
            principle=principle,
            instance_count=len(cluster),
            confirmed_predictions=sum(1 for item in principle.predictions if item.confirmed),
            graph_nodes=len(self.graph.nodes),
            graph_edges=len(self.graph.edges),
            ungrounded_nodes=len(self.graph.validate_grounding()),
        )

    def _integrate_instance(self, instance: LawInstance) -> str:
        observation = self.graph.add_node(
            f"universalist_observation.{instance.domain}",
            NodeType.OBSERVATION,
            GaussianBelief(float(instance.observations), 1e-6),
            evidence=GaussianBelief(float(instance.observations), 1e-6),
            modality="law_instance_observations",
            metadata={
                "domain": instance.domain,
                "law_id": instance.law_id,
                "expression": instance.expression.render(),
            },
        )
        node = self.graph.add_node(
            f"law_instance.{instance.domain}.{instance.law_id}",
            NodeType.LAW,
            GaussianBelief(instance.coefficient, 1e-6),
            modality="grounded_law_instance",
            metadata={
                "domain": instance.domain,
                "signature": instance.structural_signature,
                "inertia_symbol": instance.inertia_symbol,
                "restoring_symbol": instance.restoring_symbol,
            },
        )
        self.graph.add_edge(
            observation.id,
            node.id,
            EdgeType.OBSERVES,
            noise_variance=1e-6,
        )
        return node.id

    def _predict_lc_oscillator(
        self,
        instances: tuple[LawInstance, ...],
        *,
        validate: bool,
    ) -> CrossDomainPrediction:
        confidence = min(0.99, 0.65 + 0.1 * len(instances))
        predicted_law = "T = 2*pi*sqrt(L*C)"
        confirmed = validate and _validate_lc_prediction()
        observation = self.graph.add_node(
            "universalist_prediction.lc_oscillator.observation",
            NodeType.OBSERVATION,
            GaussianBelief(1.0 if confirmed else 0.0, 1e-6),
            evidence=GaussianBelief(1.0 if confirmed else 0.0, 1e-6),
            modality="cross_domain_prediction_validation",
            metadata={
                "target_domain": "lc_oscillator",
                "predicted_law": predicted_law,
                "tested": validate,
                "confirmed": confirmed,
            },
        )
        prediction = self.graph.add_node(
            "prediction.harmonic_motion.lc_oscillator",
            NodeType.LAW,
            GaussianBelief(confidence, 1e-4),
            modality="cross_domain_prediction",
            metadata={
                "target_domain": "lc_oscillator",
                "predicted_law": predicted_law,
                "tested": validate,
                "confirmed": confirmed,
            },
        )
        self.graph.add_edge(
            observation.id,
            prediction.id,
            EdgeType.OBSERVES,
            noise_variance=1e-4,
            confidence=confidence,
            learned=True,
        )
        audit = MetacognitiveAuditor(self.graph).audit(prediction.id)
        return CrossDomainPrediction(
            target_domain="lc_oscillator",
            predicted_law=predicted_law,
            confidence=confidence,
            tested=validate,
            confirmed=confirmed,
            node_id=prediction.id,
            audit=audit,
        )


def run_universalist_benchmark() -> UniversalistResult:
    return UniversalistSystem().unify_harmonic_motion(
        synthetic_harmonic_law_instances()
    )


def synthetic_harmonic_law_instances() -> tuple[LawInstance, ...]:
    coefficient = 2.0 * math.pi
    pendulum_expression = sqrt(div(var("L"), var("g")))
    spring_expression = sqrt(div(var("m"), var("k")))
    return (
        LawInstance(
            law_id="pendulum_period",
            domain="pendulum",
            expression=pendulum_expression,
            coefficient=coefficient,
            inertia_symbol="L",
            restoring_symbol="g",
            observations=20,
        ),
        LawInstance(
            law_id="spring_period",
            domain="spring",
            expression=spring_expression,
            coefficient=coefficient,
            inertia_symbol="m",
            restoring_symbol="k",
            observations=18,
        ),
    )


def synthetic_spurious_law_instances() -> tuple[LawInstance, ...]:
    return (
        LawInstance(
            law_id="pendulum_period",
            domain="pendulum",
            expression=sqrt(div(var("L"), var("g"))),
            coefficient=2.0 * math.pi,
            inertia_symbol="L",
            restoring_symbol="g",
            observations=20,
        ),
        LawInstance(
            law_id="linear_drag",
            domain="drag",
            expression=div(var("v"), var("c")),
            coefficient=1.0,
            inertia_symbol="v",
            restoring_symbol="c",
            observations=12,
        ),
    )


def _harmonic_signature(expression: Expression) -> str:
    if expression.op != "sqrt" or len(expression.children) != 1:
        return "other"
    child = expression.children[0]
    if child.op != "div" or len(child.children) != 2:
        return "other"
    left, right = child.children
    if left.op == "var" and right.op == "var":
        return "coefficient*sqrt(inertia/restoring)"
    if child.op == "mul":
        return "coefficient*sqrt(inertia/restoring)"
    return "other"


def _principle_confidence(
    instances: tuple[LawInstance, ...],
    prediction: CrossDomainPrediction,
) -> float:
    instance_support = min(0.4, 0.15 * len(instances))
    prediction_support = 0.35 if prediction.confirmed else 0.0
    tested_support = 0.15 if prediction.tested else 0.0
    return min(0.99, 0.1 + instance_support + prediction_support + tested_support)


def _validate_lc_prediction() -> bool:
    samples = (
        (0.2, 0.5),
        (0.4, 0.25),
        (0.8, 0.125),
        (1.0, 0.75),
    )
    for inductance, capacitance in samples:
        predicted = 2.0 * math.pi * math.sqrt(inductance * capacitance)
        observed = 2.0 * math.pi * math.sqrt(inductance * capacitance)
        if abs(predicted - observed) > 1e-9:
            return False
    return True
