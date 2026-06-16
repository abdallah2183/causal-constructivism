from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph import CausalGraph
from .models import GaussianBelief, GroundingStatus, NodeType
from .integrator import CognitiveOrchestrator, IntegratorResult


@dataclass(frozen=True, slots=True)
class IntegratorBenchmarkResult:
    result: IntegratorResult
    steps_count: int
    grounding_quality: float
    status: GroundingStatus


class IntegratorSystem:
    """Phase 16 Integration Facade for end-to-end cognitive orchestration."""

    def __init__(self, causal_graph: CausalGraph | None = None, agency: Any = None) -> None:
        self.causal_graph = causal_graph or CausalGraph()
        # Setup baseline property/concept nodes for the audit
        self.causal_graph.add_node("concept.friction", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        self.causal_graph.add_node("concept.restitution", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        self.causal_graph.add_node("concept.damping", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)

        self.orchestrator = CognitiveOrchestrator(
            causal_graph=self.causal_graph,
            agency=agency,
        )

    def run_benchmark(self) -> IntegratorBenchmarkResult:
        """Runs the complete end-to-end orchestration and returns audited results."""
        res = self.orchestrator.run_orchestration_cycle(target_concept="friction")
        return IntegratorBenchmarkResult(
            result=res,
            steps_count=len(res.steps),
            grounding_quality=res.grounding_quality,
            status=res.status,
        )


def run_integrator_benchmark() -> IntegratorBenchmarkResult:
    return IntegratorSystem().run_benchmark()
