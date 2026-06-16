from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cartographer import CartographerSystem, ConceptNode, ConceptualGraph, MetaRelation
from .collaborator import Collaborator, DebateRecord
from .curator import Curator, FrontierOfIgnorance
from .graph import CausalGraph
from .models import GaussianBelief, GroundingAudit, GroundingStatus, NodeType
from .narrator import Explanation, Narrator


@dataclass(frozen=True, slots=True)
class CollaboratorResult:
    record: DebateRecord
    rounds_count: int
    grounding_quality: float
    status: GroundingStatus


class CollaboratorSystem:
    """Phase 15 Integration Facade for multi-agent scientific debate records."""

    def __init__(self, causal_graph: CausalGraph | None = None) -> None:
        self.causal_graph = causal_graph or CausalGraph()
        # Expose necessary causal graph property/concept nodes for the audit
        self.causal_graph.add_node("concept.friction", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        self.causal_graph.add_node("concept.restitution", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        self.causal_graph.add_node("concept.damping", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)

        self.cartographer = CartographerSystem(graph=self.causal_graph)
        self.frontier = FrontierOfIgnorance(
            self.cartographer.ontology,
            self.causal_graph,
        )
        self.curator = Curator(
            self.frontier,
            self.cartographer,
            None,
        )
        self.collaborator = Collaborator(
            self.cartographer,
            self.curator,
            None,
            None,
        )

    def run_benchmark(self) -> CollaboratorResult:
        """Runs the debate, convergence, and consensus recording benchmark."""
        # Composer concepts
        self.cartographer.update_from_composer(["friction", "restitution"], "modulates")

        # Analog concept
        self.cartographer.ontology.add_node(ConceptNode(
            node_id="concept.damping",
            name="damping",
            type=NodeType.CONCEPT,
            domain="oscillations",
        ))

        # Setup agent priors
        agent_priors = {
            "Agent_A": {
                "concept.friction": GaussianBelief(0.25, 0.01),
                "concept.damping": GaussianBelief(0.0, 0.001),
            },
            "Agent_B": {
                "concept.friction": GaussianBelief(0.0, 0.001),
                "concept.damping": GaussianBelief(0.25, 0.01),
            },
        }

        # Run debate on friction concept
        record = self.collaborator.debate("concept.friction", agent_priors)
        grounding_quality = 1.0 if record.audit.status == GroundingStatus.CONFIDENT else 0.0

        return CollaboratorResult(
            record=record,
            rounds_count=len(record.rounds),
            grounding_quality=grounding_quality,
            status=record.audit.status,
        )


def run_collaborator_benchmark() -> CollaboratorResult:
    return CollaboratorSystem().run_benchmark()
