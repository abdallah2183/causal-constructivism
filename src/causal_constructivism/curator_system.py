from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cartographer import CartographerSystem, ConceptNode, ConceptualGraph, MetaRelation
from .curator import Curator, FrontierOfIgnorance, ResearchAgenda
from .graph import CausalGraph
from .models import GroundingAudit, GroundingStatus, NodeType
from .narrator import Explanation, Narrator
from .narrator_system import NarratorSystem


@dataclass(frozen=True, slots=True)
class CuratorResult:
    agenda: ResearchAgenda
    question_count: int
    grounding_quality: float
    status: GroundingStatus


class CuratorSystem:
    """Phase 14 Integration Facade for autonomous research agenda curators."""

    def __init__(self, causal_graph: CausalGraph | None = None) -> None:
        self.causal_graph = causal_graph or CausalGraph()
        self.narrator_sys = NarratorSystem(causal_graph=self.causal_graph)
        self.frontier = FrontierOfIgnorance(
            self.narrator_sys.cartographer.ontology,
            self.causal_graph,
        )
        self.curator = Curator(
            self.frontier,
            self.narrator_sys.cartographer,
            None,
        )

    def run_benchmark(self) -> CuratorResult:
        """Runs the gap-detection, question ranking, and experiment proposal benchmark."""
        # Composer concepts
        self.narrator_sys.cartographer.update_from_composer(["friction", "restitution"], "modulates")

        # Analog concept
        self.narrator_sys.cartographer.ontology.add_node(ConceptNode(
            node_id="concept.damping",
            name="damping",
            type=NodeType.CONCEPT,
            domain="oscillations",
        ))
        self.narrator_sys.cartographer.ontology.add_edge(MetaRelation(
            source_id="concept.restitution",
            target_id="concept.damping",
            relation_type="is_analogous_to",
            confidence=0.85,
            provenance_edges=("prop.restitution",),
        ))

        # Generate explanation at depth=1 (friction + restitution only)
        explanation = self.narrator_sys.narrator.generate_explanation("concept.friction", depth=1)

        # Curate research questions from explanation
        agenda = self.curator.propose_research_agenda(explanation)
        grounding_quality = 1.0 if agenda.audit.status == GroundingStatus.CONFIDENT else 0.0

        return CuratorResult(
            agenda=agenda,
            question_count=len(agenda.questions),
            grounding_quality=grounding_quality,
            status=agenda.audit.status,
        )


def run_curator_benchmark() -> CuratorResult:
    return CuratorSystem().run_benchmark()
