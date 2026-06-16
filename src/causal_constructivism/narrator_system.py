from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cartographer import CartographerSystem, ConceptNode, ConceptualGraph, MetaRelation
from .graph import CausalGraph
from .models import GroundingAudit, GroundingStatus, NodeType
from .narrator import Explanation, Narrator


@dataclass(frozen=True, slots=True)
class NarratorResult:
    explanation: Explanation
    sentence_count: int
    grounding_quality: float
    status: GroundingStatus


class NarratorSystem:
    """Phase 13 Integration Facade for grounded explanation generation benchmarks."""

    def __init__(self, causal_graph: CausalGraph | None = None) -> None:
        self.causal_graph = causal_graph or CausalGraph()
        self.cartographer = CartographerSystem(graph=self.causal_graph)
        self.narrator = Narrator(self.cartographer.ontology, self.causal_graph)

    def run_benchmark(self) -> NarratorResult:
        """Populates concept interactions and generates a grounded 5-sentence explanation benchmark."""
        # Composer concepts
        self.cartographer.update_from_composer(["friction", "restitution"], "modulates")

        # Analog concept
        self.cartographer.ontology.add_node(ConceptNode(
            node_id="concept.damping",
            name="damping",
            type=NodeType.CONCEPT,
            domain="oscillations",
        ))
        self.cartographer.ontology.add_edge(MetaRelation(
            source_id="concept.restitution",
            target_id="concept.damping",
            relation_type="is_analogous_to",
            confidence=0.85,
            provenance_edges=("prop.restitution",),
        ))

        # Generate explanation for the friction concept
        explanation = self.narrator.generate_explanation("concept.friction", depth=3)
        grounding_quality = 1.0 if explanation.audit.status == GroundingStatus.CONFIDENT else 0.0

        return NarratorResult(
            explanation=explanation,
            sentence_count=len(explanation.sentences),
            grounding_quality=grounding_quality,
            status=explanation.audit.status,
        )


def run_narrator_benchmark() -> NarratorResult:
    return NarratorSystem().run_benchmark()
