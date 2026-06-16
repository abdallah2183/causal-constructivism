import unittest

from causal_constructivism.cartographer import ConceptNode, ConceptualGraph, MetaRelation
from causal_constructivism.curator import Curator, FrontierOfIgnorance, ResearchQuestion
from causal_constructivism.curator_system import run_curator_benchmark
from causal_constructivism.graph import CausalGraph
from causal_constructivism.models import GroundingStatus, NodeType
from causal_constructivism.narrator import Explanation, ExplanationSentence, GroundingAudit


class CuratorTests(unittest.TestCase):
    def test_curator_gap_detection_and_ranking(self) -> None:
        causal_graph = CausalGraph()
        ontology = ConceptualGraph()

        # Add concepts
        ontology.add_node(ConceptNode(node_id="concept.friction", name="friction", type=NodeType.CONCEPT, domain="mechanics"))
        ontology.add_node(ConceptNode(node_id="concept.restitution", name="restitution", type=NodeType.CONCEPT, domain="mechanics"))
        ontology.add_node(ConceptNode(node_id="concept.damping", name="damping", type=NodeType.CONCEPT, domain="mechanics"))

        # Add edge modulates
        ontology.add_edge(MetaRelation(source_id="concept.friction", target_id="concept.restitution", relation_type="modulates", confidence=0.90))

        # Create explanation mentioning only friction and restitution
        explanation = Explanation(
            sentences=(
                ExplanationSentence(text="friction modulates restitution", provenance_edges=("concept.friction", "concept.restitution"), confidence=0.90),
            ),
            query_node="concept.friction",
            audit=GroundingAudit("explanation", GroundingStatus.CONFIDENT, 0.90, (), ()),
        )

        frontier = FrontierOfIgnorance(ontology, causal_graph)
        gaps = frontier.identify_gaps(explanation)

        # Should identify damping as a gap because it has 0 degree (low degree < 2)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].target_concepts, ("concept.damping",))

        # Test ranking
        ranked = frontier.rank_questions(gaps, None)
        self.assertEqual(len(ranked), 1)
        self.assertTrue(ranked[0].priority_score > 0.0)

    def test_curator_benchmark_agenda(self) -> None:
        result = run_curator_benchmark()

        self.assertEqual(result.question_count, 1)
        self.assertIn("damping", result.agenda.questions[0].text)
        self.assertEqual(result.grounding_quality, 1.0)
        self.assertEqual(result.status, GroundingStatus.CONFIDENT)


if __name__ == "__main__":
    unittest.main()
