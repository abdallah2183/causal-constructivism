import unittest

from causal_constructivism.cartographer import ConceptNode, ConceptualGraph, MetaRelation
from causal_constructivism.graph import CausalGraph
from causal_constructivism.models import GaussianBelief, GroundingStatus, NodeType
from causal_constructivism.narrator import ExplanationSentence, Narrator, TemplateGrammar
from causal_constructivism.narrator_system import run_narrator_benchmark


class NarratorTests(unittest.TestCase):
    def test_template_grammar_generation(self) -> None:
        text_c = TemplateGrammar.concept("friction", 0.25, 0.9, 8)
        self.assertIn("friction", text_c)
        self.assertIn("0.25", text_c)

        text_r = TemplateGrammar.relation("friction", "restitution", "modulates", "Composer law")
        self.assertIn("friction", text_r)
        self.assertIn("restitution", text_r)
        self.assertIn("modulates", text_r)

    def test_narrator_explanation_generation_and_audit(self) -> None:
        causal_graph = CausalGraph()
        causal_graph.add_node("concept.friction", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        causal_graph.add_node("concept.restitution", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        causal_graph.add_node("concept.damping", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)

        ontology = ConceptualGraph()
        node_a = ConceptNode(node_id="concept.friction", name="friction", type=NodeType.CONCEPT, domain="slide-collision")
        node_b = ConceptNode(node_id="concept.restitution", name="restitution", type=NodeType.CONCEPT, domain="slide-collision")
        ontology.add_node(node_a)
        ontology.add_node(node_b)

        edge = MetaRelation(
            source_id="concept.friction",
            target_id="concept.restitution",
            relation_type="modulates",
            confidence=0.95,
            provenance_edges=("concept.friction", "concept.restitution"),
        )
        ontology.add_edge(edge)

        narrator = Narrator(ontology, causal_graph)
        explanation = narrator.generate_explanation("concept.friction", depth=2)

        self.assertEqual(len(explanation.sentences), 3)  # 2 concept sentences + 1 relation sentence
        self.assertEqual(explanation.audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(explanation.audit.confidence, 0.90)  # Min sentence confidence is 0.90 (concept template confidence)

    def test_narrator_benchmark_result(self) -> None:
        result = run_narrator_benchmark()
        self.assertEqual(result.sentence_count, 5)
        self.assertEqual(result.grounding_quality, 1.0)
        self.assertEqual(result.status, GroundingStatus.CONFIDENT)


if __name__ == "__main__":
    unittest.main()
