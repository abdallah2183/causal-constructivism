import unittest
from dataclasses import dataclass

from causal_constructivism.cartographer import (
    CartographerBenchmarkResult,
    CartographerSystem,
    ConceptNode,
    ConceptualGraph,
    MetaRelation,
    run_cartographer_benchmark,
)
from causal_constructivism import run_cartographer_benchmark as exported_run_cartographer_benchmark
from causal_constructivism.graph import CausalGraph
from causal_constructivism.models import GroundingStatus, NodeType, GaussianBelief


@dataclass
class MockLawInstance:
    law_id: str
    domain: str


@dataclass
class MockUnifyingPrinciple:
    name: str
    confidence: float
    instances: list[MockLawInstance]


class CartographerTests(unittest.TestCase):
    def test_cartographer_ontology_construction_and_traversal(self) -> None:
        graph = ConceptualGraph()
        
        node_a = ConceptNode(node_id="concept.friction", name="friction", type=NodeType.CONCEPT, domain="mechanics")
        node_b = ConceptNode(node_id="concept.restitution", name="restitution", type=NodeType.CONCEPT, domain="collisions")
        node_c = ConceptNode(node_id="concept.damping", name="damping", type=NodeType.CONCEPT, domain="oscillations")
        
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        
        edge_1 = MetaRelation(source_id="concept.friction", target_id="concept.restitution", relation_type="modulates", confidence=0.9)
        edge_2 = MetaRelation(source_id="concept.restitution", target_id="concept.damping", relation_type="is_analogous_to", confidence=0.85)
        
        graph.add_edge(edge_1)
        graph.add_edge(edge_2)
        
        # Test neighborhood BFS
        neighbors = graph.neighborhood("concept.friction", depth=1)
        self.assertIn("concept.restitution", neighbors)
        self.assertNotIn("concept.damping", neighbors)
        
        neighbors_depth_2 = graph.neighborhood("concept.friction", depth=2)
        self.assertIn("concept.damping", neighbors_depth_2)
        
        # Test analogous traversal
        analogies = graph.analogous_concepts("concept.restitution")
        self.assertEqual(analogies, ["concept.damping"])

    def test_update_from_composer(self) -> None:
        cartographer = CartographerSystem()
        cartographer.update_from_composer(["friction", "restitution"], "modulates")
        
        self.assertIn("concept.friction", cartographer.ontology.nodes)
        self.assertIn("concept.restitution", cartographer.ontology.nodes)
        self.assertIn(
            ("concept.friction", "concept.restitution", "modulates"),
            cartographer.ontology.edges,
        )
        edge = cartographer.ontology.edges[
            ("concept.friction", "concept.restitution", "modulates")
        ]
        self.assertEqual(edge.provenance_edges, ("edge.friction_restitution",))

    def test_update_from_universalist(self) -> None:
        cartographer = CartographerSystem()
        principle = MockUnifyingPrinciple(
            name="harmonic_motion",
            confidence=0.95,
            instances=[
                MockLawInstance(law_id="pendulum_law", domain="pendulum"),
                MockLawInstance(law_id="spring_law", domain="spring"),
            ],
        )
        
        cartographer.update_from_universalist(principle)
        
        self.assertIn("meta_law.harmonic_motion", cartographer.ontology.nodes)
        self.assertIn("concept.pendulum_law", cartographer.ontology.nodes)
        self.assertIn("concept.spring_law", cartographer.ontology.nodes)
        self.assertIn(
            ("meta_law.harmonic_motion", "concept.pendulum_law", "is_analogous_to"),
            cartographer.ontology.edges,
        )

    def test_audit_meta_edge_grounding(self) -> None:
        causal_graph = CausalGraph()
        # Create a mock property node in causal graph and mark as axiom (which makes it confident)
        causal_graph.add_node("prop.friction", NodeType.PROPERTY, GaussianBelief(0.0, 1.0), is_axiom=True)
        
        cartographer = CartographerSystem(graph=causal_graph)
        edge = MetaRelation(
            source_id="concept.friction",
            target_id="concept.restitution",
            relation_type="modulates",
            confidence=0.9,
            provenance_edges=("prop.friction",),
        )
        
        status = cartographer.audit_meta_edge(edge)
        self.assertEqual(status, GroundingStatus.CONFIDENT)

    def test_distinguishing_experiments(self) -> None:
        cartographer = CartographerSystem()
        proposals = cartographer.distinguishing_experiments(
            ["friction_restitution", "damping"],
            history=[],
        )
        self.assertTrue(len(proposals) > 0)
        self.assertEqual(proposals[0]["expected_information_gain"], 0.95)

    def test_cartographer_benchmark_result(self) -> None:
        result = run_cartographer_benchmark()

        self.assertIsInstance(result, CartographerBenchmarkResult)
        self.assertEqual(result.query_node, "concept.friction")
        self.assertEqual(result.node_count, 3)
        self.assertEqual(result.edge_count, 2)
        self.assertEqual(result.grounded_edge_count, 2)
        self.assertEqual(result.grounding_quality, 1.0)
        self.assertEqual(result.status, GroundingStatus.CONFIDENT)
        self.assertEqual(
            result.neighborhood,
            ("concept.damping", "concept.friction", "concept.restitution"),
        )
        self.assertEqual(result.analogies, ("concept.damping",))
        self.assertEqual(
            result.proposed_experiments[0]["experiment_type"],
            "test_concept.damping",
        )

    def test_cartographer_benchmark_is_exported(self) -> None:
        self.assertIs(exported_run_cartographer_benchmark, run_cartographer_benchmark)


if __name__ == "__main__":
    unittest.main()
