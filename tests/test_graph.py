import unittest

from causal_constructivism.graph import CausalGraph
from causal_constructivism.models import EdgeType, GaussianBelief, NodeType


class GraphTests(unittest.TestCase):
    def test_grounding_and_version_history(self) -> None:
        graph = CausalGraph()
        observation = graph.add_node(
            "camera",
            NodeType.OBSERVATION,
            GaussianBelief(3.0, 0.2),
            evidence=GaussianBelief(3.1, 0.1),
        )
        concept = graph.add_node(
            "object",
            NodeType.CONCEPT,
            GaussianBelief(0.0, 10.0),
        )
        graph.add_edge(
            observation.id,
            concept.id,
            EdgeType.OBSERVES,
            noise_variance=0.5,
        )

        self.assertEqual(graph.validate_grounding(), ())
        replacement = graph.version_node(
            concept.id,
            prior=GaussianBelief(1.0, 2.0),
        )

        self.assertEqual(replacement.version, 2)
        self.assertEqual(graph.require_node(concept.id).superseded_by, replacement.id)
        self.assertEqual(len(graph.node_history(concept.id)), 1)
        self.assertTrue(graph.grounding_paths(replacement.id))

    def test_graph_rejects_duplicate_typed_edges(self) -> None:
        graph = CausalGraph()
        left = graph.add_node("left", NodeType.CONCEPT, GaussianBelief(0, 1))
        right = graph.add_node("right", NodeType.CONCEPT, GaussianBelief(0, 1))
        graph.add_edge(left.id, right.id, EdgeType.CAUSES)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            graph.add_edge(left.id, right.id, EdgeType.CAUSES)


if __name__ == "__main__":
    unittest.main()
