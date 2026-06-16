import tempfile
import unittest
from pathlib import Path

from causal_constructivism.graph import CausalGraph
from causal_constructivism.models import EdgeType, GaussianBelief, NodeType
from causal_constructivism.persistence import SQLiteGraphStore


class PersistenceTests(unittest.TestCase):
    def test_graph_round_trip_preserves_ids_edges_and_history(self) -> None:
        graph = CausalGraph()
        observation = graph.add_node(
            "sensor",
            NodeType.OBSERVATION,
            GaussianBelief(2.0, 0.1),
            evidence=GaussianBelief(2.1, 0.05),
            metadata={"channels": ("depth", "segment")},
        )
        state = graph.add_node(
            "state",
            NodeType.STATE,
            GaussianBelief(0.0, 5.0),
        )
        edge = graph.add_edge(
            observation.id,
            state.id,
            EdgeType.OBSERVES,
            noise_variance=0.2,
        )
        replacement = graph.version_node(
            state.id,
            prior=GaussianBelief(1.0, 1.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteGraphStore(Path(directory) / "graph.db")
            snapshot_id = store.save(graph, label="round-trip")
            restored = store.load(snapshot_id)

        self.assertEqual(len(restored.nodes), len(graph.nodes))
        self.assertEqual(len(restored.edges), len(graph.edges))
        self.assertEqual(restored.require_edge(edge.id).source_id, observation.id)
        self.assertEqual(restored.require_node(replacement.id).version, 2)
        self.assertEqual(len(restored.node_history(state.id)), 1)
        self.assertTrue(restored.grounding_paths(replacement.id))

    def test_snapshots_are_append_only_and_latest_is_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteGraphStore(Path(directory) / "graph.db")
            first = CausalGraph()
            first.add_node("one", NodeType.CONCEPT, GaussianBelief(1.0, 1.0))
            first_id = store.save(first, label="first")
            second = first.clone()
            second.add_node("two", NodeType.CONCEPT, GaussianBelief(2.0, 1.0))
            second_id = store.save(second, label="second")

            latest = store.load()
            snapshots = store.list_snapshots()

        self.assertLess(first_id, second_id)
        self.assertEqual(len(latest.nodes), 2)
        self.assertEqual([item[1] for item in snapshots], ["first", "second"])


if __name__ == "__main__":
    unittest.main()

