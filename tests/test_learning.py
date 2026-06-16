import unittest

from causal_constructivism.graph import CausalGraph
from causal_constructivism.learning import StructureLearner
from causal_constructivism.models import GaussianBelief, NodeType


class LearningTests(unittest.TestCase):
    def test_structure_learner_fits_and_integrates_linear_relation(self) -> None:
        graph = CausalGraph()
        source = graph.add_node("x", NodeType.OBSERVATION, GaussianBelief(0, 10))
        target = graph.add_node("y", NodeType.OBSERVATION, GaussianBelief(0, 10))
        learner = StructureLearner(graph)
        for value in range(1, 9):
            learner.observe(source.id, float(value))
            learner.observe(target.id, 2.0 * value + 3.0)

        proposal = learner.propose_edge(source.id, target.id)

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertGreater(proposal.score, 0)
        self.assertLess(abs(proposal.weight - 2.0), 1e-9)
        edge_id = learner.integrate(proposal)
        self.assertTrue(graph.require_edge(edge_id).learned)


if __name__ == "__main__":
    unittest.main()
