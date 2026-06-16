import unittest

from causal_constructivism.graph import CausalGraph
from causal_constructivism.inference import GaussianBeliefPropagation
from causal_constructivism.models import EdgeType, GaussianBelief, NodeType


class InferenceTests(unittest.TestCase):
    def test_observation_updates_connected_latent_state(self) -> None:
        graph = CausalGraph()
        latent = graph.add_node(
            "latent",
            NodeType.PROPERTY,
            GaussianBelief(0.0, 100.0),
        )
        observation = graph.add_node(
            "measurement",
            NodeType.OBSERVATION,
            GaussianBelief(0.0, 100.0),
            evidence=GaussianBelief(10.0, 0.01),
        )
        graph.add_edge(
            latent.id,
            observation.id,
            EdgeType.PREDICTS,
            weight=2.0,
            bias=1.0,
            noise_variance=0.01,
        )

        result = GaussianBeliefPropagation(graph, damping=0.0).run(
            [observation.id]
        )

        self.assertTrue(result.converged)
        self.assertLess(abs(latent.belief.mean - 4.5), 0.05)
        self.assertLess(latent.belief.variance, latent.prior.variance)


if __name__ == "__main__":
    unittest.main()
