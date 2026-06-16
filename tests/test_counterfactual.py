import unittest

from causal_constructivism.counterfactual import build_twin_graph
from causal_constructivism.models import (
    EdgeType,
    GaussianBelief,
    GroundingStatus,
    NodeType,
    ObjectAction,
)
from causal_constructivism.twin_physics import ParticleBodyPrior
from causal_constructivism.twin_system import default_twin_world


class CounterfactualTests(unittest.TestCase):
    def test_graph_surgery_is_isolated_and_preserves_outgoing_edges(self) -> None:
        priors = [
            ParticleBodyPrior(
                "red",
                GaussianBelief(2.0, 0.1),
                GaussianBelief(0.0, 0.1),
                GaussianBelief(0.0, 0.1),
                0.5,
            )
        ]
        graph = build_twin_graph(priors, ObjectAction("red", 10.0, 0.5))
        mass = graph.find_node_by_name("red.mass")
        downstream = graph.add_node(
            "red.acceleration",
            NodeType.STATE,
            GaussianBelief(0.0, 10.0),
        )
        graph.add_edge(
            mass.id,
            downstream.id,
            EdgeType.CAUSES,
            noise_variance=0.1,
        )
        twin = graph.clone()
        twin_mass = twin.find_node_by_name("red.mass")

        _, removed = twin.intervene(twin_mass.id, 4.0)

        self.assertGreaterEqual(len(removed), 1)
        self.assertEqual(len(twin.incoming_edges(twin_mass.id)), 0)
        self.assertEqual(len(twin.outgoing_edges(twin_mass.id)), 1)
        self.assertFalse(graph.find_node_by_name("red.mass").metadata.get("intervened"))

    def test_force_counterfactual_changes_trajectory_and_collision(self) -> None:
        system = default_twin_world(particles=120)

        result = system.counterfactual(
            {"action.force": 2.0},
            "green.position",
        )

        self.assertGreater(result.actual_collision_probability, 0.95)
        self.assertLess(result.collision_probability, 0.05)
        self.assertGreater(result.actual_belief.mean, result.belief.mean + 2.0)
        self.assertIsNot(result.grounding.status, GroundingStatus.UNGROUNDED)
        self.assertTrue(result.intervention_node_ids)
        self.assertTrue(result.shared_exogenous_node_ids)

    def test_mass_counterfactual_reduces_transferred_motion(self) -> None:
        system = default_twin_world(particles=120)

        result = system.counterfactual(
            {"red.mass": 4.0},
            "green.position",
        )

        self.assertGreater(result.actual_belief.mean, result.belief.mean)
        self.assertIs(result.grounding.status, GroundingStatus.CONFIDENT)

    def test_collision_query_returns_bernoulli_belief(self) -> None:
        system = default_twin_world(particles=80)

        result = system.counterfactual(
            {"action.force": 2.0},
            "collision.red.green",
        )

        self.assertGreater(result.actual_belief.mean, 0.95)
        self.assertLess(result.belief.mean, 0.05)


if __name__ == "__main__":
    unittest.main()
