import unittest

from causal_constructivism.cartographer import CartographerSystem, ConceptNode, MetaRelation
from causal_constructivism.collaborator import Collaborator, ScientistAgent
from causal_constructivism.collaborator_system import run_collaborator_benchmark
from causal_constructivism.curator import Curator, FrontierOfIgnorance
from causal_constructivism.graph import CausalGraph
from causal_constructivism.models import GaussianBelief, GroundingStatus, NodeType


class CollaboratorTests(unittest.TestCase):
    def test_collaborator_debate_initialization_and_flow(self) -> None:
        causal_graph = CausalGraph()
        causal_graph.add_node("concept.friction", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        causal_graph.add_node("concept.restitution", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)
        causal_graph.add_node("concept.damping", NodeType.CONCEPT, GaussianBelief(0.0, 1.0), is_axiom=True)

        cartographer = CartographerSystem(graph=causal_graph)
        cartographer.update_from_composer(["friction", "restitution"], "modulates")
        cartographer.ontology.add_node(ConceptNode(
            node_id="concept.damping",
            name="damping",
            type=NodeType.CONCEPT,
            domain="oscillations",
        ))

        frontier = FrontierOfIgnorance(cartographer.ontology, causal_graph)
        curator = Curator(frontier, cartographer, None)
        collaborator = Collaborator(cartographer, curator, None, None)

        agent_priors = {
            "Agent_A": {
                "concept.friction": GaussianBelief(0.25, 0.01),
                "concept.damping": GaussianBelief(0.0, 0.001),
            },
            "Agent_B": {
                "concept.friction": GaussianBelief(0.0, 0.001),
                "concept.damping": GaussianBelief(0.25, 0.01),
            },
        }

        record = collaborator.debate("concept.friction", agent_priors)

        # Check convergence and consensus
        self.assertEqual(len(record.rounds), 1)
        self.assertIsNotNone(record.final_consensus)
        self.assertEqual(record.final_consensus.relation_type, "consensus")
        self.assertEqual(record.audit.status, GroundingStatus.CONFIDENT)

    def test_collaborator_benchmark_result(self) -> None:
        result = run_collaborator_benchmark()

        self.assertEqual(result.rounds_count, 1)
        self.assertEqual(result.grounding_quality, 1.0)
        self.assertEqual(result.status, GroundingStatus.CONFIDENT)
        self.assertEqual(result.record.final_consensus.source_id, "concept.friction")
        self.assertEqual(result.record.final_consensus.target_id, "concept.damping")


if __name__ == "__main__":
    unittest.main()
