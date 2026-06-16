import unittest

from causal_constructivism.audit import MetacognitiveAuditor
from causal_constructivism.models import GroundingStatus, NodeType
from causal_constructivism.twin_system import default_twin_world


class ContactLearningTests(unittest.TestCase):
    def test_repeated_collision_creates_grounded_contact_relation(self) -> None:
        system = default_twin_world(particles=20)

        relation_ids = system.learn_contact_relations(3)

        self.assertEqual(len(relation_ids), 1)
        relation = system.graph.require_node(relation_ids[0])
        self.assertIs(relation.node_type, NodeType.CONTACT_RELATION)
        audit = MetacognitiveAuditor(system.graph).audit(relation.id)
        self.assertIsNot(audit.status, GroundingStatus.UNGROUNDED)


if __name__ == "__main__":
    unittest.main()

