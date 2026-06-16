import tempfile
import unittest
from pathlib import Path

from causal_constructivism.discovery import (
    AnomalyDetector,
    DiscoverySystem,
    ExperimentLog,
    NeuralStructureProposer,
    friction_features,
    synthetic_constant_velocity_records,
    synthetic_discovery_curriculum,
    synthetic_sliding_records,
)
from causal_constructivism.models import GroundingStatus, NodeType


class DiscoveryTests(unittest.TestCase):
    def test_experiment_log_round_trip(self) -> None:
        log = ExperimentLog(synthetic_sliding_records(friction=0.2, count=4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.jsonl"
            log.save_jsonl(path)
            restored = ExperimentLog.load_jsonl(path)

        self.assertEqual(restored.records, log.records)

    def test_neural_proposer_separates_friction_from_constant_velocity(self) -> None:
        proposer = NeuralStructureProposer()
        proposer.train(synthetic_discovery_curriculum())

        friction_probability = proposer.probability(
            friction_features(synthetic_sliding_records(friction=0.3))
        )
        constant_probability = proposer.probability(
            friction_features(synthetic_constant_velocity_records())
        )

        self.assertGreater(friction_probability, 0.9)
        self.assertLess(constant_probability, 0.1)

    def test_anomaly_detector_rejects_low_error_motion(self) -> None:
        detector = AnomalyDetector()
        anomaly = detector.detect(
            ExperimentLog(synthetic_constant_velocity_records()),
            "block_001",
        )

        self.assertIsNone(anomaly)

    def test_discovery_integrates_grounded_friction_concept(self) -> None:
        system = DiscoverySystem()
        result = system.discover_friction(
            ExperimentLog(synthetic_sliding_records(friction=0.25)),
            "block_001",
        )

        self.assertIsNotNone(result)
        assert result is not None
        scored, node_id, audit = result
        node = system.graph.require_node(node_id)
        self.assertEqual(scored.candidate.concept, "friction")
        self.assertAlmostEqual(scored.candidate.parameters["coefficient"], 0.25)
        self.assertIs(node.node_type, NodeType.PROPERTY)
        self.assertIs(audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(system.graph.validate_grounding(), ())

    def test_discovery_rejects_false_concept_when_no_anomaly_exists(self) -> None:
        system = DiscoverySystem()
        result = system.discover_friction(
            ExperimentLog(synthetic_constant_velocity_records()),
            "block_001",
        )

        self.assertIsNone(result)
        self.assertEqual(len(system.graph.nodes), 0)


if __name__ == "__main__":
    unittest.main()
