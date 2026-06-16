import tempfile
import unittest
from pathlib import Path

from causal_constructivism.embodied_system import (
    EmbodiedVisionSystem,
    synthetic_occlusion_frames,
)
from causal_constructivism.models import GroundingStatus
from causal_constructivism.persistence import SQLiteGraphStore


class EmbodiedVisionTests(unittest.TestCase):
    def test_occlusion_recovers_same_identity_without_false_birth(self) -> None:
        system = EmbodiedVisionSystem()

        result = system.run(synthetic_occlusion_frames())

        self.assertEqual(result.persistent_track_ids, ("object_0001", "object_0002"))
        self.assertEqual(
            sum(len(frame.tracking.born_track_ids) for frame in result.frames),
            2,
        )
        self.assertEqual(
            sum(len(frame.tracking.occluded_track_ids) for frame in result.frames),
            1,
        )
        self.assertEqual(
            sum(len(frame.tracking.lost_track_ids) for frame in result.frames),
            0,
        )
        self.assertTrue(
            all(
                audit.status is not GroundingStatus.UNGROUNDED
                for audit in result.final_audits
            )
        )

    def test_temporal_graph_is_persisted_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "embodied.db"
            system = EmbodiedVisionSystem(database_path=database)
            result = system.run(synthetic_occlusion_frames())
            restored = SQLiteGraphStore(database).load(result.snapshot_id)

        self.assertEqual(len(restored.nodes), result.node_count)
        self.assertEqual(len(restored.edges), result.edge_count)
        self.assertEqual(restored.validate_grounding(), ())


if __name__ == "__main__":
    unittest.main()

