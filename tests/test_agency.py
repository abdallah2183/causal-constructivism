import importlib.util
import tempfile
import unittest
from pathlib import Path

from causal_constructivism.agency import ClosedLoopEmbodiedScientist
from causal_constructivism.models import GroundingStatus
from causal_constructivism.persistence import SQLiteGraphStore


class AgencyTests(unittest.TestCase):
    def setUp(self) -> None:
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("MuJoCo is not installed in this runtime")

    def test_closed_loop_inferrs_three_masses_and_persists_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "closed-loop.db"
            system = ClosedLoopEmbodiedScientist(database_path=database)
            try:
                result = system.run(experiments=6)
            finally:
                system.close()
            restored = SQLiteGraphStore(database).load(result.snapshot_id)

        self.assertTrue(result.successful)
        self.assertEqual(set(result.mass_estimates), set(result.true_masses))
        self.assertTrue(
            all(error <= 0.15 for error in result.relative_errors.values())
        )
        self.assertTrue(
            all(
                step.audit.status is GroundingStatus.CONFIDENT
                for step in result.steps
            )
        )
        self.assertEqual(len(restored.nodes), result.node_count)
        self.assertEqual(len(restored.edges), result.edge_count)
        self.assertEqual(restored.validate_grounding(), ())

    def test_planner_candidate_scoring_does_not_mutate_mujoco_state(self) -> None:
        system = ClosedLoopEmbodiedScientist()
        try:
            system.reset()
            before = system.adapter.capture_state()
            _ = system.planner.score(system.planner.candidates()[0])
            after = system.adapter.capture_state()
        finally:
            system.close()

        self.assertEqual(before.time, after.time)
        self.assertEqual(before.qpos, after.qpos)
        self.assertEqual(before.qvel, after.qvel)


if __name__ == "__main__":
    unittest.main()
