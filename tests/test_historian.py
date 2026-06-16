import tempfile
import unittest
from pathlib import Path

from causal_constructivism.historian import (
    HistorianSystem,
    HistoryLog,
    Policy,
    PolicyIntervention,
    run_historian_benchmark,
    synthetic_history_trials,
)
from causal_constructivism.models import EdgeType, GroundingStatus


class HistorianTests(unittest.TestCase):
    def test_replay_stitches_experiment_history_and_keeps_graph_grounded(self) -> None:
        system = HistorianSystem()
        replay = system.replay(
            synthetic_history_trials(count=6, hidden_friction=0.25),
            Policy(name="baseline"),
        )

        experiment_links = [
            edge
            for edge in replay.graph.edges
            if edge.edge_type is EdgeType.EVOLVES_TO
            and replay.graph.require_node(edge.source_id).node_type.value == "experiment"
            and replay.graph.require_node(edge.target_id).node_type.value == "experiment"
        ]

        self.assertEqual(len(replay.experiments), 6)
        self.assertEqual(len(experiment_links), 5)
        self.assertEqual(replay.metrics.ungrounded_nodes, 0)
        self.assertIs(
            replay.metrics.final_mass_grounding.status,
            GroundingStatus.CONFIDENT,
        )

    def test_history_counterfactual_discovers_friction_before_mass(self) -> None:
        result = run_historian_benchmark(
            experiments=6,
            true_mass=2.5,
            hidden_friction=0.25,
        )

        self.assertEqual(result.actual.metrics.experiments_to_friction, 4)
        self.assertEqual(result.counterfactual.metrics.experiments_to_friction, 0)
        self.assertGreater(result.mean_mass_error_reduction, 0.0)
        self.assertGreater(result.prediction_error_reduction, 0.0)
        self.assertEqual(result.experiments_saved_to_friction, 4)
        self.assertEqual(
            [item.trial_id for item in result.actual.experiments],
            [item.trial_id for item in result.counterfactual.experiments],
        )

    def test_stronger_action_policy_reduces_mass_bias_without_friction_model(self) -> None:
        trials = synthetic_history_trials(count=3, hidden_friction=0.25)
        baseline = Policy(
            name="no_discovery",
            discovery_min_records=10,
            use_concepts_for_mass_inference=False,
        )
        result = HistorianSystem().counterfactual(
            trials,
            baseline,
            PolicyIntervention(name="stronger_pushes", action_force_multiplier=2.0),
        )

        self.assertGreater(result.mean_mass_error_reduction, 0.0)
        self.assertEqual(result.actual.metrics.experiments_to_friction, None)
        self.assertEqual(result.counterfactual.metrics.experiments_to_friction, None)

    def test_history_log_round_trip_preserves_shared_exogenous(self) -> None:
        trials = synthetic_history_trials(count=4, hidden_friction=0.18)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            HistoryLog(trials).save_jsonl(path)
            restored = HistoryLog.load_jsonl(path)

        self.assertEqual(restored.trials, trials)


if __name__ == "__main__":
    unittest.main()
