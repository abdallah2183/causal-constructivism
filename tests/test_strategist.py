import unittest

from causal_constructivism.historian import synthetic_history_trials
from causal_constructivism.models import GroundingStatus
from causal_constructivism.strategist import (
    DiscoveryPolicy,
    PolicyGenerator,
    StrategistSystem,
    run_strategist_benchmark,
)


class StrategistTests(unittest.TestCase):
    def test_policy_generator_produces_better_discovery_candidates(self) -> None:
        base = DiscoveryPolicy(policy_id="baseline", version=1)
        candidates = PolicyGenerator().generate(base)

        self.assertEqual(len(candidates), 3)
        self.assertIn("baseline.fast_discovery", {candidate.policy_id for candidate in candidates})
        self.assertTrue(any(candidate.discovery_min_records < base.discovery_min_records for candidate in candidates))

    def test_strategist_adopts_policy_with_counterfactual_efficiency_gain(self) -> None:
        result = run_strategist_benchmark(
            experiments=6,
            true_mass=2.5,
            hidden_friction=0.25,
        )

        self.assertTrue(result.adopted)
        self.assertEqual(result.baseline.experiments_to_friction, 4)
        self.assertEqual(result.selected.experiments_to_friction, 3)
        self.assertGreaterEqual(result.selected.efficiency_gain, 4 / 3)
        self.assertIs(result.audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(result.ungrounded_nodes, 0)

    def test_strategist_keeps_baseline_when_improvement_threshold_is_not_met(self) -> None:
        trials = synthetic_history_trials(count=6, hidden_friction=0.25)
        base = DiscoveryPolicy(
            policy_id="baseline",
            version=1,
            improvement_threshold=1.0,
        )

        result = StrategistSystem().optimize(trials, base)

        self.assertFalse(result.adopted)
        self.assertEqual(result.selected.policy.policy_id, "baseline")

    def test_policy_evaluation_preserves_grounding_quality(self) -> None:
        trials = synthetic_history_trials(count=6, hidden_friction=0.25)
        base = DiscoveryPolicy(policy_id="baseline", version=1)
        score = StrategistSystem().evaluate_policy(base, trials, baseline=None)

        self.assertEqual(score.confident_fraction, 1.0)
        self.assertEqual(score.grounding_gain, 1.0)


if __name__ == "__main__":
    unittest.main()
