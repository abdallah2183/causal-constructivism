import unittest

from causal_constructivism.models import GroundingStatus
from causal_constructivism.universalist import (
    PatternAbstractor,
    UniversalistSystem,
    run_universalist_benchmark,
    synthetic_harmonic_law_instances,
    synthetic_spurious_law_instances,
)


class UniversalistTests(unittest.TestCase):
    def test_harmonic_instances_abstract_to_common_pattern(self) -> None:
        cluster = PatternAbstractor().extract_harmonic_cluster(
            synthetic_harmonic_law_instances()
        )

        self.assertEqual({instance.domain for instance in cluster}, {"pendulum", "spring"})
        self.assertEqual(
            {instance.structural_signature for instance in cluster},
            {"coefficient*sqrt(inertia/restoring)"},
        )

    def test_universalist_unifies_pendulum_and_spring_and_predicts_lc(self) -> None:
        result = run_universalist_benchmark()

        self.assertEqual(result.principle.name, "harmonic_motion")
        self.assertEqual(result.instance_count, 2)
        self.assertEqual(result.confirmed_predictions, 1)
        self.assertEqual(
            result.principle.predictions[0].predicted_law,
            "T = 2*pi*sqrt(L*C)",
        )
        self.assertTrue(result.principle.predictions[0].confirmed)
        self.assertIs(result.principle.audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(result.ungrounded_nodes, 0)

    def test_spurious_law_instances_do_not_unify(self) -> None:
        with self.assertRaises(ValueError):
            UniversalistSystem().unify_harmonic_motion(
                synthetic_spurious_law_instances()
            )

    def test_untested_prediction_keeps_meta_law_speculative(self) -> None:
        result = UniversalistSystem().unify_harmonic_motion(
            synthetic_harmonic_law_instances(),
            validate_lc=False,
        )

        self.assertEqual(result.confirmed_predictions, 0)
        self.assertIs(result.principle.audit.status, GroundingStatus.SPECULATIVE)


if __name__ == "__main__":
    unittest.main()
