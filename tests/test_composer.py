import unittest

from causal_constructivism.composer import (
    ComposerSystem,
    CompoundModelSelector,
    run_composer_benchmark,
    synthetic_compound_observations,
)
from causal_constructivism.models import GroundingStatus


class ComposerTests(unittest.TestCase):
    def test_pairwise_friction_restitution_model_wins_compound_scene(self) -> None:
        result = run_composer_benchmark(
            friction=0.25,
            restitution=0.65,
            count=8,
        )

        self.assertEqual(result.selected_model.concepts, ("friction", "restitution"))
        self.assertAlmostEqual(
            result.selected_model.parameters["friction.coefficient"],
            0.25,
        )
        self.assertAlmostEqual(
            result.selected_model.parameters["restitution.coefficient"],
            0.65,
        )
        self.assertLess(result.selected_model.residual_error, 1e-12)
        self.assertEqual(result.ungrounded_nodes, 0)

    def test_interaction_node_is_grounded_when_pairwise_model_wins(self) -> None:
        result = run_composer_benchmark()

        self.assertIsNotNone(result.interaction)
        assert result.interaction is not None
        self.assertEqual(
            result.interaction.interaction_type,
            "friction_loss_modulates_restitution",
        )
        self.assertIs(result.interaction.audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(
            {concept.name: concept.audit.status for concept in result.concepts},
            {
                "friction": GroundingStatus.CONFIDENT,
                "restitution": GroundingStatus.CONFIDENT,
            },
        )

    def test_model_competition_prefers_single_concept_when_only_friction_is_needed(self) -> None:
        observations = synthetic_compound_observations(
            friction=0.25,
            restitution=1.0,
            count=8,
        )

        fits = CompoundModelSelector().fit(observations)

        self.assertEqual(fits[0].concepts, ("friction",))
        self.assertLess(fits[0].residual_error, 1e-12)

    def test_composer_rejects_low_evidence_compound_observations(self) -> None:
        observations = synthetic_compound_observations(
            friction=0.0,
            restitution=1.0,
            count=8,
        )

        with self.assertRaises(ValueError):
            ComposerSystem().discover(observations)


if __name__ == "__main__":
    unittest.main()
