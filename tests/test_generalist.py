import tempfile
import unittest
from pathlib import Path

from causal_constructivism.discovery import (
    ExperimentLog,
    ExperimentRecord,
    synthetic_sliding_records,
)
from causal_constructivism.generalist import (
    Concept,
    ConceptLibrary,
    GeneralistSystem,
    run_generalist_benchmark,
)
from causal_constructivism.models import GroundingStatus


def _records_for_surface(friction: float, surface_id: str):
    return tuple(
        ExperimentRecord(
            object_id=record.object_id,
            initial_velocity=record.initial_velocity,
            final_velocity=record.final_velocity,
            duration=record.duration,
            predicted_final_velocity=record.predicted_final_velocity,
            surface_id=surface_id,
        )
        for record in synthetic_sliding_records(
            object_id=f"block_{surface_id}",
            friction=friction,
            count=8,
        )
    )


class GeneralistTests(unittest.TestCase):
    def test_concept_library_round_trip_preserves_revision_history(self) -> None:
        result = run_generalist_benchmark(source_friction=0.30, target_friction=0.05)
        library = ConceptLibrary([result.revised_concept])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.json"
            library.save(path)
            restored = ConceptLibrary.load(path)

        concept = restored.require("friction")
        self.assertAlmostEqual(concept.parameter_value, 0.05)
        self.assertEqual(len(concept.revision_history), 1)

    def test_transfer_and_revision_refines_friction_for_new_surface(self) -> None:
        system = GeneralistSystem()
        system.learn_friction_concept(coefficient=0.30, surface_id="wood")

        result = system.transfer_and_revise(
            target_object_id="block_ice",
            environment_name="ice",
            log=ExperimentLog(_records_for_surface(0.05, "ice")),
        )

        self.assertAlmostEqual(result.revised_concept.parameter_value, 0.05)
        self.assertGreater(result.pre_revision_error, result.post_revision_error)
        self.assertLess(result.post_revision_error, 1e-9)
        self.assertIsNot(result.transfer_audit.status, GroundingStatus.UNGROUNDED)
        self.assertIs(result.revision_audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(system.graph.validate_grounding(), ())

    def test_revision_engine_does_not_revise_when_transfer_already_fits(self) -> None:
        system = GeneralistSystem()
        concept = system.learn_friction_concept(coefficient=0.30, surface_id="wood")
        transfer = system.transfer_engine.transfer(
            system.graph,
            concept,
            target_object_id="block_wood",
            environment_name="wood",
        )

        revision = system.revision_engine.revise(
            system.graph,
            concept,
            property_node_id=transfer.property_node_id,
            log=ExperimentLog(_records_for_surface(0.30, "wood")),
            environment_name="wood",
        )

        self.assertIsNone(revision)

    def test_concept_matching_ignores_surface_for_transfer(self) -> None:
        library = ConceptLibrary(
            [
                Concept(
                    name="friction",
                    parameter_name="coefficient",
                    parameter_value=0.3,
                    applicability_signature={
                        "motion_type": "sliding",
                        "surface_interaction": True,
                        "deceleration": True,
                        "surface_id": "wood",
                    },
                    confidence=0.95,
                )
            ]
        )
        system = GeneralistSystem(library)
        signature = system.transfer_engine.environment_signature(
            ExperimentLog(_records_for_surface(0.05, "ice"))
        )

        matches = system.transfer_engine.match(signature)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].concept.name, "friction")


if __name__ == "__main__":
    unittest.main()
