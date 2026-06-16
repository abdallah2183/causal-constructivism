import unittest
from dataclasses import dataclass
from typing import Any

from causal_constructivism.integrator import CognitiveOrchestrator
from causal_constructivism.integrator_system import run_integrator_benchmark, IntegratorSystem
from causal_constructivism.models import GroundingStatus, GaussianBelief
from causal_constructivism.graph import CausalGraph


@dataclass
class MockAgencyResult:
    mass_estimates: dict[str, tuple[float, float]]
    true_masses: dict[str, float]
    relative_errors: dict[str, float]
    successful: bool


class MockAgency:
    def __init__(self) -> None:
        self.run_called = 0

    def run(self, experiments: int = 1) -> MockAgencyResult:
        self.run_called += 1
        return MockAgencyResult(
            mass_estimates={"red_cube": (0.35, 0.02)},
            true_masses={"red_cube": 0.35},
            relative_errors={"red_cube": 0.0},
            successful=True,
        )


class IntegratorTests(unittest.TestCase):
    def test_integrator_benchmark_executes_successfully(self) -> None:
        result = run_integrator_benchmark()

        self.assertEqual(result.steps_count, 6)
        self.assertEqual(result.grounding_quality, 1.0)
        self.assertEqual(result.status, GroundingStatus.CONFIDENT)
        
        # Verify the steps phase names
        step_phases = [step.phase_name for step in result.result.steps]
        self.assertIn("Agency & Perception", step_phases)
        self.assertIn("Composer & Discoverer", step_phases)
        self.assertIn("Universalist Abstraction", step_phases)
        self.assertIn("Strategist Policy", step_phases)
        self.assertIn("Curator Gap Detection", step_phases)
        self.assertIn("Collaborator Discourse", step_phases)

        # Verify ontology elements
        ontology = result.result.final_ontology
        self.assertIn("concept.friction", ontology.nodes)
        self.assertIn("concept.restitution", ontology.nodes)
        self.assertIn("concept.damping", ontology.nodes)
        self.assertIn("meta_law.harmonic_motion", ontology.nodes)

    def test_integrator_wires_agency_simulator_rollout(self) -> None:
        agency = MockAgency()
        system = IntegratorSystem(agency=agency)
        benchmark_res = system.run_benchmark()

        self.assertEqual(agency.run_called, 2)  # One in agency phase, one in debate phase
        self.assertEqual(benchmark_res.steps_count, 6)
        self.assertEqual(benchmark_res.status, GroundingStatus.CONFIDENT)

        # Verify agent belief has been updated with values from MockAgency (mean=0.35, var=0.04)
        debate_record = benchmark_res.result.debate_records[0]
        # In the debate round, Agent_A and Agent_B beliefs are updated:
        round_1 = debate_record.rounds[0]
        self.assertIn("Agent_A", round_1.belief_updates)
        agent_a_updates = round_1.belief_updates["Agent_A"]
        self.assertAlmostEqual(agent_a_updates["concept.friction"].mean, 0.35)
        self.assertAlmostEqual(agent_a_updates["concept.friction"].variance, 0.0004)




if __name__ == "__main__":
    unittest.main()
