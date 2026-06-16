import unittest

from causal_constructivism.models import ActionCandidate, GroundingStatus
from causal_constructivism.system import CausalConstructivismSystem


class SystemTests(unittest.TestCase):
    def test_active_loop_converges_on_mass_and_remains_grounded(self) -> None:
        system = CausalConstructivismSystem(
            true_mass=2.5,
            sensor_noise_std=0.03,
            seed=11,
        )
        actions = [
            ActionCandidate(force=1.0, duration=0.2),
            ActionCandidate(force=3.0, duration=0.2),
            ActionCandidate(force=6.0, duration=0.2),
        ]

        steps = [system.step(actions) for _ in range(5)]

        self.assertLess(abs(steps[-1].mass_mean - 2.5), 0.15)
        self.assertLess(steps[-1].mass_variance, steps[0].mass_variance)
        self.assertIsNot(steps[-1].audit.status, GroundingStatus.UNGROUNDED)
        self.assertTrue(all(step.inference.converged for step in steps))


if __name__ == "__main__":
    unittest.main()
