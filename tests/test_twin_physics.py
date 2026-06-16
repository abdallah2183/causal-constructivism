import unittest

from causal_constructivism.models import GaussianBelief, ObjectAction
from causal_constructivism.twin_physics import (
    MultiObjectWorld1D,
    ParticleBodyPrior,
    ParticleCollisionFactor,
    RigidBody1D,
)


class TwinPhysicsTests(unittest.TestCase):
    def test_equal_mass_elastic_collision_swaps_velocity(self) -> None:
        world = MultiObjectWorld1D(
            [
                RigidBody1D("a", mass=1.0, radius=0.5, position=0.0, velocity=1.0),
                RigidBody1D("b", mass=1.0, radius=0.5, position=2.0, velocity=0.0),
            ],
            timestep=0.001,
        )

        snapshot = world.simulate(
            ObjectAction("a", force=0.0, duration=0.1),
            horizon=1.5,
        )

        self.assertTrue(snapshot.collisions)
        self.assertAlmostEqual(snapshot.states["a"][1], 0.0, places=6)
        self.assertAlmostEqual(snapshot.states["b"][1], 1.0, places=6)

    def test_particle_factor_represents_collision_branch_probability(self) -> None:
        priors = [
            ParticleBodyPrior(
                "red",
                GaussianBelief(2.0, 0.04),
                GaussianBelief(0.5, 0.0025),
                GaussianBelief(0.0, 0.0025),
                0.5,
                restitution=0.9,
            ),
            ParticleBodyPrior(
                "green",
                GaussianBelief(1.0, 0.01),
                GaussianBelief(3.0, 0.0025),
                GaussianBelief(0.0, 0.0025),
                0.5,
                restitution=0.9,
            ),
        ]
        factor = ParticleCollisionFactor(particles=120, seed=23)
        action = ObjectAction("red", force=10.0, duration=0.5)

        weak = factor.rollout(
            priors,
            action,
            horizon=2.0,
            interventions={"action.force": 2.0},
        )
        threshold = factor.rollout(
            priors,
            action,
            horizon=2.0,
            interventions={"action.force": 3.0},
        )

        self.assertLess(weak.collision_probability, 0.05)
        self.assertGreater(threshold.collision_probability, 0.05)
        self.assertLess(threshold.collision_probability, 0.5)


if __name__ == "__main__":
    unittest.main()

