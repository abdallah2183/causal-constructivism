import unittest

from causal_constructivism.embodied import (
    CameraIntrinsics,
    RGBDFrame,
    SegmentedObject,
)
from causal_constructivism.perception import SyntheticPerception


class PerceptionTests(unittest.TestCase):
    def test_rgbd_projection_recovers_camera_frame_position(self) -> None:
        intrinsics = CameraIntrinsics(320, 240, 200.0, 200.0, 160.0, 120.0)
        frame = RGBDFrame(
            "frame",
            0.0,
            intrinsics,
            (
                SegmentedObject(
                    1,
                    "cube",
                    tuple(
                        (180 + du, 110 + dv, 2.0)
                        for du in range(-1, 2)
                        for dv in range(-1, 2)
                    ),
                ),
            ),
        )
        perception = SyntheticPerception(
            depth_noise_base=0.0,
            depth_noise_scale=0.0,
            pixel_noise_std=0.0,
        )

        proposal = perception.perceive(frame)[0]

        self.assertAlmostEqual(proposal.position.means[0], 0.2, places=6)
        self.assertAlmostEqual(proposal.position.means[1], -0.1, places=6)
        self.assertAlmostEqual(proposal.position.means[2], 2.0, places=6)
        self.assertTrue(all(value > 0 for value in proposal.position.variances))


if __name__ == "__main__":
    unittest.main()

