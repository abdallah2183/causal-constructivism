import importlib.util
import unittest
from pathlib import Path

from causal_constructivism.embodied import ForceAction3D
from causal_constructivism.mujoco_adapter import (
    MuJoCoAdapter,
    MuJoCoUnavailableError,
)


class MuJoCoAdapterTests(unittest.TestCase):
    def test_missing_optional_dependency_has_actionable_error(self) -> None:
        if importlib.util.find_spec("mujoco") is not None:
            self.skipTest("MuJoCo is installed in this runtime")
        with self.assertRaisesRegex(MuJoCoUnavailableError, "optional 'mujoco'"):
            MuJoCoAdapter(
                "<mujoco/>",
                body_names=(),
                scene_is_xml=True,
            )

    def test_native_scene_steps_and_renders_when_installed(self) -> None:
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("MuJoCo is not installed in this runtime")
        scene = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "scenes"
            / "minimal_blocks.xml"
        )
        adapter = MuJoCoAdapter(
            scene,
            body_names=("red_block", "blue_block"),
            render_width=80,
            render_height=60,
        )
        try:
            adapter.reset()
            before = adapter.observe()
            after = adapter.step(
                ForceAction3D(
                    "red_block",
                    (20.0, 0.0, 0.0),
                    duration=0.1,
                )
            )
            rgb, depth, segmentation = adapter.render_raw()
        finally:
            adapter.close()

        before_red = next(
            state for state in before.states if state.object_id == "red_block"
        )
        after_red = next(
            state for state in after.states if state.object_id == "red_block"
        )
        self.assertGreater(after_red.position[0], before_red.position[0])
        self.assertEqual(rgb.shape, (60, 80, 3))
        self.assertEqual(depth.shape, (60, 80))
        self.assertEqual(segmentation.shape, (60, 80, 2))


if __name__ == "__main__":
    unittest.main()
