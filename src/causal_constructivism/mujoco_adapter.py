from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .embodied import (
    BodyState3D,
    CameraIntrinsics,
    Contact3D,
    EnvironmentObservation3D,
    EnvironmentState3D,
    ForceAction3D,
    RGBDFrame,
    SegmentedObject,
)


class MuJoCoUnavailableError(RuntimeError):
    pass


class MuJoCoAdapter:
    """Optional MuJoCo environment using the official native Python bindings."""

    def __init__(
        self,
        scene: str | Path,
        *,
        body_names: tuple[str, ...],
        camera_id: str | int = "camera_0",
        render_width: int = 640,
        render_height: int = 480,
        scene_is_xml: bool = False,
    ) -> None:
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise MuJoCoUnavailableError(
                "MuJoCo support requires the optional 'mujoco' package"
            ) from exc
        self._mujoco = mujoco
        self._np = np
        self.body_names = body_names
        self.camera_id = camera_id
        self.render_width = render_width
        self.render_height = render_height
        self.model = (
            mujoco.MjModel.from_xml_string(str(scene))
            if scene_is_xml
            else mujoco.MjModel.from_xml_path(str(Path(scene)))
        )
        self.data = mujoco.MjData(self.model)
        self._renderer: Any | None = None
        for body_name in body_names:
            self.model.body(body_name)

    def reset(self) -> None:
        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

    def step(self, action: ForceAction3D) -> EnvironmentObservation3D:
        body = self.model.body(action.object_id)
        body_id = int(body.id)
        timestep = float(self.model.opt.timestep)
        steps = max(1, math.ceil(action.duration / timestep))
        body_position = self._np.asarray(self.data.body(action.object_id).xpos)
        application_point = self._np.asarray(action.application_point)
        force = self._np.asarray(action.force)
        torque = self._np.cross(application_point - body_position, force)
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = torque
        self._mujoco.mj_step(self.model, self.data, nstep=steps)
        self.data.xfrc_applied[body_id] = 0
        return self.observe()

    def observe(self) -> EnvironmentObservation3D:
        states = []
        for body_name in self.body_names:
            body_data = self.data.body(body_name)
            states.append(
                BodyState3D(
                    object_id=body_name,
                    position=tuple(float(item) for item in body_data.xpos),
                    velocity=tuple(float(item) for item in body_data.cvel[3:]),
                )
            )
        contacts = []
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            if body_a == 0 or body_b == 0:
                continue
            force_local = self._np.zeros(6, dtype=float)
            self._mujoco.mj_contactForce(
                self.model,
                self.data,
                index,
                force_local,
            )
            frame = self._np.asarray(contact.frame).reshape(3, 3)
            force_world = frame.T @ force_local[:3]
            contacts.append(
                Contact3D(
                    object_a=self.model.body(body_a).name,
                    object_b=self.model.body(body_b).name,
                    position=tuple(float(item) for item in contact.pos),
                    normal=tuple(float(item) for item in frame[0]),
                    force=tuple(float(item) for item in force_world),
                )
            )
        return EnvironmentObservation3D(
            timestamp=float(self.data.time),
            states=tuple(states),
            contacts=tuple(contacts),
        )

    def capture_state(self) -> EnvironmentState3D:
        return EnvironmentState3D(
            time=float(self.data.time),
            qpos=tuple(float(item) for item in self.data.qpos),
            qvel=tuple(float(item) for item in self.data.qvel),
            act=tuple(float(item) for item in self.data.act),
        )

    def restore_state(self, state: EnvironmentState3D) -> None:
        if len(state.qpos) != self.model.nq or len(state.qvel) != self.model.nv:
            raise ValueError("State dimensions do not match the MuJoCo model")
        if len(state.act) != self.model.na:
            raise ValueError("Actuator state dimensions do not match the model")
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if self.model.na:
            self.data.act[:] = state.act
        self.data.qacc_warmstart[:] = 0
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self._mujoco.mj_forward(self.model, self.data)

    def simulate(
        self,
        action: ForceAction3D,
    ) -> EnvironmentObservation3D:
        state = self.capture_state()
        try:
            return self.step(action)
        finally:
            self.restore_state(state)

    def render_raw(self) -> tuple[Any, Any, Any]:
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )
        renderer = self._renderer
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(self.data, camera=self.camera_id)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        return rgb, depth, segmentation

    def render_frame(
        self,
        *,
        frame_id: str,
        class_names: dict[str, str] | None = None,
        pixel_stride: int = 3,
    ) -> RGBDFrame:
        if pixel_stride <= 0:
            raise ValueError("Pixel stride must be positive")
        _, depth, segmentation = self.render_raw()
        class_names = class_names or {}
        camera = self.model.camera(self.camera_id)
        field_of_view = math.radians(float(self.model.cam_fovy[camera.id]))
        fy = 0.5 * self.render_height / math.tan(0.5 * field_of_view)
        fx = fy
        intrinsics = CameraIntrinsics(
            width=self.render_width,
            height=self.render_height,
            fx=fx,
            fy=fy,
            cx=(self.render_width - 1) / 2,
            cy=(self.render_height - 1) / 2,
        )
        grouped: dict[int, list[tuple[int, int, float]]] = {}
        object_type = self._mujoco.mjtObj.mjOBJ_GEOM
        for v in range(0, self.render_height, pixel_stride):
            for u in range(0, self.render_width, pixel_stride):
                object_id = int(segmentation[v, u, 0])
                segment_type = int(segmentation[v, u, 1])
                if segment_type != object_type or object_id < 0:
                    continue
                body_id = int(self.model.geom_bodyid[object_id])
                if body_id == 0:
                    continue
                measured_depth = float(depth[v, u])
                if not math.isfinite(measured_depth) or measured_depth <= 0:
                    continue
                grouped.setdefault(body_id, []).append((u, v, measured_depth))

        segments = []
        for segment_id, (body_id, pixels) in enumerate(sorted(grouped.items())):
            body_name = self.model.body(body_id).name
            if body_name not in self.body_names:
                continue
            segments.append(
                SegmentedObject(
                    segment_id=segment_id,
                    class_name=class_names.get(body_name, body_name),
                    pixels=tuple(pixels),
                    confidence=1.0,
                )
            )
        return RGBDFrame(
            frame_id=frame_id,
            timestamp=float(self.data.time),
            intrinsics=intrinsics,
            segments=tuple(segments),
            camera_id=str(self.camera_id),
        )

    def close(self) -> None:
        if self._renderer is not None:
            close = getattr(self._renderer, "close", None)
            if close is not None:
                close()
            self._renderer = None
