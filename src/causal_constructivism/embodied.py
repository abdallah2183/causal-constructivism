from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .models import VectorGaussianBelief


Vector3 = tuple[float, float, float]


def _validate_vector3(value: Vector3, name: str) -> None:
    if len(value) != 3 or not all(math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain three finite values")


@dataclass(frozen=True, slots=True)
class ForceAction3D:
    object_id: str
    force: Vector3
    application_point: Vector3 = (0.0, 0.0, 0.0)
    duration: float = 0.1

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("Action object ID is required")
        _validate_vector3(self.force, "Force")
        _validate_vector3(self.application_point, "Application point")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Action duration must be finite and positive")


@dataclass(frozen=True, slots=True)
class BodyState3D:
    object_id: str
    position: Vector3
    velocity: Vector3

    def __post_init__(self) -> None:
        _validate_vector3(self.position, "Position")
        _validate_vector3(self.velocity, "Velocity")


@dataclass(frozen=True, slots=True)
class Contact3D:
    object_a: str
    object_b: str
    position: Vector3
    normal: Vector3
    force: Vector3


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be positive")


@dataclass(frozen=True, slots=True)
class SegmentedObject:
    segment_id: int
    class_name: str
    pixels: tuple[tuple[int, int, float], ...]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.segment_id < 0:
            raise ValueError("Segment ID must be non-negative")
        if not self.class_name:
            raise ValueError("Segment class is required")
        if not self.pixels:
            raise ValueError("Segment must contain pixels")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Segment confidence must be in [0, 1]")
        for u, v, depth in self.pixels:
            if u < 0 or v < 0 or not math.isfinite(depth) or depth <= 0:
                raise ValueError("Segment pixels require valid coordinates and depth")


@dataclass(frozen=True, slots=True)
class RGBDFrame:
    frame_id: str
    timestamp: float
    intrinsics: CameraIntrinsics
    segments: tuple[SegmentedObject, ...]
    camera_id: str = "camera_0"

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("Frame ID is required")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Frame timestamp must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ObjectProposal:
    detection_id: str
    class_name: str
    position: VectorGaussianBelief
    bbox: tuple[int, int, int, int]
    confidence: float
    observation_id: str


@dataclass(frozen=True, slots=True)
class EnvironmentObservation3D:
    timestamp: float
    states: tuple[BodyState3D, ...]
    contacts: tuple[Contact3D, ...]
    frame: RGBDFrame | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentState3D:
    time: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    act: tuple[float, ...]


class Environment3D(Protocol):
    def reset(self) -> None: ...

    def step(self, action: ForceAction3D) -> EnvironmentObservation3D: ...

    def observe(self) -> EnvironmentObservation3D: ...
