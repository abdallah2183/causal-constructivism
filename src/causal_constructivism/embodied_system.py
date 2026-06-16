from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audit import MetacognitiveAuditor
from .embodied import (
    CameraIntrinsics,
    ObjectProposal,
    RGBDFrame,
    SegmentedObject,
)
from .models import GroundingAudit
from .perception import SyntheticPerception
from .persistence import SQLiteGraphStore
from .temporal import TemporalGraphIntegrator
from .tracking import ObjectTracker, TrackingUpdate


@dataclass(frozen=True, slots=True)
class EmbodiedFrameResult:
    frame_id: str
    proposals: tuple[ObjectProposal, ...]
    tracking: TrackingUpdate
    state_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmbodiedDemoResult:
    frames: tuple[EmbodiedFrameResult, ...]
    persistent_track_ids: tuple[str, ...]
    node_count: int
    edge_count: int
    final_audits: tuple[GroundingAudit, ...]
    snapshot_id: int | None


class EmbodiedVisionSystem:
    """Synthetic RGB-D, permanence, temporal graph, and persistence pipeline."""

    def __init__(
        self,
        *,
        database_path: str | Path | None = None,
        maximum_misses: int = 5,
    ) -> None:
        self.perception = SyntheticPerception(seed=31)
        self.tracker = ObjectTracker(
            maximum_misses=maximum_misses,
            association_threshold=25.0,
        )
        self.temporal = TemporalGraphIntegrator()
        self.store = (
            SQLiteGraphStore(database_path) if database_path is not None else None
        )

    @property
    def graph(self):
        return self.temporal.graph

    def process(self, frame: RGBDFrame) -> EmbodiedFrameResult:
        proposals = self.perception.perceive(frame)
        tracking = self.tracker.update(
            proposals,
            timestamp=frame.timestamp,
        )
        states = self.temporal.integrate(frame, proposals, tracking)
        return EmbodiedFrameResult(
            frame_id=frame.frame_id,
            proposals=proposals,
            tracking=tracking,
            state_node_ids=tuple(node.id for node in states),
        )

    def run(
        self,
        frames: tuple[RGBDFrame, ...],
        *,
        snapshot_label: str = "embodied-vision",
    ) -> EmbodiedDemoResult:
        results = tuple(self.process(frame) for frame in frames)
        audits = tuple(
            MetacognitiveAuditor(self.graph).audit(
                self.temporal.latest_state(track.track_id).id
            )
            for track in self.tracker.tracks
        )
        snapshot_id = (
            self.store.save(self.graph, label=snapshot_label)
            if self.store is not None
            else None
        )
        return EmbodiedDemoResult(
            frames=results,
            persistent_track_ids=tuple(
                track.track_id for track in self.tracker.tracks
            ),
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            final_audits=audits,
            snapshot_id=snapshot_id,
        )


def synthetic_occlusion_frames() -> tuple[RGBDFrame, ...]:
    intrinsics = CameraIntrinsics(
        width=320,
        height=240,
        fx=240.0,
        fy=240.0,
        cx=160.0,
        cy=120.0,
    )
    positions = (
        {"red_cube": (-0.30, 0.0, 3.0), "blue_cube": (0.50, 0.0, 3.2)},
        {"red_cube": (-0.20, 0.0, 3.0), "blue_cube": (0.50, 0.0, 3.2)},
        {"blue_cube": (0.50, 0.0, 3.2)},
        {"red_cube": (0.00, 0.0, 3.0), "blue_cube": (0.50, 0.0, 3.2)},
        {"red_cube": (0.10, 0.0, 3.0), "blue_cube": (0.50, 0.0, 3.2)},
    )
    frames = []
    for index, objects in enumerate(positions):
        segments = []
        for segment_id, (class_name, position) in enumerate(
            sorted(objects.items())
        ):
            segments.append(
                _segment_from_point(
                    segment_id,
                    class_name,
                    position,
                    intrinsics,
                )
            )
        frames.append(
            RGBDFrame(
                frame_id=f"frame_{index:03d}",
                timestamp=index * 0.1,
                intrinsics=intrinsics,
                segments=tuple(segments),
            )
        )
    return tuple(frames)


def _segment_from_point(
    segment_id: int,
    class_name: str,
    position: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
) -> SegmentedObject:
    x, y, depth = position
    center_u = round(intrinsics.fx * x / depth + intrinsics.cx)
    center_v = round(intrinsics.fy * y / depth + intrinsics.cy)
    pixels = tuple(
        (center_u + offset_u, center_v + offset_v, depth)
        for offset_u in range(-2, 3)
        for offset_v in range(-2, 3)
    )
    return SegmentedObject(
        segment_id=segment_id,
        class_name=class_name,
        pixels=pixels,
        confidence=0.98,
    )

