from __future__ import annotations

import math

from .embodied import ObjectProposal, RGBDFrame
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, Node, NodeType
from .tracking import TrackingUpdate


class TemporalGraphIntegrator:
    """Materializes persistent identities and state chains from tracker output."""

    def __init__(self, graph: CausalGraph | None = None) -> None:
        self.graph = graph or CausalGraph()
        self._object_nodes: dict[str, str] = {}
        self._latest_state_nodes: dict[str, str] = {}
        self._sequence = 0

    def integrate(
        self,
        frame: RGBDFrame,
        proposals: tuple[ObjectProposal, ...],
        update: TrackingUpdate,
    ) -> tuple[Node, ...]:
        self._sequence += 1
        proposals_by_id = {
            proposal.detection_id: proposal for proposal in proposals
        }
        created_states: list[Node] = []
        for track in update.tracks:
            object_node = self._object_node(track.track_id, track.class_name)
            position_norm = math.sqrt(
                sum(component * component for component in track.position.means)
            )
            state_variance = max(sum(track.position.variances), 1e-9)
            state = self.graph.add_node(
                f"{track.track_id}.state.{self._sequence:06d}",
                NodeType.STATE,
                GaussianBelief(position_norm, state_variance),
                modality="temporal_3d",
                metadata={
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "timestamp": frame.timestamp,
                    "frame_id": frame.frame_id,
                    "status": track.status,
                    "position_mean": track.position.means,
                    "position_variance": track.position.variances,
                    "velocity_mean": track.velocity.means,
                    "velocity_variance": track.velocity.variances,
                },
            )
            self.graph.add_edge(
                object_node.id,
                state.id,
                EdgeType.HAS_STATE,
                noise_variance=state_variance,
            )
            previous_state_id = self._latest_state_nodes.get(track.track_id)
            if previous_state_id is not None:
                self.graph.add_edge(
                    previous_state_id,
                    state.id,
                    EdgeType.EVOLVES_TO,
                    noise_variance=state_variance,
                    confidence=max(0.1, 1.0 / (1.0 + state_variance)),
                )

            proposal = (
                proposals_by_id.get(track.last_detection_id)
                if track.status == "visible"
                else None
            )
            if proposal is not None:
                evidence = GaussianBelief(
                    math.sqrt(
                        sum(
                            component * component
                            for component in proposal.position.means
                        )
                    ),
                    max(sum(proposal.position.variances), 1e-9),
                )
                observation = self.graph.add_node(
                    f"{frame.frame_id}.{track.track_id}.observation",
                    NodeType.OBSERVATION,
                    evidence,
                    evidence=evidence,
                    modality="synthetic_rgbd",
                    metadata={
                        "frame_id": frame.frame_id,
                        "camera_id": frame.camera_id,
                        "detection_id": proposal.detection_id,
                        "bbox": proposal.bbox,
                        "class_name": proposal.class_name,
                        "confidence": proposal.confidence,
                    },
                )
                self.graph.add_edge(
                    observation.id,
                    state.id,
                    EdgeType.OBSERVES,
                    noise_variance=evidence.variance,
                    confidence=proposal.confidence,
                )
                if track.track_id in update.born_track_ids:
                    self.graph.add_edge(
                        observation.id,
                        object_node.id,
                        EdgeType.OBSERVES,
                        noise_variance=evidence.variance,
                        confidence=proposal.confidence,
                    )

            self._latest_state_nodes[track.track_id] = state.id
            created_states.append(state)
        return tuple(created_states)

    def latest_state(self, track_id: str) -> Node:
        try:
            return self.graph.require_node(self._latest_state_nodes[track_id])
        except KeyError as exc:
            raise KeyError(f"Unknown track state: {track_id}") from exc

    def object_node(self, track_id: str) -> Node:
        try:
            return self.graph.require_node(self._object_nodes[track_id])
        except KeyError as exc:
            raise KeyError(f"Unknown persistent object: {track_id}") from exc

    def _object_node(self, track_id: str, class_name: str) -> Node:
        node_id = self._object_nodes.get(track_id)
        if node_id is not None:
            return self.graph.require_node(node_id)
        node = self.graph.add_node(
            track_id,
            NodeType.OBJECT,
            GaussianBelief(1.0, 1e-6),
            modality="persistent_identity",
            metadata={
                "track_id": track_id,
                "class_name": class_name,
                "tracking_status": "active",
            },
        )
        self._object_nodes[track_id] = node.id
        return node

