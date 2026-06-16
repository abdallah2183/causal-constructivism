from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .embodied import ObjectProposal, Vector3
from .models import VectorGaussianBelief


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    class_name: str
    position: VectorGaussianBelief
    velocity: VectorGaussianBelief
    last_timestamp: float
    age: int = 1
    misses: int = 0
    status: str = "visible"
    last_detection_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    tracks: tuple[Track, ...]
    matched_track_ids: tuple[str, ...]
    born_track_ids: tuple[str, ...]
    occluded_track_ids: tuple[str, ...]
    lost_track_ids: tuple[str, ...]


class ObjectTracker:
    """Constant-velocity tracker with optimal global data association."""

    def __init__(
        self,
        *,
        association_threshold: float = 16.0,
        process_variance: float = 0.01,
        maximum_misses: int = 5,
    ) -> None:
        if association_threshold <= 0 or process_variance <= 0:
            raise ValueError("Tracker variances and thresholds must be positive")
        if maximum_misses < 1:
            raise ValueError("Maximum misses must be positive")
        self.association_threshold = association_threshold
        self.process_variance = process_variance
        self.maximum_misses = maximum_misses
        self._tracks: dict[str, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(
            track
            for track in self._tracks.values()
            if track.status != "lost"
        )

    def update(
        self,
        proposals: tuple[ObjectProposal, ...],
        *,
        timestamp: float,
    ) -> TrackingUpdate:
        active = sorted(self.tracks, key=lambda track: track.track_id)
        predicted = [self._predict(track, timestamp) for track in active]
        costs = [
            [self._association_cost(track, proposal) for proposal in proposals]
            for track in predicted
        ]
        assignments = _minimum_cost_assignment(
            costs,
            unassigned_cost=self.association_threshold,
        )

        assigned_tracks: set[int] = set()
        assigned_proposals: set[int] = set()
        matched: list[str] = []
        for track_index, proposal_index in assignments:
            cost = costs[track_index][proposal_index]
            if cost > self.association_threshold:
                continue
            track = self._correct(
                predicted[track_index],
                proposals[proposal_index],
                timestamp,
            )
            self._tracks[track.track_id] = track
            assigned_tracks.add(track_index)
            assigned_proposals.add(proposal_index)
            matched.append(track.track_id)

        occluded: list[str] = []
        lost: list[str] = []
        for index, track in enumerate(predicted):
            if index in assigned_tracks:
                continue
            misses = track.misses + 1
            status = "lost" if misses > self.maximum_misses else "occluded"
            updated = replace(track, misses=misses, status=status)
            self._tracks[track.track_id] = updated
            (lost if status == "lost" else occluded).append(track.track_id)

        born: list[str] = []
        for index, proposal in enumerate(proposals):
            if index in assigned_proposals:
                continue
            track = self._birth(proposal, timestamp)
            self._tracks[track.track_id] = track
            born.append(track.track_id)

        return TrackingUpdate(
            tracks=self.tracks,
            matched_track_ids=tuple(matched),
            born_track_ids=tuple(born),
            occluded_track_ids=tuple(occluded),
            lost_track_ids=tuple(lost),
        )

    def _predict(self, track: Track, timestamp: float) -> Track:
        elapsed = max(0.0, timestamp - track.last_timestamp)
        means = tuple(
            position + velocity * elapsed
            for position, velocity in zip(
                track.position.means,
                track.velocity.means,
                strict=True,
            )
        )
        variances = tuple(
            position_variance
            + elapsed * elapsed * velocity_variance
            + self.process_variance * max(elapsed, 1e-6)
            for position_variance, velocity_variance in zip(
                track.position.variances,
                track.velocity.variances,
                strict=True,
            )
        )
        return replace(
            track,
            position=VectorGaussianBelief(means, variances),
            last_timestamp=timestamp,
            age=track.age + 1,
        )

    def _correct(
        self,
        predicted: Track,
        proposal: ObjectProposal,
        timestamp: float,
    ) -> Track:
        previous = self._tracks[predicted.track_id]
        elapsed = max(timestamp - previous.last_timestamp, 1e-6)
        position_means: list[float] = []
        position_variances: list[float] = []
        velocity_means: list[float] = []
        velocity_variances: list[float] = []
        for index in range(3):
            prior_mean = predicted.position.means[index]
            prior_variance = predicted.position.variances[index]
            observed_mean = proposal.position.means[index]
            observed_variance = proposal.position.variances[index]
            gain = prior_variance / (prior_variance + observed_variance)
            corrected_mean = prior_mean + gain * (observed_mean - prior_mean)
            corrected_variance = max((1 - gain) * prior_variance, 1e-9)
            measured_velocity = (
                corrected_mean - previous.position.means[index]
            ) / elapsed
            velocity_variance = (
                corrected_variance + previous.position.variances[index]
            ) / (elapsed * elapsed)
            velocity_gain = predicted.velocity.variances[index] / (
                predicted.velocity.variances[index] + velocity_variance
            )
            corrected_velocity = predicted.velocity.means[index] + velocity_gain * (
                measured_velocity - predicted.velocity.means[index]
            )
            position_means.append(corrected_mean)
            position_variances.append(corrected_variance)
            velocity_means.append(corrected_velocity)
            velocity_variances.append(
                max(
                    (1 - velocity_gain) * predicted.velocity.variances[index],
                    1e-9,
                )
            )
        return replace(
            predicted,
            position=VectorGaussianBelief(
                tuple(position_means),
                tuple(position_variances),
            ),
            velocity=VectorGaussianBelief(
                tuple(velocity_means),
                tuple(velocity_variances),
            ),
            misses=0,
            status="visible",
            last_detection_id=proposal.detection_id,
        )

    def _association_cost(
        self,
        track: Track,
        proposal: ObjectProposal,
    ) -> float:
        if track.class_name != proposal.class_name:
            return math.inf
        return sum(
            (observed - predicted) ** 2
            / max(observed_variance + predicted_variance, 1e-9)
            for observed, predicted, observed_variance, predicted_variance in zip(
                proposal.position.means,
                track.position.means,
                proposal.position.variances,
                track.position.variances,
                strict=True,
            )
        )

    def _birth(self, proposal: ObjectProposal, timestamp: float) -> Track:
        track_id = f"object_{self._next_id:04d}"
        self._next_id += 1
        return Track(
            track_id=track_id,
            class_name=proposal.class_name,
            position=proposal.position,
            velocity=VectorGaussianBelief(
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            ),
            last_timestamp=timestamp,
            last_detection_id=proposal.detection_id,
        )


def _minimum_cost_assignment(
    costs: list[list[float]],
    *,
    unassigned_cost: float,
) -> tuple[tuple[int, int], ...]:
    """Exact assignment with dummy columns; intended for small visual scenes."""
    row_count = len(costs)
    column_count = len(costs[0]) if row_count else 0
    if row_count == 0 or column_count == 0:
        return ()
    memo: dict[tuple[int, int], tuple[float, tuple[tuple[int, int], ...]]] = {}

    def solve(
        row: int,
        used_columns: int,
    ) -> tuple[float, tuple[tuple[int, int], ...]]:
        key = (row, used_columns)
        if key in memo:
            return memo[key]
        if row == row_count:
            return (0.0, ())
        best_cost, best_pairs = solve(row + 1, used_columns)
        best_cost += unassigned_cost
        for column in range(column_count):
            mask = 1 << column
            if used_columns & mask:
                continue
            remainder_cost, remainder_pairs = solve(
                row + 1,
                used_columns | mask,
            )
            candidate_cost = costs[row][column] + remainder_cost
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_pairs = ((row, column),) + remainder_pairs
        memo[key] = (best_cost, best_pairs)
        return memo[key]

    return solve(0, 0)[1]
