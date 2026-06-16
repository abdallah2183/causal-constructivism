from __future__ import annotations

import math
import random

from .embodied import ObjectProposal, RGBDFrame, SegmentedObject
from .models import VectorGaussianBelief


class SyntheticPerception:
    """Projects segmented RGB-D pixels into uncertain camera-frame proposals."""

    def __init__(
        self,
        *,
        depth_noise_base: float = 0.005,
        depth_noise_scale: float = 0.002,
        pixel_noise_std: float = 0.25,
        seed: int = 31,
    ) -> None:
        if depth_noise_base < 0 or depth_noise_scale < 0:
            raise ValueError("Depth noise must be non-negative")
        if pixel_noise_std < 0:
            raise ValueError("Pixel noise must be non-negative")
        self.depth_noise_base = depth_noise_base
        self.depth_noise_scale = depth_noise_scale
        self.pixel_noise_std = pixel_noise_std
        self._random = random.Random(seed)

    def perceive(self, frame: RGBDFrame) -> tuple[ObjectProposal, ...]:
        return tuple(
            self._proposal(frame, segment) for segment in frame.segments
        )

    def _proposal(
        self,
        frame: RGBDFrame,
        segment: SegmentedObject,
    ) -> ObjectProposal:
        intrinsics = frame.intrinsics
        points: list[tuple[float, float, float]] = []
        minimum_u = intrinsics.width
        minimum_v = intrinsics.height
        maximum_u = 0
        maximum_v = 0
        for u, v, measured_depth in segment.pixels:
            depth_std = (
                self.depth_noise_base
                + self.depth_noise_scale * measured_depth
            )
            depth = max(
                1e-6,
                measured_depth + self._random.gauss(0.0, depth_std),
            )
            noisy_u = u + self._random.gauss(0.0, self.pixel_noise_std)
            noisy_v = v + self._random.gauss(0.0, self.pixel_noise_std)
            x = (noisy_u - intrinsics.cx) * depth / intrinsics.fx
            y = (noisy_v - intrinsics.cy) * depth / intrinsics.fy
            points.append((x, y, depth))
            minimum_u = min(minimum_u, u)
            minimum_v = min(minimum_v, v)
            maximum_u = max(maximum_u, u)
            maximum_v = max(maximum_v, v)

        means = tuple(
            sum(point[index] for point in points) / len(points)
            for index in range(3)
        )
        sample_variances = tuple(
            sum((point[index] - means[index]) ** 2 for point in points)
            / len(points)
            for index in range(3)
        )
        mean_depth = means[2]
        depth_std = self.depth_noise_base + self.depth_noise_scale * mean_depth
        projection_variance = (
            self.pixel_noise_std * mean_depth / intrinsics.fx
        ) ** 2
        variances = (
            max(sample_variances[0] / len(points) + projection_variance, 1e-9),
            max(sample_variances[1] / len(points) + projection_variance, 1e-9),
            max(sample_variances[2] / len(points) + depth_std**2, 1e-9),
        )
        confidence = segment.confidence / (
            1.0 + math.sqrt(sum(variances))
        )
        return ObjectProposal(
            detection_id=f"{frame.frame_id}.segment_{segment.segment_id}",
            class_name=segment.class_name,
            position=VectorGaussianBelief(means, variances),
            bbox=(minimum_u, minimum_v, maximum_u, maximum_v),
            confidence=max(0.0, min(1.0, confidence)),
            observation_id=frame.frame_id,
        )

