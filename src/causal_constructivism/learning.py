from __future__ import annotations

import math
from collections import defaultdict

from .graph import CausalGraph
from .models import EdgeType, StructureProposal


class StructureLearner:
    """Fits sparse linear causal candidates from paired observation history."""

    def __init__(
        self,
        graph: CausalGraph,
        *,
        minimum_samples: int = 4,
        complexity_weight: float = 2.0,
    ) -> None:
        self.graph = graph
        self.minimum_samples = minimum_samples
        self.complexity_weight = complexity_weight
        self._history: dict[str, list[float]] = defaultdict(list)

    def observe(self, node_id: str, value: float) -> None:
        self.graph.require_node(node_id)
        if not math.isfinite(value):
            raise ValueError("Observed values must be finite")
        self._history[node_id].append(value)

    def propose_edge(
        self, source_id: str, target_id: str
    ) -> StructureProposal | None:
        source = self._history[source_id]
        target = self._history[target_id]
        sample_count = min(len(source), len(target))
        if sample_count < self.minimum_samples:
            return None
        x = source[-sample_count:]
        y = target[-sample_count:]
        x_mean = sum(x) / sample_count
        y_mean = sum(y) / sample_count
        x_variance_sum = sum((item - x_mean) ** 2 for item in x)
        if x_variance_sum <= 1e-12:
            return None

        covariance_sum = sum(
            (x_item - x_mean) * (y_item - y_mean)
            for x_item, y_item in zip(x, y, strict=True)
        )
        weight = covariance_sum / x_variance_sum
        if abs(weight) <= 1e-9:
            return None
        bias = y_mean - weight * x_mean
        residuals = [
            y_item - (weight * x_item + bias)
            for x_item, y_item in zip(x, y, strict=True)
        ]
        residual_variance = max(
            sum(item * item for item in residuals) / sample_count, 1e-9
        )
        baseline_variance = max(
            sum((item - y_mean) ** 2 for item in y) / sample_count, 1e-9
        )
        evidence_gain = 0.5 * sample_count * math.log(
            baseline_variance / residual_variance
        )
        complexity_penalty = self.complexity_weight * math.log(sample_count)
        return StructureProposal(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            bias=bias,
            noise_variance=residual_variance,
            evidence_gain=evidence_gain,
            complexity_penalty=complexity_penalty,
        )

    def integrate(self, proposal: StructureProposal) -> str:
        if proposal.score <= 0:
            raise ValueError("Proposal does not improve penalized model evidence")
        edge = self.graph.add_edge(
            proposal.source_id,
            proposal.target_id,
            EdgeType.PREDICTS,
            weight=proposal.weight,
            bias=proposal.bias,
            noise_variance=proposal.noise_variance,
            confidence=min(1.0, 1.0 - math.exp(-proposal.score)),
            learned=True,
            metadata={"model_selection_score": proposal.score},
        )
        return edge.id

