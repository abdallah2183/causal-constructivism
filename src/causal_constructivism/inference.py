from __future__ import annotations

import heapq
import math

from .graph import CausalGraph
from .models import (
    Edge,
    EdgeType,
    GaussianBelief,
    InferenceResult,
    utc_now,
)


SUPPORTED_FACTOR_TYPES = {
    EdgeType.CAUSES,
    EdgeType.OBSERVES,
    EdgeType.PREDICTS,
}


class GaussianBeliefPropagation:
    """Residual-scheduled Gaussian BP over linear pairwise factors."""

    def __init__(
        self,
        graph: CausalGraph,
        *,
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 1_000,
        damping: float = 0.35,
    ) -> None:
        if not 0 <= damping < 1:
            raise ValueError("Damping must be in [0, 1)")
        self.graph = graph
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        self._messages: dict[tuple[str, str], GaussianBelief] = {}

    def run(self, seed_node_ids: list[str] | None = None) -> InferenceResult:
        supported_edges = [
            edge for edge in self.graph.edges if edge.edge_type in SUPPORTED_FACTOR_TYPES
        ]
        queue: list[tuple[float, int, str, str]] = []
        counter = 0
        seeds = set(seed_node_ids or [node.id for node in self.graph.nodes])
        for edge in supported_edges:
            if edge.source_id in seeds or edge.target_id in seeds:
                for sender, receiver in (
                    (edge.source_id, edge.target_id),
                    (edge.target_id, edge.source_id),
                ):
                    heapq.heappush(queue, (-math.inf, counter, sender, receiver))
                    counter += 1

        changed_nodes: set[str] = set()
        max_residual = 0.0
        iterations = 0

        while queue and iterations < self.max_iterations:
            _, _, sender_id, receiver_id = heapq.heappop(queue)
            edge = self._edge_between(sender_id, receiver_id)
            if edge is None:
                continue
            proposed = self._compute_message(edge, sender_id, receiver_id)
            key = (sender_id, receiver_id)
            previous = self._messages.get(key)
            updated = self._damp(previous, proposed)
            residual = self._residual(previous, updated)
            self._messages[key] = updated
            iterations += 1
            max_residual = max(max_residual, residual)

            if residual <= self.convergence_tolerance:
                continue

            receiver = self.graph.require_node(receiver_id)
            old_belief = receiver.belief
            receiver.belief = self._node_belief(receiver_id)
            receiver.updated_at = utc_now()
            if self._residual(old_belief, receiver.belief) > self.convergence_tolerance:
                changed_nodes.add(receiver_id)

            for neighbor_id in self.graph.neighbors(receiver_id):
                if neighbor_id == sender_id:
                    continue
                next_edge = self._edge_between(receiver_id, neighbor_id)
                if next_edge and next_edge.edge_type in SUPPORTED_FACTOR_TYPES:
                    priority = -max(residual, self.convergence_tolerance)
                    heapq.heappush(
                        queue, (priority, counter, receiver_id, neighbor_id)
                    )
                    counter += 1

        converged = not queue
        return InferenceResult(
            converged=converged,
            iterations=iterations,
            max_residual=max_residual,
            free_energy=self.free_energy(),
            changed_nodes=tuple(sorted(changed_nodes)),
        )

    def free_energy(self) -> float:
        complexity = sum(
            node.belief.kl_divergence(node.prior)
            for node in self.graph.iter_active_nodes()
        )
        accuracy_cost = 0.0
        for node in self.graph.iter_active_nodes():
            if node.evidence is None:
                continue
            variance = node.belief.variance + node.evidence.variance
            error = node.evidence.mean - node.belief.mean
            accuracy_cost += 0.5 * (
                math.log(2 * math.pi * variance) + (error * error) / variance
            )
        return complexity + accuracy_cost

    def prediction_error(self, node_id: str) -> float:
        node = self.graph.require_node(node_id)
        if node.evidence is None:
            return 0.0
        return abs(node.evidence.mean - self._cavity(node_id, None).mean)

    def _node_belief(self, node_id: str) -> GaussianBelief:
        node = self.graph.require_node(node_id)
        components = [node.prior]
        if node.evidence is not None:
            components.append(node.evidence)
        components.extend(
            message
            for (sender, receiver), message in self._messages.items()
            if receiver == node_id
        )
        return GaussianBelief.fuse(components)

    def _cavity(self, node_id: str, excluded_neighbor: str | None) -> GaussianBelief:
        node = self.graph.require_node(node_id)
        components = [node.prior]
        if node.evidence is not None:
            components.append(node.evidence)
        components.extend(
            message
            for (sender, receiver), message in self._messages.items()
            if receiver == node_id and sender != excluded_neighbor
        )
        return GaussianBelief.fuse(components)

    def _compute_message(
        self, edge: Edge, sender_id: str, receiver_id: str
    ) -> GaussianBelief:
        sender = self._cavity(sender_id, receiver_id)
        if sender_id == edge.source_id:
            return GaussianBelief(
                mean=edge.weight * sender.mean + edge.bias,
                variance=(
                    edge.weight * edge.weight * sender.variance
                    + edge.noise_variance / max(edge.confidence, 1e-6)
                ),
            )
        return GaussianBelief(
            mean=(sender.mean - edge.bias) / edge.weight,
            variance=(
                sender.variance
                + edge.noise_variance / max(edge.confidence, 1e-6)
            )
            / (edge.weight * edge.weight),
        )

    def _edge_between(self, left_id: str, right_id: str) -> Edge | None:
        for edge in self.graph.incident_edges(left_id):
            if {edge.source_id, edge.target_id} == {left_id, right_id}:
                return edge
        return None

    def _damp(
        self,
        previous: GaussianBelief | None,
        proposed: GaussianBelief,
    ) -> GaussianBelief:
        if previous is None or self.damping == 0:
            return proposed
        keep = self.damping
        return GaussianBelief(
            mean=keep * previous.mean + (1 - keep) * proposed.mean,
            variance=max(
                keep * previous.variance + (1 - keep) * proposed.variance,
                1e-9,
            ),
        )

    @staticmethod
    def _residual(
        previous: GaussianBelief | None, current: GaussianBelief
    ) -> float:
        if previous is None:
            return math.inf
        return max(
            abs(previous.mean - current.mean),
            abs(math.log(previous.variance) - math.log(current.variance)),
        )
