from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import Iterable

from .models import Edge, EdgeType, GaussianBelief, Node, NodeType, utc_now


class CausalGraph:
    """Typed property graph with immutable superseded node versions."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._history: dict[str, list[Node]] = defaultdict(list)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges.values())

    def add_node(
        self,
        name: str,
        node_type: NodeType,
        prior: GaussianBelief,
        **kwargs: object,
    ) -> Node:
        node = Node.create(name, node_type, prior, **kwargs)
        if node.id in self._nodes:
            raise ValueError(f"Node already exists: {node.id}")
        self._nodes[node.id] = node
        return node

    def restore_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Node already exists: {node.id}")
        self._nodes[node.id] = deepcopy(node)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        **kwargs: object,
    ) -> Edge:
        self.require_node(source_id)
        self.require_node(target_id)
        edge = Edge.create(source_id, target_id, edge_type, **kwargs)
        duplicate = any(
            existing.target_id == target_id and existing.edge_type is edge_type
            for existing in self.outgoing_edges(source_id)
        )
        if duplicate:
            raise ValueError(
                f"Duplicate {edge_type.value} edge: {source_id} -> {target_id}"
            )
        self._edges[edge.id] = edge
        self._outgoing[source_id].add(edge.id)
        self._incoming[target_id].add(edge.id)
        return edge

    def restore_edge(self, edge: Edge) -> None:
        self.require_node(edge.source_id)
        self.require_node(edge.target_id)
        if edge.id in self._edges:
            raise ValueError(f"Edge already exists: {edge.id}")
        self._edges[edge.id] = deepcopy(edge)
        self._outgoing[edge.source_id].add(edge.id)
        self._incoming[edge.target_id].add(edge.id)

    def require_node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown node: {node_id}") from exc

    def require_edge(self, edge_id: str) -> Edge:
        try:
            return self._edges[edge_id]
        except KeyError as exc:
            raise KeyError(f"Unknown edge: {edge_id}") from exc

    def find_node_by_name(self, name: str) -> Node:
        matches = [
            node
            for node in self._nodes.values()
            if node.name == name and node.deprecated_at is None
        ]
        if not matches:
            raise KeyError(f"Unknown active node name: {name}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous active node name: {name}")
        return matches[0]

    def incoming_edges(self, node_id: str) -> tuple[Edge, ...]:
        self.require_node(node_id)
        return tuple(self._edges[item] for item in self._incoming[node_id])

    def outgoing_edges(self, node_id: str) -> tuple[Edge, ...]:
        self.require_node(node_id)
        return tuple(self._edges[item] for item in self._outgoing[node_id])

    def incident_edges(self, node_id: str) -> tuple[Edge, ...]:
        edge_ids = self._incoming[node_id] | self._outgoing[node_id]
        return tuple(self._edges[item] for item in edge_ids)

    def remove_edge(self, edge_id: str) -> Edge:
        edge = self.require_edge(edge_id)
        self._outgoing[edge.source_id].discard(edge_id)
        self._incoming[edge.target_id].discard(edge_id)
        del self._edges[edge_id]
        return edge

    def remove_incoming_edges(self, node_id: str) -> tuple[Edge, ...]:
        self.require_node(node_id)
        return tuple(
            self.remove_edge(edge_id)
            for edge_id in tuple(self._incoming[node_id])
        )

    def neighbors(self, node_id: str) -> tuple[str, ...]:
        neighbors: set[str] = set()
        for edge in self.incident_edges(node_id):
            neighbors.add(
                edge.target_id if edge.source_id == node_id else edge.source_id
            )
        return tuple(neighbors)

    def set_evidence(self, node_id: str, evidence: GaussianBelief) -> Node:
        node = self.require_node(node_id)
        node.evidence = evidence
        node.belief = GaussianBelief.fuse([node.prior, evidence])
        node.updated_at = utc_now()
        return node

    def clear_evidence(self, node_id: str) -> Node:
        node = self.require_node(node_id)
        node.evidence = None
        node.belief = node.prior
        node.updated_at = utc_now()
        return node

    def deprecate_node(self, node_id: str, *, reason: str) -> Node:
        node = self.require_node(node_id)
        node.deprecated_at = utc_now()
        node.metadata = {**node.metadata, "deprecation_reason": reason}
        node.updated_at = utc_now()
        return node

    def intervene(
        self,
        node_id: str,
        value: float,
        *,
        variance: float = 1e-9,
    ) -> tuple[Node, tuple[Edge, ...]]:
        node = self.require_node(node_id)
        removed = self.remove_incoming_edges(node_id)
        fixed = GaussianBelief(value, variance)
        node.prior = fixed
        node.belief = fixed
        node.evidence = None
        node.metadata = {
            **node.metadata,
            "intervened": True,
            "intervention_value": value,
            "severed_edge_ids": tuple(edge.id for edge in removed),
        }
        node.updated_at = utc_now()
        return node, removed

    def clone(self) -> CausalGraph:
        return deepcopy(self)

    def version_node(
        self,
        node_id: str,
        *,
        prior: GaussianBelief | None = None,
        metadata_updates: dict[str, object] | None = None,
    ) -> Node:
        current = self.require_node(node_id)
        snapshot = replace(current, metadata=dict(current.metadata))
        self._history[node_id].append(snapshot)
        current.deprecated_at = utc_now()

        replacement = replace(
            current,
            id=f"{current.id}:v{current.version + 1}",
            prior=prior or current.prior,
            belief=prior or current.belief,
            evidence=None,
            metadata={**current.metadata, **(metadata_updates or {})},
            version=current.version + 1,
            created_at=utc_now(),
            updated_at=utc_now(),
            deprecated_at=None,
            superseded_by=None,
        )
        current.superseded_by = replacement.id
        self._nodes[replacement.id] = replacement
        incident_edges = tuple(self.incident_edges(node_id))
        for edge in incident_edges:
            source_id = replacement.id if edge.source_id == node_id else edge.source_id
            target_id = replacement.id if edge.target_id == node_id else edge.target_id
            self.add_edge(
                source_id,
                target_id,
                edge.edge_type,
                weight=edge.weight,
                bias=edge.bias,
                noise_variance=edge.noise_variance,
                confidence=edge.confidence,
                learned=edge.learned,
                metadata={**edge.metadata, "versioned_from_edge": edge.id},
            )
        return replacement

    def node_history(self, node_id: str) -> tuple[Node, ...]:
        return tuple(self._history[node_id])

    def restore_history(self, node_id: str, snapshots: Iterable[Node]) -> None:
        self._history[node_id] = [deepcopy(snapshot) for snapshot in snapshots]

    def history_items(self) -> tuple[tuple[str, tuple[Node, ...]], ...]:
        return tuple(
            (node_id, tuple(snapshots))
            for node_id, snapshots in self._history.items()
        )

    def grounding_paths(
        self, node_id: str, *, max_depth: int = 32
    ) -> tuple[tuple[str, ...], ...]:
        self.require_node(node_id)
        paths: list[tuple[str, ...]] = []
        queue: deque[tuple[str, tuple[str, ...], frozenset[str]]] = deque(
            [(node_id, (), frozenset({node_id}))]
        )
        while queue:
            current_id, edge_path, visited = queue.popleft()
            if len(edge_path) > max_depth:
                continue
            current = self.require_node(current_id)
            if current.is_grounding_root:
                paths.append(edge_path)
                continue
            for edge in self.incoming_edges(current_id):
                if edge.source_id not in visited:
                    queue.append(
                        (
                            edge.source_id,
                            edge_path + (edge.id,),
                            visited | {edge.source_id},
                        )
                    )
        return tuple(paths)

    def validate_grounding(self) -> tuple[str, ...]:
        return tuple(
            node.id
            for node in self._nodes.values()
            if not node.deprecated_at
            and not node.is_grounding_root
            and not self.grounding_paths(node.id)
        )

    def iter_active_nodes(self) -> Iterable[Node]:
        return (node for node in self._nodes.values() if node.deprecated_at is None)
