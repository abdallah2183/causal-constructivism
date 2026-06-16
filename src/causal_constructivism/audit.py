from __future__ import annotations

from .graph import CausalGraph
from .models import GroundingAudit, GroundingStatus


class MetacognitiveAuditor:
    def __init__(self, graph: CausalGraph) -> None:
        self.graph = graph

    def audit(self, node_id: str) -> GroundingAudit:
        node = self.graph.require_node(node_id)
        paths = self.graph.grounding_paths(node_id)
        if node.is_grounding_root:
            roots = (node.id,)
            edge_ids: tuple[str, ...] = ()
        else:
            roots_list: list[str] = []
            edge_list: list[str] = []
            for path in paths:
                current_id = node_id
                for edge_id in path:
                    edge = self.graph.require_edge(edge_id)
                    edge_list.append(edge_id)
                    current_id = edge.source_id
                roots_list.append(current_id)
            roots = tuple(dict.fromkeys(roots_list))
            edge_ids = tuple(dict.fromkeys(edge_list))

        if not roots:
            status = GroundingStatus.UNGROUNDED
            confidence = 0.0
        else:
            edge_confidence = min(
                (self.graph.require_edge(edge_id).confidence for edge_id in edge_ids),
                default=1.0,
            )
            confidence = min(node.belief.confidence, edge_confidence)
            status = (
                GroundingStatus.CONFIDENT
                if confidence >= 0.7
                else GroundingStatus.SPECULATIVE
            )
        return GroundingAudit(
            node_id=node_id,
            status=status,
            confidence=confidence,
            root_ids=roots,
            trace_edge_ids=edge_ids,
        )

