from __future__ import annotations

import itertools
import math
from collections import defaultdict

from .graph import CausalGraph
from .models import (
    ContactRelationProposal,
    EdgeType,
    GaussianBelief,
    NodeType,
)
from .twin_physics import MultiObjectSnapshot


class ContactRelationLearner:
    """Bayesian existence learner for pairwise collision relations."""

    def __init__(
        self,
        graph: CausalGraph,
        *,
        minimum_observations: int = 3,
        minimum_probability: float = 0.6,
        complexity_weight: float = 1.0,
    ) -> None:
        self.graph = graph
        self.minimum_observations = minimum_observations
        self.minimum_probability = minimum_probability
        self.complexity_weight = complexity_weight
        self._history: dict[tuple[str, str], list[bool]] = defaultdict(list)

    def observe(
        self,
        snapshot: MultiObjectSnapshot,
        object_ids: list[str],
    ) -> None:
        collided = {event.pair for event in snapshot.collisions}
        for left, right in itertools.combinations(sorted(object_ids), 2):
            pair = (left, right)
            self._history[pair].append(pair in collided)

    def propose(
        self,
        object_a: str,
        object_b: str,
    ) -> ContactRelationProposal | None:
        pair = tuple(sorted((object_a, object_b)))
        history = self._history[pair]
        if len(history) < self.minimum_observations:
            return None
        contacts = sum(history)
        probability = (contacts + 1) / (len(history) + 2)
        if probability < self.minimum_probability:
            return None
        null_probability = 0.1
        relation_probability = 0.9
        evidence_gain = sum(
            math.log(
                (relation_probability if observed else 1 - relation_probability)
                / (null_probability if observed else 1 - null_probability)
            )
            for observed in history
        )
        complexity_penalty = self.complexity_weight * math.log(len(history) + 1)
        return ContactRelationProposal(
            object_a=pair[0],
            object_b=pair[1],
            probability=probability,
            evidence_gain=evidence_gain,
            complexity_penalty=complexity_penalty,
            observations=len(history),
        )

    def integrate(self, proposal: ContactRelationProposal) -> str:
        if proposal.score <= 0:
            raise ValueError("Contact proposal does not improve model evidence")
        left = self.graph.find_node_by_name(proposal.object_a)
        right = self.graph.find_node_by_name(proposal.object_b)
        relation = self.graph.add_node(
            f"contact.{proposal.object_a}.{proposal.object_b}",
            NodeType.CONTACT_RELATION,
            GaussianBelief(proposal.probability, 1e-6),
            modality="relational",
            metadata={
                "active": True,
                "model_selection_score": proposal.score,
                "observations": proposal.observations,
            },
        )
        for object_node in (left, right):
            self.graph.add_edge(
                object_node.id,
                relation.id,
                EdgeType.TOUCHES,
                noise_variance=1e-6,
                confidence=proposal.probability,
                learned=True,
            )
        return relation.id

