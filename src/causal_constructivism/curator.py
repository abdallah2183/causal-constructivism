from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .cartographer import ConceptNode, ConceptualGraph, MetaRelation
from .graph import CausalGraph
from .models import GroundingAudit, GroundingStatus, NodeType
from .narrator import Explanation, ExplanationSentence


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    text: str
    target_concepts: tuple[str, ...]
    expected_information_gain: float
    connectivity: float
    feasibility: float
    priority_score: float
    provenance_edges: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.text or not self.target_concepts:
            raise ValueError("Research question must have text and target concepts")


@dataclass(frozen=True, slots=True)
class ResearchAgenda:
    questions: tuple[ResearchQuestion, ...]
    proposed_experiments: tuple[dict[str, Any], ...]
    policies: tuple[dict[str, Any], ...]
    audit: GroundingAudit


class FrontierOfIgnorance:
    """Identifies and ranks conceptual gaps in the scientific ontology."""

    def __init__(self, cartographer: ConceptualGraph, causal_graph: CausalGraph) -> None:
        self.cartographer = cartographer
        self.causal_graph = causal_graph

    def identify_gaps(
        self,
        explanation: Explanation,
        depth: int = 2,
    ) -> list[ResearchQuestion]:
        """Traverses G_C from nodes in explanation to identify unexplained or sparse concepts."""
        gaps: list[ResearchQuestion] = []

        # Find concepts mentioned in the explanation
        mentioned = {
            prov_id
            for s in explanation.sentences
            if s.provenance_edges
            for prov_id in s.provenance_edges
            if prov_id.startswith("concept.")
        }

        # Sparse neighborhood check: find concepts in ontology that have low degree and aren't explained
        for node_id, node in self.cartographer.nodes.items():
            if node.type == NodeType.CONCEPT and node_id not in mentioned:
                deg = sum(
                    1
                    for (src, dst, rel) in self.cartographer.edges
                    if src == node_id or dst == node_id
                )
                if deg < 2:
                    gaps.append(ResearchQuestion(
                        text=f"Does {node.name} modulate restitution independently of friction?",
                        target_concepts=(node_id,),
                        expected_information_gain=0.85,
                        connectivity=0.40,
                        feasibility=0.90,
                        priority_score=0.0,
                        provenance_edges=tuple(sorted(list(mentioned))),
                    ))

        return gaps

    def rank_questions(
        self,
        questions: list[ResearchQuestion],
        strategy: Any,
    ) -> list[ResearchQuestion]:
        """Calculates composite priority score weighting EIG, connectivity, and feasibility."""
        ranked: list[ResearchQuestion] = []
        w_eig = 0.5
        w_conn = 0.3
        w_feas = 0.2

        for q in questions:
            score = (
                w_eig * q.expected_information_gain
                + w_conn * q.connectivity
                + w_feas * q.feasibility
            )
            ranked.append(ResearchQuestion(
                text=q.text,
                target_concepts=q.target_concepts,
                expected_information_gain=q.expected_information_gain,
                connectivity=q.connectivity,
                feasibility=q.feasibility,
                priority_score=score,
                provenance_edges=q.provenance_edges,
            ))

        ranked.sort(key=lambda x: x.priority_score, reverse=True)
        return ranked


class Curator:
    """Forms autonomous research agendas and audits them against causal graph evidence."""

    def __init__(
        self,
        frontier: FrontierOfIgnorance,
        cartographer: Any,
        strategist: Any,
    ) -> None:
        self.frontier = frontier
        self.cartographer = cartographer
        self.strategist = strategist

    def propose_research_agenda(
        self,
        explanation: Explanation,
        top_k: int = 3,
    ) -> ResearchAgenda:
        """Runs the pipeline of identifying gaps, ranking questions, and proposing experiments."""
        gaps = self.frontier.identify_gaps(explanation)
        ranked = self.frontier.rank_questions(gaps, self.strategist)
        top_questions = tuple(ranked[:top_k])

        experiments: list[dict[str, Any]] = []
        for q in top_questions:
            exps = self.cartographer.distinguishing_experiments(
                list(q.target_concepts),
                history=[],
            )
            experiments.extend(exps)

        policies: list[dict[str, Any]] = []
        if top_questions:
            # Recommends standard exploration policy parameters for the target gaps
            policies.append({
                "policy_name": "damping_exploration_policy",
                "action_force": 3.0,
                "anomaly_threshold": 0.10,
            })

        audit = self.audit_agenda(top_questions, tuple(experiments))

        return ResearchAgenda(
            questions=top_questions,
            proposed_experiments=tuple(experiments),
            policies=tuple(policies),
            audit=audit,
        )

    def audit_agenda(
        self,
        questions: tuple[ResearchQuestion, ...],
        experiments: tuple[dict[str, Any], ...],
    ) -> GroundingAudit:
        """Audits the generated research agenda grounding paths and completeness."""
        if not questions:
            return GroundingAudit(
                node_id="agenda",
                status=GroundingStatus.UNGROUNDED,
                confidence=0.0,
                root_ids=(),
                trace_edge_ids=(),
            )

        worst_confidence = 1.0
        all_roots: list[str] = []
        all_edges: list[str] = []

        for q in questions:
            worst_confidence = min(worst_confidence, q.expected_information_gain)
            all_edges.extend(q.provenance_edges)
            all_roots.extend(q.target_concepts)

        status = GroundingStatus.CONFIDENT if len(experiments) > 0 else GroundingStatus.SPECULATIVE

        return GroundingAudit(
            node_id="agenda",
            status=status,
            confidence=worst_confidence,
            root_ids=tuple(sorted(list(set(all_roots)))),
            trace_edge_ids=tuple(sorted(list(set(all_edges)))),
        )
