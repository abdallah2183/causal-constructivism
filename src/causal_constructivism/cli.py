from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agency import ClosedLoopEmbodiedScientist
from .composer import run_composer_benchmark
from .discovery import (
    DiscoverySystem,
    ExperimentLog,
    synthetic_sliding_records,
)
from .embodied import ForceAction3D
from .embodied_system import EmbodiedVisionSystem, synthetic_occlusion_frames
from .models import ActionCandidate
from .mujoco_adapter import MuJoCoAdapter, MuJoCoUnavailableError
from .cartographer import run_cartographer_benchmark
from .generalist import ConceptLibrary, run_generalist_benchmark
from .historian import run_historian_benchmark
from .narrator_system import run_narrator_benchmark
from .curator_system import run_curator_benchmark
from .collaborator_system import run_collaborator_benchmark
from .strategist import run_strategist_benchmark
from .theorist import run_theorist_benchmark
from .universalist import run_universalist_benchmark
from .integrator_system import run_integrator_benchmark
from .programmer import run_programmer_benchmark
from .system import CausalConstructivismSystem

from .twin_system import default_twin_world


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Causal Constructivism active mass-inference demo."
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--true-mass", type=float, default=2.5)
    parser.add_argument("--sensor-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--twin",
        action="store_true",
        help="Run the Phase 2 Twin World counterfactual benchmark.",
    )
    parser.add_argument(
        "--intervention-variable",
        default="action.force",
    )
    parser.add_argument(
        "--intervention-value",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--query-variable",
        default="green.position",
    )
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument(
        "--embodied",
        action="store_true",
        help="Run the Phase 3 synthetic RGB-D permanence benchmark.",
    )
    parser.add_argument(
        "--database",
        default="artifacts/embodied-vision.db",
        help="SQLite graph database used by the embodied benchmark.",
    )
    parser.add_argument(
        "--mujoco",
        action="store_true",
        help="Run the native MuJoCo starter-scene smoke benchmark.",
    )
    parser.add_argument(
        "--scene",
        default="assets/scenes/minimal_blocks.xml",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help="Run the Phase 4 closed-loop embodied active-inference benchmark.",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=6,
        help="Number of closed-loop experiments to run.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run the Phase 5 neural structure-proposal friction discovery benchmark.",
    )
    parser.add_argument(
        "--discovery-log",
        default="artifacts/discovery-log.jsonl",
        help="JSONL experiment log written by the discovery benchmark.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.25,
        help="Hidden friction coefficient for the synthetic discovery benchmark.",
    )
    parser.add_argument(
        "--generalist",
        action="store_true",
        help="Run the Phase 6 concept-transfer and causal-revision benchmark.",
    )
    parser.add_argument(
        "--source-friction",
        type=float,
        default=0.30,
        help="Known source-environment friction coefficient.",
    )
    parser.add_argument(
        "--target-friction",
        type=float,
        default=0.05,
        help="Hidden target-environment friction coefficient.",
    )
    parser.add_argument(
        "--concept-library",
        default="artifacts/concept-library.json",
        help="Concept library path written by the generalist benchmark.",
    )
    parser.add_argument(
        "--historian",
        action="store_true",
        help="Run the Phase 7 history-level counterfactual benchmark.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=6,
        help="Number of experiments in the historian benchmark.",
    )
    parser.add_argument(
        "--hidden-friction",
        type=float,
        default=0.25,
        help="Hidden friction coefficient in the historian benchmark.",
    )
    parser.add_argument(
        "--composer",
        action="store_true",
        help="Run the Phase 8 multi-concept composer benchmark.",
    )
    parser.add_argument(
        "--composer-friction",
        type=float,
        default=0.25,
        help="Hidden friction coefficient in the composer benchmark.",
    )
    parser.add_argument(
        "--composer-restitution",
        type=float,
        default=0.65,
        help="Hidden restitution coefficient in the composer benchmark.",
    )
    parser.add_argument(
        "--compound-count",
        type=int,
        default=8,
        help="Number of compound observations in the composer benchmark.",
    )
    parser.add_argument(
        "--theorist",
        action="store_true",
        help="Run the Phase 9 symbolic equation-discovery benchmark.",
    )
    parser.add_argument(
        "--law-observations",
        type=int,
        default=20,
        help="Number of observations in the theorist benchmark.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="Gravity used by the theorist pendulum benchmark.",
    )
    parser.add_argument(
        "--universalist",
        action="store_true",
        help="Run the Phase 10 cross-domain unification benchmark.",
    )
    parser.add_argument(
        "--strategist",
        action="store_true",
        help="Run the Phase 11 self-improving methodology benchmark.",
    )
    parser.add_argument(
        "--cartographer",
        action="store_true",
        help="Run the Phase 12 conceptual-atlas benchmark.",
    )
    parser.add_argument(
        "--narrator",
        action="store_true",
        help="Run the Phase 13 grounded explanation benchmark.",
    )
    parser.add_argument(
        "--curator",
        action="store_true",
        help="Run the Phase 14 autonomous research-agenda benchmark.",
    )
    parser.add_argument(
        "--collaborator",
        action="store_true",
        help="Run the Phase 15 multi-agent discourse benchmark.",
    )
    parser.add_argument(
        "--integrator",
        action="store_true",
        help="Run the Phase 16 end-to-end cognitive loop orchestration benchmark.",
    )
    parser.add_argument(
        "--programmer",
        action="store_true",
        help="Run the Phase 17 local programming-core benchmark.",
    )
    parser.add_argument(
        "--programmer-task",
        default="Add a verified programming capability to this Python project.",
        help="Programming task text used by the Phase 17 planner.",
    )
    parser.add_argument(
        "--programmer-run-tests",
        action="store_true",
        help="Run the full local test suite during the Phase 17 benchmark.",
    )
    parser.add_argument(
        "--programmer-memory",
        default=None,
        help="Optional JSONL memory path for recording the Phase 17 evidence report.",
    )
    return parser



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.programmer:
        return run_programmer(args)
    if args.integrator:
        return run_integrator(args)
    if args.collaborator:
        return run_collaborator(args)
    if args.curator:
        return run_curator(args)
    if args.narrator:
        return run_narrator(args)
    if args.cartographer:
        return run_cartographer(args)
    if args.strategist:
        return run_strategist(args)

    if args.universalist:
        return run_universalist(args)
    if args.theorist:
        return run_theorist(args)
    if args.composer:
        return run_composer(args)
    if args.historian:
        return run_historian(args)
    if args.generalist:
        return run_generalist(args)
    if args.discover:
        return run_discovery(args)
    if args.closed_loop:
        return run_closed_loop(args)
    if args.mujoco:
        return run_mujoco(args)
    if args.embodied:
        return run_embodied_vision(args)
    if args.twin:
        return run_twin_world(args)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    system = CausalConstructivismSystem(
        true_mass=args.true_mass,
        sensor_noise_std=args.sensor_noise,
        seed=args.seed,
    )
    actions = [
        ActionCandidate(force=1.0, duration=0.2),
        ActionCandidate(force=3.0, duration=0.2),
        ActionCandidate(force=6.0, duration=0.2),
    ]
    rows = []
    for index in range(1, args.steps + 1):
        step = system.step(actions)
        row = {
            "step": index,
            "force": step.observation.force,
            "acceleration": step.observation.acceleration,
            "mass_mean": step.mass_mean,
            "mass_std": step.mass_variance**0.5,
            "free_energy": step.inference.free_energy,
            "grounding": step.audit.status.value,
            "grounding_confidence": step.audit.confidence,
        }
        rows.append(row)

    if args.as_json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(
                "step={step} force={force:.1f} acceleration={acceleration:.3f} "
                "mass={mass_mean:.3f}+/-{mass_std:.3f} "
                "F={free_energy:.3f} grounding={grounding}"
                .format(**row)
            )
    return 0


def run_programmer(args: argparse.Namespace) -> int:
    memory_path = Path(args.programmer_memory) if args.programmer_memory else None
    result = run_programmer_benchmark(
        ".",
        args.programmer_task,
        run_tests=args.programmer_run_tests,
        memory_path=memory_path,
    )
    row = {
        "phase": 17,
        "status": result.status,
        "task": result.task.text,
        "project": {
            "root": result.index.root,
            "modules": len(result.index.modules),
            "source_files": len(result.index.source_files),
            "test_files": len(result.index.test_files),
            "symbols": result.index.symbol_count,
            "syntax_errors": result.index.syntax_error_count,
        },
        "plan": {
            "target_files": list(result.plan.target_files),
            "rationale": list(result.plan.rationale),
            "risk": result.plan.risk,
            "required_checks": [
                list(command) for command in result.plan.required_checks
            ],
        },
        "verification": [
            {
                "command": list(check.command),
                "returncode": check.returncode,
                "passed": check.passed,
            }
            for check in result.verification
        ],
        "failures": [
            {
                "source": failure.source,
                "line": failure.line,
                "message": failure.message,
            }
            for failure in result.failures
        ],
        "accelerator": {
            "available": result.accelerator.available,
            "name": result.accelerator.name,
            "total_memory_mib": result.accelerator.total_memory_mib,
            "driver_version": result.accelerator.driver_version,
            "reason": result.accelerator.reason,
        },
        "memory_written": result.memory_written,
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_cartographer(args: argparse.Namespace) -> int:
    result = run_cartographer_benchmark()
    row = {
        "query_node": result.query_node,
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "grounded_edge_count": result.grounded_edge_count,
        "grounding_quality": result.grounding_quality,
        "grounding": result.status.value,
        "neighborhood": list(result.neighborhood),
        "analogies": list(result.analogies),
        "proposed_experiments": list(result.proposed_experiments),
        "nodes": [
            {
                "id": node.node_id,
                "name": node.name,
                "type": node.type.value,
                "domain": node.domain,
                "parameters": node.parameters,
                "grounded_law_ref": node.grounded_law_ref,
            }
            for node in result.ontology.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.relation_type,
                "confidence": edge.confidence,
                "provenance_edges": list(edge.provenance_edges),
            }
            for edge in result.ontology.edges.values()
        ],
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_narrator(args: argparse.Namespace) -> int:
    result = run_narrator_benchmark()
    row = {
        "query_node": result.explanation.query_node,
        "sentence_count": result.sentence_count,
        "grounding_quality": result.grounding_quality,
        "grounding": result.status.value,
        "sentences": [
            {
                "text": sentence.text,
                "confidence": sentence.confidence,
                "provenance_edges": list(sentence.provenance_edges),
            }
            for sentence in result.explanation.sentences
        ],
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_curator(args: argparse.Namespace) -> int:
    result = run_curator_benchmark()
    row = {
        "question_count": result.question_count,
        "grounding_quality": result.grounding_quality,
        "grounding": result.status.value,
        "questions": [
            {
                "text": question.text,
                "target_concepts": list(question.target_concepts),
                "expected_information_gain": question.expected_information_gain,
                "connectivity": question.connectivity,
                "feasibility": question.feasibility,
                "priority_score": question.priority_score,
                "provenance_edges": list(question.provenance_edges),
            }
            for question in result.agenda.questions
        ],
        "proposed_experiments": list(result.agenda.proposed_experiments),
        "policies": list(result.agenda.policies),
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_collaborator(args: argparse.Namespace) -> int:
    result = run_collaborator_benchmark()
    row = {
        "contested_node": result.record.contested_node,
        "rounds_count": result.rounds_count,
        "grounding_quality": result.grounding_quality,
        "grounding": result.status.value,
        "final_consensus": (
            {
                "source": result.record.final_consensus.source_id,
                "target": result.record.final_consensus.target_id,
                "type": result.record.final_consensus.relation_type,
                "confidence": result.record.final_consensus.confidence,
                "provenance_edges": list(result.record.final_consensus.provenance_edges),
            }
            if result.record.final_consensus is not None
            else None
        ),
        "rounds": [
            {
                "round_number": debate_round.round_number,
                "disagreement_type": debate_round.disagreement_type,
                "contradictions": [
                    {
                        "source": contradiction.source_id,
                        "target": contradiction.target_id,
                        "type": contradiction.relation_type,
                        "confidence": contradiction.confidence,
                        "provenance_edges": list(contradiction.provenance_edges),
                    }
                    for contradiction in debate_round.contradictions
                ],
                "proposed_experiments": [
                    {
                        "text": question.text,
                        "target_concepts": list(question.target_concepts),
                        "priority_score": question.priority_score,
                    }
                    for question in debate_round.proposed_experiments
                ],
                "executed_experiment": (
                    debate_round.executed_experiment.text
                    if debate_round.executed_experiment is not None
                    else None
                ),
                "belief_updates": {
                    agent_id: {
                        node_id: {
                            "mean": belief.mean,
                            "variance": belief.variance,
                        }
                        for node_id, belief in updates.items()
                    }
                    for agent_id, updates in debate_round.belief_updates.items()
                },
            }
            for debate_round in result.record.rounds
        ],
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_integrator(args: argparse.Namespace) -> int:
    result = run_integrator_benchmark()
    row = {
        "steps_count": result.steps_count,
        "grounding_quality": result.grounding_quality,
        "status": result.status.value,
        "steps": [
            {
                "phase_name": step.phase_name,
                "action_taken": step.action_taken,
                "status": step.status,
                "details": step.details,
            }
            for step in result.result.steps
        ],
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_strategist(args: argparse.Namespace) -> int:

    if args.history_length <= 0:
        raise SystemExit("--history-length must be positive")
    if args.true_mass <= 0:
        raise SystemExit("--true-mass must be positive")
    if args.hidden_friction < 0:
        raise SystemExit("--hidden-friction must be non-negative")
    result = run_strategist_benchmark(
        experiments=args.history_length,
        true_mass=args.true_mass,
        hidden_friction=args.hidden_friction,
    )
    row = {
        "baseline": {
            "policy": result.baseline.policy.policy_id,
            "experiments_to_friction": result.baseline.experiments_to_friction,
            "mean_mass_error": result.baseline.mean_mass_error,
            "composite_score": result.baseline.composite_score,
        },
        "selected": {
            "policy": result.selected.policy.policy_id,
            "experiments_to_friction": result.selected.experiments_to_friction,
            "mean_mass_error": result.selected.mean_mass_error,
            "composite_score": result.selected.composite_score,
            "efficiency_gain": result.selected.efficiency_gain,
            "accuracy_gain": result.selected.accuracy_gain,
            "grounding_gain": result.selected.grounding_gain,
        },
        "candidates": [
            {
                "policy": candidate.policy.policy_id,
                "experiments_to_friction": candidate.experiments_to_friction,
                "mean_mass_error": candidate.mean_mass_error,
                "composite_score": candidate.composite_score,
                "efficiency_gain": candidate.efficiency_gain,
                "accuracy_gain": candidate.accuracy_gain,
                "grounding_gain": candidate.grounding_gain,
            }
            for candidate in result.candidates
        ],
        "adopted": result.adopted,
        "grounding": result.audit.status.value,
        "grounding_confidence": result.audit.confidence,
        "graph_nodes": result.graph_nodes,
        "graph_edges": result.graph_edges,
        "ungrounded_nodes": result.ungrounded_nodes,
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_universalist(args: argparse.Namespace) -> int:
    result = run_universalist_benchmark()
    prediction = result.principle.predictions[0]
    row = {
        "principle": result.principle.name,
        "template": result.principle.template,
        "confidence": result.principle.confidence,
        "instances": [
            {
                "domain": instance.domain,
                "law_id": instance.law_id,
                "expression": (
                    f"{instance.coefficient:.6g}*"
                    f"{instance.expression.render()}"
                ),
                "signature": instance.structural_signature,
            }
            for instance in result.principle.instances
        ],
        "prediction": {
            "target_domain": prediction.target_domain,
            "predicted_law": prediction.predicted_law,
            "confidence": prediction.confidence,
            "tested": prediction.tested,
            "confirmed": prediction.confirmed,
            "grounding": prediction.audit.status.value if prediction.audit else None,
        },
        "confirmed_predictions": result.confirmed_predictions,
        "grounding": result.principle.audit.status.value,
        "grounding_confidence": result.principle.audit.confidence,
        "graph_nodes": result.graph_nodes,
        "graph_edges": result.graph_edges,
        "ungrounded_nodes": result.ungrounded_nodes,
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_theorist(args: argparse.Namespace) -> int:
    if args.law_observations <= 0:
        raise SystemExit("--law-observations must be positive")
    if args.gravity <= 0:
        raise SystemExit("--gravity must be positive")
    result = run_theorist_benchmark(
        count=args.law_observations,
        gravity=args.gravity,
    )
    row = {
        "selected_law": result.selected_law.template.name,
        "expression": result.selected_law.expression,
        "parameters": list(result.selected_law.parameters),
        "rmse": result.selected_law.rmse,
        "score": result.selected_law.score,
        "evidence_gain": result.selected_law.evidence_gain,
        "complexity_penalty": result.selected_law.complexity_penalty,
        "scope_bonus": result.selected_law.scope_bonus,
        "candidate_laws": [
            {
                "name": law.template.name,
                "expression": law.expression,
                "parameters": list(law.parameters),
                "rmse": law.rmse,
                "score": law.score,
                "evidence_gain": law.evidence_gain,
                "complexity_penalty": law.complexity_penalty,
            }
            for law in result.candidate_laws
        ],
        "grounding": result.audit.status.value,
        "grounding_confidence": result.audit.confidence,
        "graph_nodes": result.graph_nodes,
        "graph_edges": result.graph_edges,
        "ungrounded_nodes": result.ungrounded_nodes,
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_composer(args: argparse.Namespace) -> int:
    if args.composer_friction < 0:
        raise SystemExit("--composer-friction must be non-negative")
    if not 0 <= args.composer_restitution <= 1.5:
        raise SystemExit("--composer-restitution must be in [0, 1.5]")
    if args.compound_count <= 0:
        raise SystemExit("--compound-count must be positive")
    result = run_composer_benchmark(
        friction=args.composer_friction,
        restitution=args.composer_restitution,
        count=args.compound_count,
    )
    row = {
        "selected_concepts": list(result.selected_model.concepts),
        "parameters": result.selected_model.parameters,
        "score": result.selected_model.score,
        "evidence_gain": result.selected_model.evidence_gain,
        "residual_error": result.selected_model.residual_error,
        "candidate_models": [
            {
                "concepts": list(model.concepts),
                "parameters": model.parameters,
                "score": model.score,
                "evidence_gain": model.evidence_gain,
                "residual_error": model.residual_error,
            }
            for model in result.candidate_models
        ],
        "concept_grounding": {
            concept.name: concept.audit.status.value
            for concept in result.concepts
        },
        "interaction": (
            {
                "source": result.interaction.source_concept,
                "target": result.interaction.target_concept,
                "type": result.interaction.interaction_type,
                "grounding": result.interaction.audit.status.value,
            }
            if result.interaction is not None
            else None
        ),
        "graph_nodes": result.graph_nodes,
        "graph_edges": result.graph_edges,
        "ungrounded_nodes": result.ungrounded_nodes,
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_historian(args: argparse.Namespace) -> int:
    if args.history_length <= 0:
        raise SystemExit("--history-length must be positive")
    if args.true_mass <= 0:
        raise SystemExit("--true-mass must be positive")
    if args.hidden_friction < 0:
        raise SystemExit("--hidden-friction must be non-negative")
    result = run_historian_benchmark(
        experiments=args.history_length,
        true_mass=args.true_mass,
        hidden_friction=args.hidden_friction,
    )
    row = {
        "experiments": len(result.actual.experiments),
        "intervention": result.intervention.name,
        "shared_exogenous_ids": list(result.shared_exogenous_ids),
        "actual": {
            "policy": result.actual.policy.name,
            "mean_mass_error": result.actual.metrics.mean_mass_error,
            "final_mass_error": result.actual.metrics.final_mass_error,
            "mean_prediction_error": result.actual.metrics.mean_prediction_error,
            "experiments_to_friction": result.actual.metrics.experiments_to_friction,
            "confident_fraction": result.actual.metrics.confident_fraction,
            "ungrounded_nodes": result.actual.metrics.ungrounded_nodes,
            "final_mass_grounding": result.actual.metrics.final_mass_grounding.status.value,
            "graph_nodes": len(result.actual.graph.nodes),
            "graph_edges": len(result.actual.graph.edges),
        },
        "counterfactual": {
            "policy": result.counterfactual.policy.name,
            "mean_mass_error": result.counterfactual.metrics.mean_mass_error,
            "final_mass_error": result.counterfactual.metrics.final_mass_error,
            "mean_prediction_error": result.counterfactual.metrics.mean_prediction_error,
            "experiments_to_friction": result.counterfactual.metrics.experiments_to_friction,
            "confident_fraction": result.counterfactual.metrics.confident_fraction,
            "ungrounded_nodes": result.counterfactual.metrics.ungrounded_nodes,
            "final_mass_grounding": (
                result.counterfactual.metrics.final_mass_grounding.status.value
            ),
            "graph_nodes": len(result.counterfactual.graph.nodes),
            "graph_edges": len(result.counterfactual.graph.edges),
        },
        "comparison": {
            "mean_mass_error_reduction": result.mean_mass_error_reduction,
            "prediction_error_reduction": result.prediction_error_reduction,
            "experiments_saved_to_friction": result.experiments_saved_to_friction,
        },
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_generalist(args: argparse.Namespace) -> int:
    if args.source_friction < 0 or args.target_friction < 0:
        raise SystemExit("Friction coefficients must be non-negative")
    result = run_generalist_benchmark(
        source_friction=args.source_friction,
        target_friction=args.target_friction,
    )
    ConceptLibrary([result.revised_concept]).save(args.concept_library)
    row = {
        "source_concept": result.learned_concept.name,
        "source_coefficient": result.learned_concept.parameter_value,
        "target_environment": result.revision.environment,
        "revised_coefficient": result.revised_concept.parameter_value,
        "true_target_coefficient": args.target_friction,
        "pre_revision_error": result.pre_revision_error,
        "post_revision_error": result.post_revision_error,
        "evidence_gain": result.revision.evidence_gain,
        "transfer_grounding": result.transfer_audit.status.value,
        "revision_grounding": result.revision_audit.status.value,
        "revision_count": len(result.revised_concept.revision_history),
        "graph_nodes": result.graph_nodes,
        "graph_edges": result.graph_edges,
        "concept_library": str(Path(args.concept_library).resolve()),
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_discovery(args: argparse.Namespace) -> int:
    if args.friction < 0:
        raise SystemExit("--friction must be non-negative")
    log = ExperimentLog(
        synthetic_sliding_records(
            object_id="block_001",
            friction=args.friction,
            count=8,
        )
    )
    log.save_jsonl(args.discovery_log)
    system = DiscoverySystem()
    result = system.discover_friction(log, "block_001")
    if result is None:
        row = {
            "discovered": False,
            "records": len(log.records),
            "log": str(Path(args.discovery_log).resolve()),
        }
    else:
        proposal, node_id, audit = result
        row = {
            "discovered": True,
            "concept": proposal.candidate.concept,
            "target": proposal.candidate.target_object_id,
            "coefficient": proposal.candidate.parameters["coefficient"],
            "neural_confidence": proposal.candidate.confidence,
            "evidence_gain": proposal.evidence_gain,
            "score": proposal.score,
            "node_id": node_id,
            "grounding": audit.status.value,
            "grounding_confidence": audit.confidence,
            "ungrounded_nodes": len(system.graph.validate_grounding()),
            "records": len(log.records),
            "log": str(Path(args.discovery_log).resolve()),
        }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(row)
    return 0


def run_closed_loop(args: argparse.Namespace) -> int:
    if args.experiments <= 0:
        raise SystemExit("--experiments must be positive")
    system = ClosedLoopEmbodiedScientist(database_path=args.database)
    try:
        result = system.run(experiments=args.experiments)
    finally:
        system.close()
    row = {
        "experiments": len(result.steps),
        "successful": result.successful,
        "mass_estimates": result.mass_estimates,
        "true_masses": result.true_masses,
        "relative_errors": result.relative_errors,
        "selected_actions": [
            {
                "step": step.index,
                "object": step.selected.action.object_id,
                "force": step.selected.action.force,
                "acceleration": step.measured_acceleration,
                "mass_mean": step.mass_mean,
                "mass_std": step.mass_std,
                "grounding": step.audit.status.value,
            }
            for step in result.steps
        ],
        "graph_nodes": result.node_count,
        "graph_edges": result.edge_count,
        "snapshot_id": result.snapshot_id,
        "database": str(Path(args.database).resolve()),
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(
            "experiments={experiments} successful={successful} "
            "errors={relative_errors} nodes={graph_nodes} edges={graph_edges} "
            "snapshot={snapshot_id}"
            .format(**row)
        )
    return 0


def run_mujoco(args: argparse.Namespace) -> int:
    try:
        adapter = MuJoCoAdapter(
            args.scene,
            body_names=("red_block", "blue_block"),
            render_width=160,
            render_height=120,
        )
    except MuJoCoUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        adapter.reset()
        before = adapter.observe()
        after = adapter.step(
            ForceAction3D(
                "red_block",
                (20.0, 0.0, 0.0),
                duration=0.1,
            )
        )
        rgb, depth, segmentation = adapter.render_raw()
    finally:
        adapter.close()
    before_red = next(
        state for state in before.states if state.object_id == "red_block"
    )
    after_red = next(
        state for state in after.states if state.object_id == "red_block"
    )
    row = {
        "scene": str(Path(args.scene).resolve()),
        "before_position": before_red.position,
        "after_position": after_red.position,
        "after_velocity": after_red.velocity,
        "contacts": len(after.contacts),
        "rgb_shape": rgb.shape,
        "depth_shape": depth.shape,
        "segmentation_shape": segmentation.shape,
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(
            "scene={scene} before={before_position} after={after_position} "
            "velocity={after_velocity} contacts={contacts} "
            "rgb={rgb_shape} depth={depth_shape} segmentation={segmentation_shape}"
            .format(**row)
        )
    return 0


def run_embodied_vision(args: argparse.Namespace) -> int:
    database = Path(args.database)
    system = EmbodiedVisionSystem(database_path=database)
    result = system.run(synthetic_occlusion_frames())
    rows = {
        "frames": len(result.frames),
        "persistent_tracks": list(result.persistent_track_ids),
        "births": sum(
            len(frame.tracking.born_track_ids) for frame in result.frames
        ),
        "occlusions": sum(
            len(frame.tracking.occluded_track_ids) for frame in result.frames
        ),
        "losses": sum(
            len(frame.tracking.lost_track_ids) for frame in result.frames
        ),
        "graph_nodes": result.node_count,
        "graph_edges": result.edge_count,
        "grounding": [audit.status.value for audit in result.final_audits],
        "snapshot_id": result.snapshot_id,
        "database": str(database.resolve()),
    }
    if args.as_json:
        print(json.dumps(rows, indent=2))
    else:
        print(
            "frames={frames} tracks={persistent_tracks} births={births} "
            "occlusions={occlusions} losses={losses} nodes={graph_nodes} "
            "edges={graph_edges} grounding={grounding} snapshot={snapshot_id}"
            .format(**rows)
        )
    return 0


def run_twin_world(args: argparse.Namespace) -> int:
    if args.particles < 10:
        raise SystemExit("--particles must be at least 10")
    system = default_twin_world(particles=args.particles)
    relation_ids = system.learn_contact_relations(3)
    result = system.counterfactual(
        {args.intervention_variable: args.intervention_value},
        args.query_variable,
    )
    row = {
        "query": result.query_variable,
        "intervention": {
            args.intervention_variable: args.intervention_value,
        },
        "actual_mean": result.actual_belief.mean,
        "actual_std": result.actual_belief.variance**0.5,
        "counterfactual_mean": result.belief.mean,
        "counterfactual_std": result.belief.variance**0.5,
        "actual_collision_probability": result.actual_collision_probability,
        "counterfactual_collision_probability": result.collision_probability,
        "grounding": result.grounding.status.value,
        "grounding_confidence": result.grounding.confidence,
        "grounding_roots": len(result.grounding.root_ids),
        "learned_contact_relations": len(relation_ids),
    }
    if args.as_json:
        print(json.dumps(row, indent=2))
    else:
        print(
            "query={query} intervention={intervention} "
            "actual={actual_mean:.3f}+/-{actual_std:.3f} "
            "counterfactual={counterfactual_mean:.3f}+/-{counterfactual_std:.3f} "
            "P(collision)={counterfactual_collision_probability:.3f} "
            "grounding={grounding} relations={learned_contact_relations}"
            .format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
