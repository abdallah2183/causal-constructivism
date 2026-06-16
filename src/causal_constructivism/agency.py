from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .audit import MetacognitiveAuditor
from .embodied import ForceAction3D
from .embodied_system import EmbodiedFrameResult
from .models import EdgeType, GaussianBelief, GroundingAudit, NodeType
from .mujoco_adapter import MuJoCoAdapter
from .perception import SyntheticPerception
from .persistence import SQLiteGraphStore
from .physics import BayesianMassEstimator
from .temporal import TemporalGraphIntegrator
from .tracking import ObjectTracker


@dataclass(frozen=True, slots=True)
class ActionPrimitiveScore:
    action: ForceAction3D
    expected_free_energy: float
    epistemic_cost: float
    pragmatic_cost: float
    predicted_displacement: float
    predicted_visible_segments: int


@dataclass(frozen=True, slots=True)
class ClosedLoopStep:
    index: int
    selected: ActionPrimitiveScore
    measured_acceleration: float
    mass_mean: float
    mass_std: float
    frame: EmbodiedFrameResult
    audit: GroundingAudit


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    steps: tuple[ClosedLoopStep, ...]
    mass_estimates: dict[str, tuple[float, float]]
    true_masses: dict[str, float]
    relative_errors: dict[str, float]
    successful: bool
    node_count: int
    edge_count: int
    snapshot_id: int | None


class EmbodiedEFEPlanner:
    """Pruned 3D push planner using MuJoCo rollouts plus mass entropy."""

    def __init__(
        self,
        adapter: MuJoCoAdapter,
        estimators: dict[str, BayesianMassEstimator],
        *,
        force_magnitudes: tuple[float, ...] = (4.0, 8.0, 12.0),
        duration: float = 0.2,
        displacement_weight: float = 0.01,
        visibility_weight: float = 0.5,
    ) -> None:
        self.adapter = adapter
        self.estimators = estimators
        self.force_magnitudes = force_magnitudes
        self.duration = duration
        self.displacement_weight = displacement_weight
        self.visibility_weight = visibility_weight

    def candidates(self) -> tuple[ForceAction3D, ...]:
        actions = []
        for object_id in self.estimators:
            for magnitude in self.force_magnitudes:
                actions.append(
                    ForceAction3D(
                        object_id=object_id,
                        force=(magnitude, 0.0, 0.0),
                        duration=self.duration,
                    )
                )
        return tuple(actions)

    def score(self, action: ForceAction3D) -> ActionPrimitiveScore:
        before = self.adapter.observe()
        before_state = _state_for(before, action.object_id)
        state = self.adapter.capture_state()
        try:
            predicted = self.adapter.step(action)
            predicted_state = _state_for(predicted, action.object_id)
            displacement = math.dist(before_state.position, predicted_state.position)
            visible_segments = len(
                self.adapter.render_frame(
                    frame_id="candidate",
                    pixel_stride=8,
                ).segments
            )
        finally:
            self.adapter.restore_state(state)
        total_entropy = 0.0
        for object_id, estimator in self.estimators.items():
            if object_id == action.object_id:
                total_entropy += estimator.expected_entropy(_force_norm(action))
            else:
                total_entropy += estimator.entropy
        pragmatic_cost = (
            self.displacement_weight * displacement * displacement
            + self.visibility_weight
            * max(0, len(self.estimators) - visible_segments)
        )
        return ActionPrimitiveScore(
            action=action,
            expected_free_energy=total_entropy + pragmatic_cost,
            epistemic_cost=total_entropy,
            pragmatic_cost=pragmatic_cost,
            predicted_displacement=displacement,
            predicted_visible_segments=visible_segments,
        )

    def select(self) -> ActionPrimitiveScore:
        return min(
            (self.score(action) for action in self.candidates()),
            key=lambda score: score.expected_free_energy,
        )


class ClosedLoopEmbodiedScientist:
    """Observe, track, plan, execute, infer, audit, repeat in MuJoCo."""

    def __init__(
        self,
        *,
        scene: str | Path = "assets/scenes/embodied_scientist.xml",
        body_names: tuple[str, ...] = (
            "red_cube",
            "blue_sphere",
            "green_cylinder",
        ),
        database_path: str | Path | None = None,
        render_width: int = 240,
        render_height: int = 180,
    ) -> None:
        self.adapter = MuJoCoAdapter(
            scene,
            body_names=body_names,
            render_width=render_width,
            render_height=render_height,
        )
        self.body_names = body_names
        self.perception = SyntheticPerception(
            depth_noise_base=0.0,
            depth_noise_scale=0.0,
            pixel_noise_std=0.0,
        )
        self.tracker = ObjectTracker(
            association_threshold=40.0,
            maximum_misses=3,
            process_variance=0.05,
        )
        self.temporal = TemporalGraphIntegrator()
        self.estimators = {
            body_name: BayesianMassEstimator(
                minimum_mass=0.2,
                maximum_mass=6.0,
                bins=500,
                sensor_noise_std=0.04,
            )
            for body_name in body_names
        }
        self.planner = EmbodiedEFEPlanner(self.adapter, self.estimators)
        self.store = (
            SQLiteGraphStore(database_path) if database_path is not None else None
        )
        self._frame_index = 0
        self._mass_node_ids: dict[str, str] = {}

    @property
    def graph(self):
        return self.temporal.graph

    def reset(self) -> None:
        self.adapter.reset()
        self._ensure_dynamics_root()
        self._observe_frame("initial")

    def run(self, *, experiments: int = 6) -> ClosedLoopResult:
        if experiments <= 0:
            raise ValueError("Experiment count must be positive")
        self.reset()
        steps = []
        for index in range(1, experiments + 1):
            selected = self.planner.select()
            before = self.adapter.observe()
            before_state = _state_for(before, selected.action.object_id)
            self._add_action_node(index, selected)
            self.adapter.step(selected.action)
            after = self.adapter.observe()
            after_state = _state_for(after, selected.action.object_id)
            acceleration = (
                after_state.velocity[0] - before_state.velocity[0]
            ) / selected.action.duration
            self.estimators[selected.action.object_id].update(
                _force_norm(selected.action),
                acceleration,
            )
            frame = self._observe_frame(f"experiment_{index:03d}")
            mass_node = self._update_mass_belief(
                selected.action.object_id,
                index,
                acceleration,
            )
            audit = MetacognitiveAuditor(self.graph).audit(mass_node.id)
            estimator = self.estimators[selected.action.object_id]
            steps.append(
                ClosedLoopStep(
                    index=index,
                    selected=selected,
                    measured_acceleration=acceleration,
                    mass_mean=estimator.mean,
                    mass_std=math.sqrt(estimator.variance),
                    frame=frame,
                    audit=audit,
                )
            )

        estimates = {
            object_id: (
                estimator.mean,
                math.sqrt(estimator.variance),
            )
            for object_id, estimator in self.estimators.items()
        }
        true_masses = {
            object_id: self.body_mass(object_id) for object_id in self.body_names
        }
        relative_errors = {
            object_id: abs(estimates[object_id][0] - true_mass) / true_mass
            for object_id, true_mass in true_masses.items()
        }
        snapshot_id = (
            self.store.save(self.graph, label="closed-loop")
            if self.store is not None
            else None
        )
        return ClosedLoopResult(
            steps=tuple(steps),
            mass_estimates=estimates,
            true_masses=true_masses,
            relative_errors=relative_errors,
            successful=all(error <= 0.15 for error in relative_errors.values()),
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            snapshot_id=snapshot_id,
        )

    def body_mass(self, body_name: str) -> float:
        body_id = int(self.adapter.model.body(body_name).id)
        return float(self.adapter.model.body_mass[body_id])

    def close(self) -> None:
        self.adapter.close()

    def _observe_frame(self, label: str) -> EmbodiedFrameResult:
        frame = self.adapter.render_frame(
            frame_id=f"closed_loop_{self._frame_index:04d}_{label}",
            class_names={body_name: body_name for body_name in self.body_names},
            pixel_stride=4,
        )
        self._frame_index += 1
        proposals = self.perception.perceive(frame)
        tracking = self.tracker.update(proposals, timestamp=frame.timestamp)
        states = self.temporal.integrate(frame, proposals, tracking)
        self._bind_mass_nodes(tracking.tracks)
        return EmbodiedFrameResult(
            frame_id=frame.frame_id,
            proposals=proposals,
            tracking=tracking,
            state_node_ids=tuple(node.id for node in states),
        )

    def _ensure_dynamics_root(self) -> None:
        try:
            self.graph.find_node_by_name("mujoco_dynamics")
        except KeyError:
            self.graph.add_node(
                "mujoco_dynamics",
                NodeType.LAW,
                GaussianBelief(1.0, 1e-9),
                is_axiom=True,
                metadata={"backend": "mujoco", "scene": "embodied_scientist"},
            )

    def _bind_mass_nodes(self, tracks) -> None:
        root = self.graph.find_node_by_name("mujoco_dynamics")
        for track in tracks:
            body_name = track.class_name
            if body_name in self._mass_node_ids:
                continue
            object_node = self.temporal.object_node(track.track_id)
            estimator = self.estimators[body_name]
            mass_node = self.graph.add_node(
                f"{body_name}.mass",
                NodeType.PROPERTY,
                GaussianBelief(estimator.mean, estimator.variance),
                modality="inferred_physical",
                metadata={
                    "body_name": body_name,
                    "track_id": track.track_id,
                },
            )
            self.graph.add_edge(
                root.id,
                mass_node.id,
                EdgeType.CAUSES,
                noise_variance=1e-6,
            )
            self.graph.add_edge(
                object_node.id,
                mass_node.id,
                EdgeType.PART_OF,
                noise_variance=max(estimator.variance, 1e-6),
            )
            self._mass_node_ids[body_name] = mass_node.id

    def _add_action_node(
        self,
        index: int,
        selected: ActionPrimitiveScore,
    ) -> None:
        root = self.graph.find_node_by_name("mujoco_dynamics")
        action = selected.action
        evidence = GaussianBelief(_force_norm(action), 1e-9)
        plan = self.graph.add_node(
            f"efe_plan.{index:03d}",
            NodeType.GOAL,
            GaussianBelief(selected.expected_free_energy, 1e-6),
            evidence=GaussianBelief(selected.expected_free_energy, 1e-6),
            modality="planner",
            metadata={
                "epistemic_cost": selected.epistemic_cost,
                "pragmatic_cost": selected.pragmatic_cost,
                "predicted_displacement": selected.predicted_displacement,
                "predicted_visible_segments": selected.predicted_visible_segments,
            },
        )
        command = self.graph.add_node(
            f"executed_action.{index:03d}.{action.object_id}",
            NodeType.ACTION,
            evidence,
            evidence=evidence,
            modality="motor_command",
            metadata={
                "body_name": action.object_id,
                "force": action.force,
                "duration": action.duration,
                "executed": True,
            },
        )
        self.graph.add_edge(
            root.id,
            plan.id,
            EdgeType.CAUSES,
            noise_variance=1e-6,
            metadata={"planning_grounding": True},
        )
        self.graph.add_edge(
            plan.id,
            command.id,
            EdgeType.CAUSES,
            noise_variance=1e-6,
        )
        mass_node_id = self._mass_node_ids.get(action.object_id)
        if mass_node_id:
            self.graph.add_edge(
                command.id,
                mass_node_id,
                EdgeType.OBSERVES,
                noise_variance=1e-6,
                metadata={"action_grounding": True},
            )

    def _update_mass_belief(
        self,
        body_name: str,
        index: int,
        acceleration: float,
    ):
        estimator = self.estimators[body_name]
        mass_node = self.graph.require_node(self._mass_node_ids[body_name])
        new_belief = GaussianBelief(estimator.mean, estimator.variance)
        mass_node.belief = new_belief
        mass_node.prior = new_belief
        observation = self.graph.add_node(
            f"mass_observation.{index:03d}.{body_name}",
            NodeType.OBSERVATION,
            new_belief,
            evidence=new_belief,
            modality="action_outcome",
            metadata={
                "body_name": body_name,
                "acceleration": acceleration,
            },
        )
        self.graph.add_edge(
            observation.id,
            mass_node.id,
            EdgeType.OBSERVES,
            noise_variance=max(new_belief.variance, 1e-9),
            confidence=new_belief.confidence,
        )
        return mass_node


def _force_norm(action: ForceAction3D) -> float:
    return math.sqrt(sum(component * component for component in action.force))


def _state_for(observation, object_id: str):
    for state in observation.states:
        if state.object_id == object_id:
            return state
    raise KeyError(f"Missing state for object: {object_id}")
