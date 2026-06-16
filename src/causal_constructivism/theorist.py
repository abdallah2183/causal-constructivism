from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .audit import MetacognitiveAuditor
from .graph import CausalGraph
from .models import EdgeType, GaussianBelief, GroundingAudit, NodeType


@dataclass(frozen=True, slots=True)
class Expression:
    op: str
    children: tuple[Expression, ...] = ()
    value: float | None = None
    name: str | None = None

    def evaluate(self, variables: dict[str, float]) -> float:
        if self.op == "const":
            if self.value is None:
                raise ValueError("Constant expression requires a value")
            return self.value
        if self.op == "var":
            if self.name is None:
                raise ValueError("Variable expression requires a name")
            try:
                return variables[self.name]
            except KeyError as exc:
                raise KeyError(f"Missing variable: {self.name}") from exc
        values = [child.evaluate(variables) for child in self.children]
        if self.op == "add":
            return sum(values)
        if self.op == "sub":
            return values[0] - values[1]
        if self.op == "mul":
            result = 1.0
            for value in values:
                result *= value
            return result
        if self.op == "div":
            denominator = values[1]
            if abs(denominator) < 1e-12:
                raise ZeroDivisionError("Expression division by zero")
            return values[0] / denominator
        if self.op == "pow":
            return values[0] ** values[1]
        if self.op == "sqrt":
            if values[0] < 0:
                raise ValueError("Cannot take square root of a negative value")
            return math.sqrt(values[0])
        if self.op == "exp":
            return math.exp(values[0])
        if self.op == "log":
            if values[0] <= 0:
                raise ValueError("Cannot take log of a non-positive value")
            return math.log(values[0])
        if self.op == "sin":
            return math.sin(values[0])
        if self.op == "cos":
            return math.cos(values[0])
        raise ValueError(f"Unsupported expression op: {self.op}")

    @property
    def complexity(self) -> int:
        return 1 + sum(child.complexity for child in self.children)

    def render(self) -> str:
        if self.op == "const":
            if self.value is None:
                raise ValueError("Constant expression requires a value")
            return f"{self.value:g}"
        if self.op == "var":
            if self.name is None:
                raise ValueError("Variable expression requires a name")
            return self.name
        if self.op == "add":
            return "(" + " + ".join(child.render() for child in self.children) + ")"
        if self.op == "sub":
            return f"({self.children[0].render()} - {self.children[1].render()})"
        if self.op == "mul":
            return "(" + " * ".join(child.render() for child in self.children) + ")"
        if self.op == "div":
            return f"({self.children[0].render()} / {self.children[1].render()})"
        if self.op == "pow":
            return f"({self.children[0].render()} ^ {self.children[1].render()})"
        if self.op in {"sqrt", "exp", "log", "sin", "cos"}:
            return f"{self.op}({self.children[0].render()})"
        raise ValueError(f"Unsupported expression op: {self.op}")


def var(name: str) -> Expression:
    return Expression("var", name=name)


def const(value: float) -> Expression:
    return Expression("const", value=value)


def mul(*children: Expression) -> Expression:
    return Expression("mul", children=tuple(children))


def div(left: Expression, right: Expression) -> Expression:
    return Expression("div", children=(left, right))


def pow_expr(base: Expression, exponent: float) -> Expression:
    return Expression("pow", children=(base, const(exponent)))


def sqrt(child: Expression) -> Expression:
    return Expression("sqrt", children=(child,))


@dataclass(frozen=True, slots=True)
class SymbolicObservation:
    variables: dict[str, float]
    target: float

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("Symbolic observation requires variables")
        if not math.isfinite(self.target):
            raise ValueError("Observation target must be finite")
        for name, value in self.variables.items():
            if not name:
                raise ValueError("Variable names must be non-empty")
            if not math.isfinite(value):
                raise ValueError(f"Variable {name} must be finite")


@dataclass(frozen=True, slots=True)
class LawTemplate:
    name: str
    basis: tuple[Expression, ...]
    target_name: str
    dimensionally_consistent: bool = False

    @property
    def complexity(self) -> int:
        return len(self.basis) + sum(expression.complexity for expression in self.basis)

    def render(self, parameters: tuple[float, ...]) -> str:
        terms = [
            f"{parameter:.6g}*{expression.render()}"
            for parameter, expression in zip(parameters, self.basis, strict=True)
        ]
        return f"{self.target_name} = " + " + ".join(terms)


@dataclass(frozen=True, slots=True)
class LawFit:
    template: LawTemplate
    parameters: tuple[float, ...]
    expression: str
    rmse: float
    evidence_gain: float
    complexity_penalty: float
    scope_bonus: float

    @property
    def score(self) -> float:
        return self.evidence_gain + self.scope_bonus - self.complexity_penalty


@dataclass(frozen=True, slots=True)
class TheoristResult:
    selected_law: LawFit
    candidate_laws: tuple[LawFit, ...]
    law_node_id: str
    audit: GroundingAudit
    graph_nodes: int
    graph_edges: int
    ungrounded_nodes: int


class EquationSearcher:
    def __init__(
        self,
        *,
        complexity_weight: float = 0.12,
        scope_weight: float = 0.25,
    ) -> None:
        self.complexity_weight = complexity_weight
        self.scope_weight = scope_weight

    def fit(
        self,
        observations: Iterable[SymbolicObservation],
        templates: Iterable[LawTemplate],
    ) -> tuple[LawFit, ...]:
        rows = tuple(observations)
        candidates = tuple(templates)
        if not rows:
            raise ValueError("Equation search requires observations")
        if not candidates:
            raise ValueError("Equation search requires templates")
        baseline_sse = _constant_baseline_sse(rows)
        fits = []
        for template in candidates:
            parameters = _least_squares(rows, template.basis)
            residual_sse = _residual_sse(rows, template.basis, parameters)
            evidence_gain = 0.5 * len(rows) * math.log(
                max(baseline_sse, 1e-12) / max(residual_sse, 1e-12)
            )
            complexity_penalty = (
                self.complexity_weight
                * template.complexity
                * math.log(len(rows) + 1)
            )
            scope_bonus = self.scope_weight if template.dimensionally_consistent else 0.0
            fits.append(
                LawFit(
                    template=template,
                    parameters=parameters,
                    expression=template.render(parameters),
                    rmse=math.sqrt(residual_sse / len(rows)),
                    evidence_gain=evidence_gain,
                    complexity_penalty=complexity_penalty,
                    scope_bonus=scope_bonus,
                )
            )
        return tuple(sorted(fits, key=lambda fit: fit.score, reverse=True))


class TheoristSystem:
    def __init__(
        self,
        *,
        graph: CausalGraph | None = None,
        searcher: EquationSearcher | None = None,
    ) -> None:
        self.graph = graph or CausalGraph()
        self.searcher = searcher or EquationSearcher()

    def discover_law(
        self,
        observations: Iterable[SymbolicObservation],
        templates: Iterable[LawTemplate],
        *,
        phenomenon: str,
        threshold: float = 5.0,
    ) -> TheoristResult:
        rows = tuple(observations)
        fits = self.searcher.fit(rows, templates)
        selected = fits[0]
        if selected.score <= threshold:
            raise ValueError("No symbolic law cleared the discovery threshold")
        law_node = self._integrate_law(selected, rows, phenomenon=phenomenon)
        audit = MetacognitiveAuditor(self.graph).audit(law_node.id)
        return TheoristResult(
            selected_law=selected,
            candidate_laws=fits,
            law_node_id=law_node.id,
            audit=audit,
            graph_nodes=len(self.graph.nodes),
            graph_edges=len(self.graph.edges),
            ungrounded_nodes=len(self.graph.validate_grounding()),
        )

    def discover_pendulum_law(
        self,
        observations: Iterable[SymbolicObservation],
    ) -> TheoristResult:
        return self.discover_law(
            observations,
            pendulum_law_templates(),
            phenomenon="pendulum_period",
        )

    def _integrate_law(
        self,
        law: LawFit,
        observations: tuple[SymbolicObservation, ...],
        *,
        phenomenon: str,
    ):
        observation_node = self.graph.add_node(
            f"theorist_observation.{phenomenon}",
            NodeType.OBSERVATION,
            GaussianBelief(law.rmse, 1e-6),
            evidence=GaussianBelief(law.rmse, 1e-6),
            modality="symbolic_regression",
            metadata={
                "observations": len(observations),
                "rmse": law.rmse,
                "target": law.template.target_name,
            },
        )
        consistency_node = self.graph.add_node(
            f"theorist_consistency.{law.template.name}",
            NodeType.LAW,
            GaussianBelief(1.0, 1e-9),
            is_axiom=True,
            modality="theoretical_constraint",
            metadata={
                "dimensionally_consistent": law.template.dimensionally_consistent,
                "search_space": "bounded_symbolic_regression",
            },
        )
        law_node = self.graph.add_node(
            f"discovered_law.{law.template.name}",
            NodeType.LAW,
            GaussianBelief(law.score, max(1e-6, 1.0 / (1.0 + law.evidence_gain))),
            modality="discovered_symbolic_law",
            metadata={
                "phenomenon": phenomenon,
                "expression": law.expression,
                "parameters": law.parameters,
                "rmse": law.rmse,
                "evidence_gain": law.evidence_gain,
                "complexity_penalty": law.complexity_penalty,
                "scope_bonus": law.scope_bonus,
            },
        )
        self.graph.add_edge(
            observation_node.id,
            law_node.id,
            EdgeType.OBSERVES,
            noise_variance=max(1e-6, law.rmse * law.rmse + 1e-6),
            confidence=law_node.belief.confidence,
            learned=True,
        )
        self.graph.add_edge(
            consistency_node.id,
            law_node.id,
            EdgeType.CAUSES,
            noise_variance=1e-6,
            learned=True,
        )
        return law_node


def run_theorist_benchmark(
    *,
    count: int = 20,
    gravity: float = 9.81,
) -> TheoristResult:
    return TheoristSystem().discover_pendulum_law(
        synthetic_pendulum_observations(count=count, gravity=gravity)
    )


def pendulum_law_templates() -> tuple[LawTemplate, ...]:
    length = var("L")
    gravity = var("g")
    return (
        LawTemplate(
            name="constant_period",
            basis=(const(1.0),),
            target_name="T",
        ),
        LawTemplate(
            name="linear_length_period",
            basis=(length,),
            target_name="T",
        ),
        LawTemplate(
            name="quadratic_length_period",
            basis=(length, pow_expr(length, 2.0)),
            target_name="T",
        ),
        LawTemplate(
            name="sqrt_length_over_gravity_period",
            basis=(sqrt(div(length, gravity)),),
            target_name="T",
            dimensionally_consistent=True,
        ),
        LawTemplate(
            name="overfit_polynomial_period",
            basis=(const(1.0), length, pow_expr(length, 2.0), pow_expr(length, 3.0)),
            target_name="T",
        ),
    )


def synthetic_pendulum_observations(
    *,
    count: int = 20,
    gravity: float = 9.81,
) -> tuple[SymbolicObservation, ...]:
    if count <= 0:
        raise ValueError("Observation count must be positive")
    if gravity <= 0:
        raise ValueError("Gravity must be positive")
    rows = []
    for index in range(count):
        length = 0.25 + 0.08 * index
        period = 2.0 * math.pi * math.sqrt(length / gravity)
        rows.append(
            SymbolicObservation(
                variables={"L": length, "g": gravity},
                target=period,
            )
        )
    return tuple(rows)


def synthetic_linear_observations(
    *,
    count: int = 20,
    slope: float = 2.0,
    gravity: float = 9.81,
) -> tuple[SymbolicObservation, ...]:
    if count <= 0:
        raise ValueError("Observation count must be positive")
    rows = []
    for index in range(count):
        length = 0.25 + 0.08 * index
        rows.append(
            SymbolicObservation(
                variables={"L": length, "g": gravity},
                target=slope * length,
            )
        )
    return tuple(rows)


def _constant_baseline_sse(rows: tuple[SymbolicObservation, ...]) -> float:
    mean = sum(row.target for row in rows) / len(rows)
    return sum((row.target - mean) ** 2 for row in rows)


def _design_matrix(
    rows: tuple[SymbolicObservation, ...],
    basis: tuple[Expression, ...],
) -> list[list[float]]:
    return [
        [expression.evaluate(row.variables) for expression in basis]
        for row in rows
    ]


def _least_squares(
    rows: tuple[SymbolicObservation, ...],
    basis: tuple[Expression, ...],
) -> tuple[float, ...]:
    matrix = _design_matrix(rows, basis)
    targets = [row.target for row in rows]
    columns = len(basis)
    normal = [
        [
            sum(row[i] * row[j] for row in matrix)
            for j in range(columns)
        ]
        for i in range(columns)
    ]
    rhs = [
        sum(row[i] * target for row, target in zip(matrix, targets, strict=True))
        for i in range(columns)
    ]
    return tuple(_solve_linear_system(normal, rhs))


def _residual_sse(
    rows: tuple[SymbolicObservation, ...],
    basis: tuple[Expression, ...],
    parameters: tuple[float, ...],
) -> float:
    total = 0.0
    for row in rows:
        prediction = sum(
            parameter * expression.evaluate(row.variables)
            for parameter, expression in zip(parameters, basis, strict=True)
        )
        total += (row.target - prediction) ** 2
    return total


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return [0.0 for _ in range(size)]
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[row][size] for row in range(size)]
