import math
import unittest

from causal_constructivism.models import GroundingStatus
from causal_constructivism.theorist import (
    EquationSearcher,
    Expression,
    pendulum_law_templates,
    run_theorist_benchmark,
    synthetic_linear_observations,
    synthetic_pendulum_observations,
)


class TheoristTests(unittest.TestCase):
    def test_expression_tree_evaluates_square_root_law_basis(self) -> None:
        expression = Expression(
            "sqrt",
            children=(
                Expression(
                    "div",
                    children=(Expression("var", name="L"), Expression("var", name="g")),
                ),
            ),
        )

        self.assertAlmostEqual(
            expression.evaluate({"L": 1.0, "g": 9.81}),
            math.sqrt(1.0 / 9.81),
        )

    def test_theorist_discovers_pendulum_square_root_law(self) -> None:
        result = run_theorist_benchmark(count=20, gravity=9.81)

        self.assertEqual(
            result.selected_law.template.name,
            "sqrt_length_over_gravity_period",
        )
        self.assertAlmostEqual(result.selected_law.parameters[0], 2.0 * math.pi)
        self.assertLess(result.selected_law.rmse, 1e-12)
        self.assertIs(result.audit.status, GroundingStatus.CONFIDENT)
        self.assertEqual(result.ungrounded_nodes, 0)

    def test_symbolic_search_rejects_overfit_polynomial_for_linear_data(self) -> None:
        fits = EquationSearcher().fit(
            synthetic_linear_observations(count=20, slope=2.0),
            pendulum_law_templates(),
        )

        self.assertEqual(fits[0].template.name, "linear_length_period")
        self.assertLess(fits[0].rmse, 1e-12)
        overfit = next(
            fit for fit in fits if fit.template.name == "overfit_polynomial_period"
        )
        self.assertLess(fits[0].score, overfit.score + 1000.0)
        self.assertGreater(fits[0].score, overfit.score)

    def test_low_information_observations_do_not_create_law(self) -> None:
        observations = synthetic_pendulum_observations(count=1)

        with self.assertRaises(ValueError):
            run_theorist_benchmark(count=len(observations))


if __name__ == "__main__":
    unittest.main()
