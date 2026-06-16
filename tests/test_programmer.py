import json
import sys
import tempfile
import unittest
from pathlib import Path

from causal_constructivism.programmer import (
    CodeIndexer,
    FailureAnalyzer,
    LocalVerifier,
    ProgrammerCore,
    ProgrammerMemory,
    ProgrammingTask,
    TaskPlanner,
    run_programmer_benchmark,
)


class ProgrammerTests(unittest.TestCase):
    def make_project(self) -> tempfile.TemporaryDirectory[str]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "src" / "sample").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "src" / "sample" / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "sample" / "graph.py").write_text(
            "\n".join([
                "import json",
                "",
                "class Graph:",
                "    def add_edge(self, source, target):",
                "        return (source, target)",
                "",
                "def normalize_edge(edge):",
                "    return tuple(edge)",
            ]),
            encoding="utf-8",
        )
        (root / "tests" / "test_graph.py").write_text(
            "\n".join([
                "from sample.graph import normalize_edge",
                "",
                "def test_normalize_edge():",
                "    assert normalize_edge(['a', 'b']) == ('a', 'b')",
            ]),
            encoding="utf-8",
        )
        return temp

    def test_code_indexer_maps_symbols_and_imports(self) -> None:
        with self.make_project() as temp:
            index = CodeIndexer(Path(temp)).build()

        graph_module = next(
            module for module in index.modules if module.path == "src/sample/graph.py"
        )
        symbol_names = {symbol.name for symbol in graph_module.symbols}

        self.assertIn("json", graph_module.imports)
        self.assertIn("Graph", symbol_names)
        self.assertIn("add_edge", symbol_names)
        self.assertIn("normalize_edge", symbol_names)
        self.assertEqual(index.syntax_error_count, 0)

    def test_task_planner_targets_relevant_files(self) -> None:
        with self.make_project() as temp:
            index = CodeIndexer(Path(temp)).build()
            plan = TaskPlanner().plan(
                ProgrammingTask("fix graph add edge behavior and add graph test"),
                index,
            )

        self.assertIn("src/sample/graph.py", plan.target_files)
        self.assertIn("tests/test_graph.py", plan.target_files)
        self.assertIn(plan.risk, {"low", "medium", "high"})

    def test_verifier_and_failure_analyzer_capture_python_failure(self) -> None:
        with self.make_project() as temp:
            verifier = LocalVerifier(Path(temp))
            results = verifier.run((
                (
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('File \"broken.py\", line 7\\nAssertionError\\n'); sys.exit(1)",
                ),
            ))
            failures = FailureAnalyzer().analyze(results)

        self.assertFalse(results[0].passed)
        self.assertEqual(failures[0].source, "broken.py")
        self.assertEqual(failures[0].line, 7)

    def test_programmer_core_verifies_and_writes_memory(self) -> None:
        with self.make_project() as temp:
            memory_path = Path(temp) / "artifacts" / "programmer-memory.jsonl"
            report = ProgrammerCore(Path(temp)).inspect_task(
                "improve graph behavior",
                run_compile=True,
                run_tests=False,
                memory_path=memory_path,
            )
            memory_line = memory_path.read_text(encoding="utf-8").strip()

        payload = json.loads(memory_line)
        self.assertEqual(report.status, "verified")
        self.assertEqual(payload["status"], "verified")
        self.assertIn("src/sample/graph.py", report.plan.target_files)

    def test_programmer_benchmark_runs_on_temp_project(self) -> None:
        with self.make_project() as temp:
            report = run_programmer_benchmark(
                temp,
                "add graph programming support",
                run_tests=False,
            )

        self.assertEqual(report.status, "verified")
        self.assertGreaterEqual(report.index.symbol_count, 3)


if __name__ == "__main__":
    unittest.main()
