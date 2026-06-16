from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
}


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    name: str
    kind: str
    path: str
    line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class ModuleIndex:
    path: str
    imports: tuple[str, ...]
    symbols: tuple[CodeSymbol, ...]
    syntax_error: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectIndex:
    root: str
    modules: tuple[ModuleIndex, ...]

    @property
    def source_files(self) -> tuple[str, ...]:
        return tuple(module.path for module in self.modules if module.path.startswith("src/"))

    @property
    def test_files(self) -> tuple[str, ...]:
        return tuple(module.path for module in self.modules if module.path.startswith("tests/"))

    @property
    def symbol_count(self) -> int:
        return sum(len(module.symbols) for module in self.modules)

    @property
    def syntax_error_count(self) -> int:
        return sum(1 for module in self.modules if module.syntax_error is not None)


@dataclass(frozen=True, slots=True)
class ProgrammingTask:
    text: str

    def tokens(self) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", self.text.lower())
            if len(token) > 2
        }


@dataclass(frozen=True, slots=True)
class PatchPlan:
    task: ProgrammingTask
    target_files: tuple[str, ...]
    rationale: tuple[str, ...]
    required_checks: tuple[tuple[str, ...], ...]
    risk: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class FailureFinding:
    source: str
    line: int | None
    message: str


@dataclass(frozen=True, slots=True)
class AcceleratorProfile:
    available: bool
    name: str | None = None
    total_memory_mib: int | None = None
    driver_version: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProgrammerReport:
    task: ProgrammingTask
    index: ProjectIndex
    plan: PatchPlan
    verification: tuple[VerificationResult, ...]
    failures: tuple[FailureFinding, ...]
    accelerator: AcceleratorProfile
    memory_written: str | None = None

    @property
    def status(self) -> str:
        if self.index.syntax_error_count:
            return "blocked"
        if self.verification and all(result.passed for result in self.verification):
            return "verified"
        if self.verification:
            return "failed"
        return "planned"


class CodeIndexer:
    """Builds a local AST index over Python source files."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def iter_python_files(self) -> Iterable[Path]:
        for path in sorted(self.root.rglob("*.py")):
            if any(part in EXCLUDED_DIRS for part in path.relative_to(self.root).parts):
                continue
            yield path

    def build(self) -> ProjectIndex:
        modules = tuple(self._index_file(path) for path in self.iter_python_files())
        return ProjectIndex(
            root=str(self.root),
            modules=modules,
        )

    def _index_file(self, path: Path) -> ModuleIndex:
        relative = path.relative_to(self.root).as_posix()
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as exc:
            return ModuleIndex(
                path=relative,
                imports=(),
                symbols=(),
                syntax_error=f"{exc.msg} at line {exc.lineno}",
            )

        imports: list[str] = []
        symbols: list[CodeSymbol] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append("." * node.level + module)
            elif isinstance(node, ast.ClassDef):
                symbols.append(self._symbol(node.name, "class", relative, node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._symbol(node.name, "function", relative, node))

        return ModuleIndex(
            path=relative,
            imports=tuple(sorted(set(imports))),
            symbols=tuple(sorted(symbols, key=lambda symbol: (symbol.line, symbol.name))),
        )

    @staticmethod
    def _symbol(name: str, kind: str, path: str, node: ast.AST) -> CodeSymbol:
        line = getattr(node, "lineno", 0)
        end_line = getattr(node, "end_lineno", line)
        return CodeSymbol(
            name=name,
            kind=kind,
            path=path,
            line=line,
            end_line=end_line,
        )


class TaskPlanner:
    """Selects likely files for a programming task from local code evidence."""

    def plan(self, task: ProgrammingTask, index: ProjectIndex) -> PatchPlan:
        tokens = task.tokens()
        scored: list[tuple[int, str, list[str]]] = []
        for module in index.modules:
            score, reasons = self._score_module(tokens, module)
            if score > 0:
                scored.append((score, module.path, reasons))

        scored.sort(key=lambda item: (-item[0], item[1]))
        target_files = self._with_related_tests(
            tuple(path for _, path, _ in scored[:8]),
            index,
        )
        rationale = tuple(
            f"{path}: {', '.join(reasons[:3])}"
            for _, path, reasons in scored[:8]
        )
        if not target_files:
            target_files = tuple(index.source_files[:5])
            rationale = ("No strong lexical match; start with core source files.",)

        required_checks = (
            (sys.executable, "-m", "compileall", "-q", "src", "tests"),
            ("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/test.ps1"),
        )
        risk = self._risk(task, target_files)
        return PatchPlan(
            task=task,
            target_files=target_files,
            rationale=rationale,
            required_checks=required_checks,
            risk=risk,
        )

    @staticmethod
    def _with_related_tests(
        target_files: tuple[str, ...],
        index: ProjectIndex,
    ) -> tuple[str, ...]:
        available = {module.path for module in index.modules}
        expanded: list[str] = []
        for path in target_files:
            if path not in expanded:
                expanded.append(path)
            if path.startswith("src/") and path.endswith(".py"):
                stem = Path(path).stem
                related_test = f"tests/test_{stem}.py"
                if related_test in available and related_test not in expanded:
                    expanded.append(related_test)
        return tuple(expanded[:8])

    @staticmethod
    def _score_module(tokens: set[str], module: ModuleIndex) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        module_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", module.path.lower()))
        path_hits = tokens & module_tokens
        if path_hits:
            score += 8 * len(path_hits)
            reasons.append(f"path matches {sorted(path_hits)}")

        for symbol in module.symbols:
            symbol_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", symbol.name.lower()))
            hits = tokens & symbol_tokens
            if hits:
                score += 3 * len(hits)
                reasons.append(f"{symbol.kind} {symbol.name} matches {sorted(hits)}")

        import_hits = tokens & {
            part
            for imported in module.imports
            for part in imported.lower().replace(".", "_").split("_")
        }
        if import_hits:
            score += len(import_hits)
            reasons.append(f"imports match {sorted(import_hits)}")

        return score, reasons

    @staticmethod
    def _risk(task: ProgrammingTask, target_files: tuple[str, ...]) -> str:
        text = task.text.lower()
        if any(word in text for word in ("delete", "remove", "rewrite", "migration")):
            return "high"
        if any(path.startswith("src/causal_constructivism/") for path in target_files):
            return "medium"
        return "low"


class LocalVerifier:
    """Runs local verification commands and captures evidence."""

    def __init__(self, root: Path, timeout_seconds: int = 120) -> None:
        self.root = root.resolve()
        self.timeout_seconds = timeout_seconds

    def run(self, commands: Iterable[tuple[str, ...]]) -> tuple[VerificationResult, ...]:
        results: list[VerificationResult] = []
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            results.append(VerificationResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ))
        return tuple(results)


class FailureAnalyzer:
    """Extracts actionable failure hints from command output."""

    FILE_LINE_RE = re.compile(r'File "([^"]+)", line (\d+)')
    PYTEST_RE = re.compile(r"^(FAILED|ERROR)\s+(.+)$")

    def analyze(self, results: Iterable[VerificationResult]) -> tuple[FailureFinding, ...]:
        findings: list[FailureFinding] = []
        for result in results:
            if result.passed:
                continue
            output = f"{result.stdout}\n{result.stderr}"
            findings.extend(self._from_output(output))
            if not findings:
                findings.append(FailureFinding(
                    source=" ".join(result.command),
                    line=None,
                    message=f"command failed with exit code {result.returncode}",
                ))
        return tuple(findings)

    def _from_output(self, output: str) -> list[FailureFinding]:
        findings: list[FailureFinding] = []
        for line in output.splitlines():
            file_match = self.FILE_LINE_RE.search(line)
            if file_match:
                findings.append(FailureFinding(
                    source=file_match.group(1),
                    line=int(file_match.group(2)),
                    message=line.strip(),
                ))
                continue
            pytest_match = self.PYTEST_RE.match(line.strip())
            if pytest_match:
                findings.append(FailureFinding(
                    source=pytest_match.group(2),
                    line=None,
                    message=pytest_match.group(1).lower(),
                ))
        return findings


class AcceleratorProbe:
    """Detects local NVIDIA acceleration without requiring extra dependencies."""

    def probe(self) -> AcceleratorProfile:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return AcceleratorProfile(
                available=False,
                reason=str(exc),
            )

        if completed.returncode != 0:
            return AcceleratorProfile(
                available=False,
                reason=completed.stderr.strip() or "nvidia-smi failed",
            )
        line = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        memory = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        return AcceleratorProfile(
            available=True,
            name=parts[0] if parts else None,
            total_memory_mib=memory,
            driver_version=parts[2] if len(parts) > 2 else None,
        )


class ProgrammerMemory:
    """Append-only JSONL memory for successful and failed programming traces."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, report: ProgrammerReport) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": report.task.text,
            "status": report.status,
            "target_files": list(report.plan.target_files),
            "risk": report.plan.risk,
            "verification": [
                {
                    "command": list(result.command),
                    "returncode": result.returncode,
                    "passed": result.passed,
                }
                for result in report.verification
            ],
            "failures": [asdict(failure) for failure in report.failures],
            "accelerator": asdict(report.accelerator),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return self.path


class ProgrammerCore:
    """Phase 17 local programming core: inspect, plan, verify, and remember."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def inspect_task(
        self,
        task_text: str,
        *,
        run_compile: bool = True,
        run_tests: bool = False,
        memory_path: Path | None = None,
    ) -> ProgrammerReport:
        task = ProgrammingTask(task_text)
        index = CodeIndexer(self.root).build()
        plan = TaskPlanner().plan(task, index)

        commands: list[tuple[str, ...]] = []
        if run_compile:
            commands.append(plan.required_checks[0])
        if run_tests:
            commands.append(plan.required_checks[1])

        verification = LocalVerifier(self.root).run(commands) if commands else ()
        failures = FailureAnalyzer().analyze(verification)
        accelerator = AcceleratorProbe().probe()

        report = ProgrammerReport(
            task=task,
            index=index,
            plan=plan,
            verification=verification,
            failures=failures,
            accelerator=accelerator,
        )
        written: str | None = None
        if memory_path is not None:
            written = str(ProgrammerMemory(memory_path).append(report))
        return ProgrammerReport(
            task=report.task,
            index=report.index,
            plan=report.plan,
            verification=report.verification,
            failures=report.failures,
            accelerator=report.accelerator,
            memory_written=written,
        )


def run_programmer_benchmark(
    root: Path | str = ".",
    task: str = "Add a verified programming capability to this Python project.",
    *,
    run_tests: bool = False,
    memory_path: Path | None = None,
) -> ProgrammerReport:
    return ProgrammerCore(Path(root)).inspect_task(
        task,
        run_compile=True,
        run_tests=run_tests,
        memory_path=memory_path,
    )
