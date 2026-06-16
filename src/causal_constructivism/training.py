from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .programmer import AcceleratorProbe, AcceleratorProfile


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    source: str
    kind: str
    prompt: str
    labels: tuple[str, ...]

    def tokens(self) -> tuple[str, ...]:
        stop = {
            "and",
            "for",
            "from",
            "that",
            "the",
            "this",
            "with",
        }
        return tuple(
            token
            for token in TOKEN_RE.findall(self.prompt.lower())
            if len(token) > 2 and token not in stop
        )


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    examples: tuple[TrainingExample, ...]

    @property
    def label_count(self) -> int:
        labels = {label for example in self.examples for label in example.labels}
        return len(labels)

    @property
    def token_count(self) -> int:
        tokens = {token for example in self.examples for token in example.tokens()}
        return len(tokens)


@dataclass(frozen=True, slots=True)
class LocalTraceModel:
    version: str
    example_count: int
    label_counts: dict[str, int]
    token_label_counts: dict[str, dict[str, int]]
    token_counts: dict[str, int]

    def predict(self, prompt: str, top_k: int = 8) -> tuple[tuple[str, float], ...]:
        tokens = [
            token
            for token in TOKEN_RE.findall(prompt.lower())
            if len(token) > 2
        ]
        labels = sorted(self.label_counts)
        if not labels:
            return ()

        vocab_size = max(len(self.token_counts), 1)
        total_labels = sum(self.label_counts.values())
        scores: list[tuple[str, float]] = []
        for label in labels:
            label_count = self.label_counts[label]
            score = math.log((label_count + 1) / (total_labels + len(labels)))
            for token in tokens:
                token_label_count = self.token_label_counts.get(token, {}).get(label, 0)
                score += math.log((token_label_count + 1) / (label_count + vocab_size))
            scores.append((label, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return tuple(scores[:top_k])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> LocalTraceModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=payload["version"],
            example_count=payload["example_count"],
            label_counts={str(k): int(v) for k, v in payload["label_counts"].items()},
            token_label_counts={
                str(token): {str(label): int(count) for label, count in labels.items()}
                for token, labels in payload["token_label_counts"].items()
            },
            token_counts={str(k): int(v) for k, v in payload["token_counts"].items()},
        )


@dataclass(frozen=True, slots=True)
class TrainingReport:
    status: str
    examples: int
    labels: int
    tokens: int
    model_path: str
    dataset_path: str | None
    accelerator: AcceleratorProfile
    accelerator_used: bool
    predictions: tuple[tuple[str, float], ...]


class TraceDatasetBuilder:
    """Converts local execution traces into supervised examples."""

    def build(
        self,
        paths: Iterable[Path],
        *,
        include_missing: bool = False,
    ) -> TrainingDataset:
        examples: list[TrainingExample] = []
        for path in paths:
            if not path.exists():
                if include_missing:
                    continue
                raise FileNotFoundError(path)
            if path.suffix == ".jsonl":
                examples.extend(self._from_jsonl(path))
            elif path.suffix == ".json":
                examples.extend(self._from_json(path))
        return TrainingDataset(tuple(examples))

    def _from_jsonl(self, path: Path) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.extend(self._from_payload(path.as_posix(), payload))
        return examples

    def _from_json(self, path: Path) -> list[TrainingExample]:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._from_payload(path.as_posix(), payload)

    def _from_payload(self, source: str, payload: dict[str, Any]) -> list[TrainingExample]:
        if "target_files" in payload and "task" in payload:
            return [
                TrainingExample(
                    source=source,
                    kind="programmer",
                    prompt=str(payload["task"]),
                    labels=tuple(f"file:{path}" for path in payload.get("target_files", [])),
                )
            ]
        if "task" in payload and isinstance(payload.get("plan"), dict):
            target_files = payload["plan"].get("target_files", [])
            if target_files:
                return [
                    TrainingExample(
                        source=source,
                        kind="programmer",
                        prompt=str(payload["task"]),
                        labels=tuple(f"file:{path}" for path in target_files),
                    )
                ]
        if "files" in payload and "prompt" in payload:
            labels = [f"artifact:{name}" for name in payload.get("files", [])]
            labels.extend(
                f"section:{section.get('section_id', section.get('id', 'unknown'))}"
                for section in payload.get("sections", [])
            )
            title = payload.get("title")
            if title:
                labels.append(f"title:{title}")
            return [
                TrainingExample(
                    source=source,
                    kind="website",
                    prompt=str(payload["prompt"]),
                    labels=tuple(labels),
                )
            ]
        return []

    @staticmethod
    def save(dataset: TrainingDataset, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in dataset.examples:
                handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")
        return path


class LocalTraceTrainer:
    """Trains a small local prompt-to-artifact model from verified traces."""

    def train(self, dataset: TrainingDataset) -> LocalTraceModel:
        label_counts: Counter[str] = Counter()
        token_counts: Counter[str] = Counter()
        token_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for example in dataset.examples:
            labels = set(example.labels)
            label_counts.update(labels)
            for token in example.tokens():
                token_counts[token] += 1
                for label in labels:
                    token_label_counts[token][label] += 1

        return LocalTraceModel(
            version="local-trace-model-v1",
            example_count=len(dataset.examples),
            label_counts=dict(sorted(label_counts.items())),
            token_label_counts={
                token: dict(sorted(labels.items()))
                for token, labels in sorted(token_label_counts.items())
            },
            token_counts=dict(sorted(token_counts.items())),
        )


def run_local_training(
    trace_paths: Iterable[Path],
    *,
    model_path: Path,
    dataset_path: Path | None = None,
    eval_prompt: str = "Build a website and update graph tests.",
) -> TrainingReport:
    dataset = TraceDatasetBuilder().build(trace_paths, include_missing=False)
    if not dataset.examples:
        raise ValueError("No training examples were found in the supplied traces.")
    if dataset_path is not None:
        TraceDatasetBuilder.save(dataset, dataset_path)

    model = LocalTraceTrainer().train(dataset)
    model.save(model_path)
    accelerator = AcceleratorProbe().probe()
    predictions = model.predict(eval_prompt)

    return TrainingReport(
        status="trained",
        examples=len(dataset.examples),
        labels=dataset.label_count,
        tokens=dataset.token_count,
        model_path=str(model_path),
        dataset_path=str(dataset_path) if dataset_path else None,
        accelerator=accelerator,
        accelerator_used=False,
        predictions=predictions,
    )
