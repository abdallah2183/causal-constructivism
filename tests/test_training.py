import json
import tempfile
import unittest
from pathlib import Path

from causal_constructivism.training import (
    LocalTraceModel,
    LocalTraceTrainer,
    TraceDatasetBuilder,
    TrainingDataset,
    TrainingExample,
    run_local_training,
)


class TrainingTests(unittest.TestCase):
    def test_dataset_builder_reads_programmer_and_website_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            programmer = root / "programmer.jsonl"
            website = root / "website.jsonl"
            programmer.write_text(
                json.dumps({
                    "task": "fix graph validation",
                    "target_files": ["src/graph.py", "tests/test_graph.py"],
                }) + "\n",
                encoding="utf-8",
            )
            website.write_text(
                json.dumps({
                    "prompt": "Build a website for a local code agent",
                    "title": "Local Code Agent",
                    "files": ["index.html", "styles.css"],
                    "sections": [{"section_id": "proof"}],
                }) + "\n",
                encoding="utf-8",
            )
            dataset = TraceDatasetBuilder().build((programmer, website))

        self.assertEqual(len(dataset.examples), 2)
        labels = {label for example in dataset.examples for label in example.labels}
        self.assertIn("file:src/graph.py", labels)
        self.assertIn("artifact:index.html", labels)
        self.assertIn("section:proof", labels)
        self.assertIn("title:Local Code Agent", labels)

    def test_local_trace_trainer_learns_prompt_label_associations(self) -> None:
        dataset = TrainingDataset((
            TrainingExample(
                source="memory",
                kind="programmer",
                prompt="fix graph validation",
                labels=("file:src/graph.py",),
            ),
            TrainingExample(
                source="memory",
                kind="website",
                prompt="build website landing page",
                labels=("artifact:index.html",),
            ),
        ))
        model = LocalTraceTrainer().train(dataset)
        predictions = model.predict("graph validation task", top_k=1)

        self.assertEqual(model.example_count, 2)
        self.assertEqual(predictions[0][0], "file:src/graph.py")

    def test_model_round_trip(self) -> None:
        dataset = TrainingDataset((
            TrainingExample(
                source="memory",
                kind="website",
                prompt="build local website",
                labels=("artifact:index.html", "artifact:styles.css"),
            ),
        ))
        model = LocalTraceTrainer().train(dataset)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            model.save(path)
            loaded = LocalTraceModel.load(path)

        self.assertEqual(loaded.version, model.version)
        self.assertEqual(loaded.predict("website", top_k=1)[0][0], "artifact:index.html")

    def test_run_local_training_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = root / "trace.jsonl"
            model_path = root / "model.json"
            dataset_path = root / "dataset.jsonl"
            trace.write_text(
                json.dumps({
                    "prompt": "Build a website for a local code agent",
                    "title": "Local Code Agent",
                    "files": ["index.html", "styles.css"],
                    "sections": [{"section_id": "proof"}],
                }) + "\n",
                encoding="utf-8",
            )
            report = run_local_training(
                (trace,),
                model_path=model_path,
                dataset_path=dataset_path,
                eval_prompt="build website",
            )

            self.assertEqual(report.status, "trained")
            self.assertTrue(model_path.exists())
            self.assertTrue(dataset_path.exists())
            self.assertGreater(len(report.predictions), 0)


if __name__ == "__main__":
    unittest.main()
