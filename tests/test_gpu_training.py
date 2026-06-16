import json
import tempfile
import unittest
from pathlib import Path

from causal_constructivism.gpu_training import (
    TraceExample,
    augment_examples,
    build_parser,
    build_vocab,
    load_trace_examples,
    tokenize,
)


class GpuTrainingTests(unittest.TestCase):
    def test_load_trace_examples_reads_jsonl_prompts_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.jsonl"
            path.write_text(
                json.dumps({
                    "prompt": "Build a local verifier",
                    "labels": ["artifact:index.html", "section:proof"],
                }) + "\n",
                encoding="utf-8",
            )

            examples = load_trace_examples(path)

        self.assertEqual(examples, [
            TraceExample(
                prompt="Build a local verifier",
                labels=("artifact:index.html", "section:proof"),
            )
        ])

    def test_augment_examples_is_deterministic_and_preserves_labels(self) -> None:
        examples = [TraceExample(prompt="build website locally", labels=("artifact:index.html",))]

        first = augment_examples(examples, copies=3)
        second = augment_examples(examples, copies=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(example.labels == ("artifact:index.html",) for example in first))

    def test_build_vocab_includes_padding_unknown_tokens_and_labels(self) -> None:
        examples = [
            TraceExample(
                prompt="Build GPU trace trainer",
                labels=("phase:20", "artifact:checkpoint"),
            )
        ]

        vocab, labels = build_vocab(examples)

        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<unk>"], 1)
        self.assertIn("gpu", vocab)
        self.assertEqual(set(labels), {"phase:20", "artifact:checkpoint"})

    def test_tokenize_keeps_words_and_underscores(self) -> None:
        self.assertEqual(tokenize("GPU trace_model v2!"), ["gpu", "trace_model", "v2"])

    def test_parser_supports_long_running_cuda_options(self) -> None:
        args = build_parser().parse_args([
            "--output-dir",
            "artifacts/gpu-training-long",
            "--duration-seconds",
            "21600",
            "--batch-size",
            "1024",
            "--require-cuda",
            "--no-amp",
        ])

        self.assertEqual(args.output_dir, "artifacts/gpu-training-long")
        self.assertEqual(args.duration_seconds, 21600)
        self.assertEqual(args.batch_size, 1024)
        self.assertTrue(args.require_cuda)
        self.assertFalse(args.amp)


if __name__ == "__main__":
    unittest.main()
