import json
import tempfile
import unittest
from pathlib import Path

from causal_constructivism.code_corpus import (
    build_code_corpus,
    is_secret_like_path,
    looks_secret_like,
)
from causal_constructivism.code_language_training import load_code_corpus


class CodeCorpusTests(unittest.TestCase):
    def test_build_code_corpus_writes_code_records_and_skips_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "demo"
            project.mkdir()
            (project / "app.py").write_text(
                "def hello(name: str) -> str:\n    message = f'hello {name}'\n    return message\n",
                encoding="utf-8",
            )
            (project / ".env").write_text(
                "API_KEY='this-should-not-enter-the-corpus'\n",
                encoding="utf-8",
            )
            (project / "config.py").write_text(
                "API_KEY = 'this-should-not-enter-the-corpus-value'\n",
                encoding="utf-8",
            )
            output = root / "corpus.jsonl"

            report = build_code_corpus((root,), output, max_files=10)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(report.written_examples, 1)
        self.assertEqual(report.skipped_secret_like, 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["language"], "python")
        self.assertEqual(payload["project"], "demo")
        self.assertIn("def hello", payload["code"])

    def test_secret_detection_catches_paths_and_inline_values(self) -> None:
        self.assertTrue(is_secret_like_path(Path("id_rsa")))
        self.assertTrue(is_secret_like_path(Path("service.pem")))
        self.assertTrue(looks_secret_like("OPENAI_API_KEY = 'sk-this-is-a-long-secret-value'"))

    def test_load_code_corpus_builds_byte_training_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corpus.jsonl"
            code = "print('hello')\n" * 100
            path.write_text(
                json.dumps({
                    "code": code,
                    "language": "python",
                    "project": "demo",
                    "source_path": "demo/app.py",
                }) + "\n",
                encoding="utf-8",
            )

            corpus = load_code_corpus(path, max_bytes=10_000)

        self.assertEqual(corpus.examples, 1)
        self.assertGreater(corpus.bytes_count, 1024)
        self.assertEqual(corpus.languages["python"], 1)


if __name__ == "__main__":
    unittest.main()
