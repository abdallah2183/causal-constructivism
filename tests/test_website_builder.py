import json
import tempfile
import unittest
from pathlib import Path

from causal_constructivism.website_builder import (
    OnePromptWebsiteBuilder,
    WebsiteBuildResult,
    WebsitePrompt,
    run_website_builder_benchmark,
)


class WebsiteBuilderTests(unittest.TestCase):
    def test_prompt_keywords_are_stable(self) -> None:
        prompt = WebsitePrompt(
            "Build a complete website for a local cognitive engine with GPU memory."
        )

        self.assertEqual(prompt.keywords[:4], ("local", "cognitive", "engine", "gpu"))

    def test_builder_writes_complete_static_site(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = OnePromptWebsiteBuilder(Path(temp)).build(
                "Build a complete website for Causal Constructivism.",
                slug="demo-site",
            )
            output = Path(result.output_dir)

            self.assertIsInstance(result, WebsiteBuildResult)
            self.assertEqual(result.slug, "demo-site")
            self.assertTrue((output / "index.html").exists())
            self.assertTrue((output / "styles.css").exists())
            self.assertTrue((output / "app.js").exists())
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "README.md").exists())

            html = (output / "index.html").read_text(encoding="utf-8")
            css = (output / "styles.css").read_text(encoding="utf-8")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertIn("Causal Constructivism", html)
        self.assertIn("oklch", css)
        self.assertNotIn("#000", css)
        self.assertNotIn("#fff", css)
        self.assertNotIn("background-clip: text", css)
        self.assertFalse(manifest["claims"]["neural_training"])
        self.assertTrue(manifest["claims"]["static_site"])

    def test_builder_records_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.jsonl"
            result = run_website_builder_benchmark(
                "Build a website for a verified local code agent.",
                output_dir=Path(temp) / "sites",
                slug="agent",
                trace_path=trace,
            )
            payload = json.loads(trace.read_text(encoding="utf-8").strip())

        self.assertEqual(result.trace_path, str(trace))
        self.assertEqual(payload["slug"], "agent")
        self.assertIn("index.html", payload["files"])


if __name__ == "__main__":
    unittest.main()
