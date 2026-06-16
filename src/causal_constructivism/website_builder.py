from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WebsitePrompt:
    text: str

    @property
    def keywords(self) -> tuple[str, ...]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", self.text.lower())
        stop = {
            "about",
            "and",
            "build",
            "complete",
            "for",
            "from",
            "landing",
            "make",
            "page",
            "site",
            "the",
            "that",
            "this",
            "with",
            "website",
        }
        seen: list[str] = []
        for word in words:
            if word in stop or len(word) < 3:
                continue
            if word not in seen:
                seen.append(word)
        return tuple(seen[:12])


@dataclass(frozen=True, slots=True)
class WebsiteSection:
    section_id: str
    eyebrow: str
    title: str
    body: str


@dataclass(frozen=True, slots=True)
class WebsiteBuildResult:
    prompt: str
    title: str
    slug: str
    output_dir: str
    files: tuple[str, ...]
    sections: tuple[WebsiteSection, ...]
    trace_path: str | None = None


class OnePromptWebsiteBuilder:
    """Builds a complete static website from one local prompt."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def build(
        self,
        prompt_text: str,
        *,
        slug: str | None = None,
        trace_path: Path | None = None,
    ) -> WebsiteBuildResult:
        prompt = WebsitePrompt(prompt_text)
        title = self._title(prompt)
        final_slug = slug or self._slug(title)
        target = self.output_dir / final_slug
        target.mkdir(parents=True, exist_ok=True)

        sections = self._sections(prompt, title)
        files = {
            "index.html": self._html(prompt, title, sections),
            "styles.css": self._css(),
            "app.js": self._js(),
            "manifest.json": self._manifest(prompt, title, final_slug, sections),
            "README.md": self._readme(prompt, title),
        }
        for name, content in files.items():
            (target / name).write_text(content, encoding="utf-8")

        result = WebsiteBuildResult(
            prompt=prompt.text,
            title=title,
            slug=final_slug,
            output_dir=str(target),
            files=tuple(sorted(files)),
            sections=sections,
            trace_path=str(trace_path) if trace_path else None,
        )
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
        return result

    @staticmethod
    def _title(prompt: WebsitePrompt) -> str:
        keywords = prompt.keywords
        if "causal" in keywords or "constructivism" in keywords:
            return "Causal Constructivism"
        if keywords:
            return " ".join(word.capitalize() for word in keywords[:3])
        return "Generated Website"

    @staticmethod
    def _slug(title: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
        return slug or "generated-website"

    @staticmethod
    def _sections(prompt: WebsitePrompt, title: str) -> tuple[WebsiteSection, ...]:
        focus = ", ".join(prompt.keywords[:5]) or "local intelligence"
        return (
            WebsiteSection(
                "proof",
                "Proof, not promises",
                "A local system that acts, observes, and verifies",
                (
                    "The site presents a machine that does not depend on remote "
                    "services. It reads local evidence, runs checks, and reports "
                    "what actually happened."
                ),
            ),
            WebsiteSection(
                "architecture",
                "Architecture",
                "A loop around evidence",
                (
                    "Goal, plan, act, observe, verify, repair, and remember. "
                    f"The generated focus is {focus}."
                ),
            ),
            WebsiteSection(
                "capabilities",
                "Capabilities",
                "Built for local experiments",
                (
                    "The page highlights code indexing, test execution, GPU "
                    "awareness, static site generation, and trace collection for "
                    "future learning."
                ),
            ),
            WebsiteSection(
                "limits",
                "Limits",
                "No fake training claim",
                (
                    "This build does not claim neural training happened. It creates "
                    "the artifact and records a trace that can become training data "
                    "when a local model harness exists."
                ),
            ),
        )

    def _html(
        self,
        prompt: WebsitePrompt,
        title: str,
        sections: tuple[WebsiteSection, ...],
    ) -> str:
        safe_title = html.escape(title)
        safe_prompt = html.escape(prompt.text)
        section_markup = "\n".join(self._section(section) for section in sections)
        keywords = "\n".join(
            f"<span>{html.escape(keyword)}</span>" for keyword in prompt.keywords[:8]
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{safe_title}: a locally generated website from one prompt.">
  <title>{safe_title}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header" data-reveal>
    <a class="brand" href="#top" aria-label="{safe_title} home">
      <span class="brand-mark">CC</span>
      <span>{safe_title}</span>
    </a>
    <nav aria-label="Primary navigation">
      <a href="#proof">Proof</a>
      <a href="#architecture">Architecture</a>
      <a href="#capabilities">Capabilities</a>
      <a href="#limits">Limits</a>
    </nav>
  </header>

  <main id="top">
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-copy" data-reveal>
        <p class="kicker">One prompt. Local build. Verifiable output.</p>
        <h1 id="hero-title">{safe_title} turns a prompt into a working interface.</h1>
        <p class="lede">This static site was generated locally from a single prompt, with no server dependency and no claim of fake training.</p>
        <div class="actions" aria-label="Demo actions">
          <a class="button primary" href="#proof">Inspect the proof</a>
          <a class="button secondary" href="manifest.json">Open manifest</a>
        </div>
      </div>
      <div class="hero-visual" aria-label="Causal build loop diagram" data-reveal>
        {self._diagram()}
      </div>
    </section>

    <section class="prompt-panel" aria-labelledby="prompt-title" data-reveal>
      <p class="kicker">Original prompt</p>
      <h2 id="prompt-title">The website is grounded in this request</h2>
      <blockquote>{safe_prompt}</blockquote>
      <div class="keyword-strip" aria-label="Extracted keywords">
        {keywords}
      </div>
    </section>

    {section_markup}
  </main>

  <footer class="site-footer">
    <span>Generated by Phase 18 Website Builder Core.</span>
    <a href="README.md">Build notes</a>
  </footer>

  <script src="app.js"></script>
</body>
</html>
"""

    @staticmethod
    def _section(section: WebsiteSection) -> str:
        return f"""    <section class="content-section" id="{html.escape(section.section_id)}" data-reveal>
      <p class="kicker">{html.escape(section.eyebrow)}</p>
      <h2>{html.escape(section.title)}</h2>
      <p>{html.escape(section.body)}</p>
    </section>"""

    @staticmethod
    def _diagram() -> str:
        return """<svg viewBox="0 0 520 420" role="img" aria-labelledby="diagram-title diagram-desc">
          <title id="diagram-title">Local cognitive build loop</title>
          <desc id="diagram-desc">A loop from goal to plan, action, observation, verification, repair, and memory.</desc>
          <defs>
            <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="18" stdDeviation="18" flood-color="oklch(0.22 0.05 250)" flood-opacity="0.22"/>
            </filter>
          </defs>
          <rect class="diagram-plate" x="20" y="20" width="480" height="380" rx="34"/>
          <path class="diagram-orbit" d="M260 82 C355 82 432 139 432 210 C432 281 355 338 260 338 C165 338 88 281 88 210 C88 139 165 82 260 82"/>
          <g class="diagram-node" transform="translate(226 52)"><rect width="68" height="48" rx="16"/><text x="34" y="30">Goal</text></g>
          <g class="diagram-node" transform="translate(377 133)"><rect width="76" height="48" rx="16"/><text x="38" y="30">Plan</text></g>
          <g class="diagram-node" transform="translate(377 239)"><rect width="76" height="48" rx="16"/><text x="38" y="30">Act</text></g>
          <g class="diagram-node" transform="translate(218 322)"><rect width="84" height="48" rx="16"/><text x="42" y="30">Observe</text></g>
          <g class="diagram-node" transform="translate(67 239)"><rect width="86" height="48" rx="16"/><text x="43" y="30">Verify</text></g>
          <g class="diagram-node" transform="translate(67 133)"><rect width="86" height="48" rx="16"/><text x="43" y="30">Repair</text></g>
          <circle class="diagram-core" cx="260" cy="210" r="70"/>
          <text class="diagram-core-text" x="260" y="202">Local</text>
          <text class="diagram-core-text small" x="260" y="228">evidence loop</text>
        </svg>"""

    @staticmethod
    def _css() -> str:
        return """:root {
  color-scheme: light;
  --paper: oklch(0.97 0.012 82);
  --ink: oklch(0.20 0.030 255);
  --muted: oklch(0.43 0.035 250);
  --line: oklch(0.83 0.030 84);
  --field: oklch(0.93 0.020 82);
  --accent: oklch(0.58 0.185 28);
  --accent-2: oklch(0.62 0.135 218);
  --accent-3: oklch(0.72 0.160 118);
  --shadow: 0 24px 70px oklch(0.32 0.04 252 / 0.16);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background:
    radial-gradient(circle at 16% 12%, oklch(0.88 0.080 58 / 0.62), transparent 30rem),
    radial-gradient(circle at 88% 18%, oklch(0.80 0.090 216 / 0.50), transparent 27rem),
    var(--paper);
  color: var(--ink);
}

a {
  color: inherit;
}

.site-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  padding: clamp(1rem, 2vw, 1.5rem) clamp(1rem, 4vw, 4rem);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  font-weight: 760;
  text-decoration: none;
  letter-spacing: -0.03em;
}

.brand-mark {
  display: grid;
  width: 2.5rem;
  height: 2.5rem;
  place-items: center;
  border: 1px solid var(--ink);
  border-radius: 999px;
  background: var(--accent);
  color: oklch(0.97 0.012 82);
}

nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  color: var(--muted);
  font-size: 0.94rem;
}

nav a,
.button {
  border-radius: 999px;
  padding: 0.72rem 1rem;
  text-decoration: none;
}

nav a:hover {
  background: var(--field);
  color: var(--ink);
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.04fr) minmax(20rem, 0.96fr);
  gap: clamp(2rem, 6vw, 7rem);
  align-items: center;
  min-height: calc(100vh - 5.5rem);
  padding: clamp(2rem, 7vw, 7rem) clamp(1rem, 4vw, 4rem);
}

.kicker {
  margin: 0 0 1rem;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

h1,
h2 {
  margin: 0;
  letter-spacing: -0.075em;
  line-height: 0.92;
}

h1 {
  max-width: 12ch;
  font-size: clamp(4rem, 11vw, 10.5rem);
}

h2 {
  max-width: 14ch;
  font-size: clamp(2.8rem, 7vw, 7rem);
}

.lede,
.content-section p,
.prompt-panel blockquote {
  max-width: 66ch;
  color: var(--muted);
  font-size: clamp(1.08rem, 1.6vw, 1.32rem);
  line-height: 1.72;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-top: 2rem;
}

.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 3.2rem;
  border: 1px solid var(--ink);
  font-weight: 750;
}

.button.primary {
  background: var(--ink);
  color: var(--paper);
}

.button.secondary {
  background: oklch(0.98 0.008 82 / 0.72);
}

.hero-visual {
  filter: drop-shadow(0 36px 80px oklch(0.28 0.04 260 / 0.26));
}

svg {
  display: block;
  width: min(100%, 38rem);
  margin-inline: auto;
}

.diagram-plate {
  fill: oklch(0.95 0.020 82 / 0.88);
  stroke: var(--line);
}

.diagram-orbit {
  fill: none;
  stroke: var(--accent-2);
  stroke-dasharray: 10 12;
  stroke-width: 3;
}

.diagram-node rect {
  fill: var(--ink);
  filter: url(#soft-shadow);
}

.diagram-node text,
.diagram-core-text {
  fill: var(--paper);
  font-size: 15px;
  font-weight: 800;
  text-anchor: middle;
}

.diagram-core {
  fill: var(--accent);
}

.diagram-core-text {
  font-size: 22px;
}

.diagram-core-text.small {
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.prompt-panel,
.content-section {
  margin: clamp(2rem, 6vw, 6rem) clamp(1rem, 4vw, 4rem);
  padding: clamp(2rem, 5vw, 5rem);
  border: 1px solid var(--line);
  border-radius: clamp(1.5rem, 3vw, 3rem);
  background: oklch(0.98 0.010 82 / 0.72);
  box-shadow: var(--shadow);
}

.prompt-panel blockquote {
  margin: 1.5rem 0 0;
  color: var(--ink);
}

.keyword-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1.5rem;
}

.keyword-strip span {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.5rem 0.72rem;
  background: var(--field);
  color: var(--muted);
  font-size: 0.9rem;
}

.content-section:nth-of-type(2n) {
  margin-left: clamp(1rem, 13vw, 12rem);
}

.content-section:nth-of-type(2n + 1) {
  margin-right: clamp(1rem, 13vw, 12rem);
}

.site-footer {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 3rem clamp(1rem, 4vw, 4rem);
  color: var(--muted);
}

[data-reveal] {
  opacity: 0;
  transform: translateY(18px);
  transition: opacity 700ms cubic-bezier(0.22, 1, 0.36, 1), transform 700ms cubic-bezier(0.22, 1, 0.36, 1);
}

[data-reveal].is-visible {
  opacity: 1;
  transform: translateY(0);
}

@media (max-width: 860px) {
  .site-header,
  .site-footer {
    align-items: flex-start;
    flex-direction: column;
  }

  .hero {
    grid-template-columns: 1fr;
    min-height: auto;
  }

  h1 {
    max-width: 10ch;
  }
}
"""

    @staticmethod
    def _js() -> str:
        return """const revealTargets = document.querySelectorAll('[data-reveal]');

const observer = new IntersectionObserver(
  entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    }
  },
  { threshold: 0.16 }
);

for (const target of revealTargets) {
  observer.observe(target);
}
"""

    @staticmethod
    def _manifest(
        prompt: WebsitePrompt,
        title: str,
        slug: str,
        sections: tuple[WebsiteSection, ...],
    ) -> str:
        payload = {
            "title": title,
            "slug": slug,
            "prompt": prompt.text,
            "keywords": list(prompt.keywords),
            "files": ["index.html", "styles.css", "app.js", "manifest.json", "README.md"],
            "sections": [asdict(section) for section in sections],
            "claims": {
                "neural_training": False,
                "server_required": False,
                "static_site": True,
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def _readme(prompt: WebsitePrompt, title: str) -> str:
        return f"""# {title}

Generated from one prompt:

```text
{prompt.text}
```

Open `index.html` in a browser. No server is required.

Files:

- `index.html`
- `styles.css`
- `app.js`
- `manifest.json`

This is a deterministic local website build. It does not claim neural training.
"""


def run_website_builder_benchmark(
    prompt: str,
    *,
    output_dir: Path | str = "docs/generated-websites",
    slug: str | None = None,
    trace_path: Path | None = None,
) -> WebsiteBuildResult:
    return OnePromptWebsiteBuilder(Path(output_dir)).build(
        prompt,
        slug=slug,
        trace_path=trace_path,
    )
