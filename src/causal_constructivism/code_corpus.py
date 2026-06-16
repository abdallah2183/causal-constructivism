from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CODE_EXTENSIONS = {
    ".astro": "astro",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript-react",
    ".json": "json",
    ".kt": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".php": "php",
    ".ps1": "powershell",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scss": "scss",
    ".sh": "shell",
    ".sql": "sql",
    ".svelte": "svelte",
    ".swift": "swift",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".vue": "vue",
    ".yaml": "yaml",
    ".yml": "yaml",
}

EXCLUDED_DIRS = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "bin",
    "build",
    "coverage",
    "debug",
    "dist",
    "env",
    "node_modules",
    "obj",
    "out",
    "release",
    "target",
    "vendor",
    "venv",
    "__pycache__",
}

SECRET_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
}

SECRET_SUFFIXES = {
    ".cer",
    ".crt",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
}

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b[A-Za-z0-9_]*api[_-]?key[A-Za-z0-9_]*\s*[:=]\s*['\"][^'\"]{12,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_]*secret[A-Za-z0-9_]*\s*[:=]\s*['\"][^'\"]{12,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_]*token[A-Za-z0-9_]*\s*[:=]\s*['\"][^'\"]{16,}", re.IGNORECASE),
    re.compile(r"\bpassword\s*[:=]\s*['\"][^'\"]{8,}", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class CodeCorpusRecord:
    source_path: str
    project: str
    language: str
    size_bytes: int
    sha256: str
    prompt: str
    labels: tuple[str, ...]
    code: str


@dataclass(frozen=True, slots=True)
class CodeCorpusReport:
    roots: tuple[str, ...]
    output_path: str
    scanned_files: int
    written_examples: int
    skipped_files: int
    skipped_secret_like: int
    skipped_large: int
    total_code_bytes: int
    languages: dict[str, int]
    projects: dict[str, int]


def default_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    cwd = Path.cwd()
    roots.append(cwd)
    if cwd.parent.exists():
        roots.append(cwd.parent)

    home = Path.home()
    for relative in (
        "Desktop",
        "Documents",
        "source",
        "repos",
        "OneDrive/Desktop",
        "OneDrive/Documents",
    ):
        path = home / relative
        if path.exists():
            roots.append(path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen:
            deduped.append(resolved)
            seen.add(key)
    return tuple(deduped)


def is_excluded_dir(path: Path) -> bool:
    return any(part.lower() in EXCLUDED_DIRS for part in path.parts)


def is_secret_like_path(path: Path) -> bool:
    name = path.name.lower()
    if name in SECRET_FILE_NAMES:
        return True
    if path.suffix.lower() in SECRET_SUFFIXES:
        return True
    return any(fragment in name for fragment in ("secret", "credential", "private-key"))


def looks_secret_like(text: str) -> bool:
    sample = text[:200_000]
    return any(pattern.search(sample) for pattern in SECRET_PATTERNS)


def language_for(path: Path) -> str | None:
    return CODE_EXTENSIONS.get(path.suffix.lower())


def project_name(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return root.name
    return relative.parts[0] if len(relative.parts) > 1 else root.name


def classify_labels(path: Path, root: Path, language: str, project: str) -> tuple[str, ...]:
    labels = {
        f"language:{language}",
        f"extension:{path.suffix.lower()}",
        f"project:{project}",
        f"file:{path.name}",
    }
    lower_parts = {part.lower() for part in path.parts}
    if any(part in lower_parts for part in ("test", "tests", "__tests__", "spec")):
        labels.add("kind:test")
    if path.name.lower() in {"package.json", "pyproject.toml", "cargo.toml", "pom.xml"}:
        labels.add("kind:manifest")
    if language in {"html", "css", "scss", "javascript-react", "typescript-react", "vue", "svelte"}:
        labels.add("kind:frontend")
    if language in {"python", "go", "rust", "java", "csharp", "php", "ruby"}:
        labels.add("kind:backend")
    if language in {"markdown"}:
        labels.add("kind:documentation")
    try:
        parent = path.parent.relative_to(root).as_posix()
    except ValueError:
        parent = path.parent.name
    if parent and parent != ".":
        labels.add(f"directory:{parent}")
    return tuple(sorted(labels))


def iter_code_files(roots: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if is_excluded_dir(path):
                continue
            if language_for(path) is None:
                continue
            try:
                key = os.path.normcase(str(path.resolve()))
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            yield root, path


def build_code_corpus(
    roots: Iterable[Path],
    output_path: Path,
    *,
    max_files: int = 50_000,
    max_file_bytes: int = 512_000,
    max_total_bytes: int = 1_000_000_000,
    min_file_bytes: int = 40,
) -> CodeCorpusReport:
    resolved_roots = tuple(root.resolve() for root in roots if root.exists())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    written = 0
    skipped = 0
    skipped_secret_like = 0
    skipped_large = 0
    total_bytes = 0
    languages: dict[str, int] = {}
    projects: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for root, path in iter_code_files(resolved_roots):
            if written >= max_files or total_bytes >= max_total_bytes:
                break
            scanned += 1
            if is_secret_like_path(path):
                skipped += 1
                skipped_secret_like += 1
                continue
            try:
                size = path.stat().st_size
            except OSError:
                skipped += 1
                continue
            if size < min_file_bytes:
                skipped += 1
                continue
            if size > max_file_bytes:
                skipped += 1
                skipped_large += 1
                continue
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                skipped += 1
                continue
            if looks_secret_like(text):
                skipped += 1
                skipped_secret_like += 1
                continue
            stripped = text.strip()
            if len(stripped) < min_file_bytes:
                skipped += 1
                continue

            language = language_for(path)
            if language is None:
                skipped += 1
                continue
            project = project_name(root, path)
            try:
                relative_path = path.relative_to(root).as_posix()
            except ValueError:
                relative_path = path.as_posix()
            digest = hashlib.sha256(stripped.encode("utf-8", errors="ignore")).hexdigest()
            labels = classify_labels(path, root, language, project)
            prompt = f"Rebuild {relative_path} for project {project} in {language}."
            record = CodeCorpusRecord(
                source_path=relative_path,
                project=project,
                language=language,
                size_bytes=size,
                sha256=digest,
                prompt=prompt,
                labels=labels,
                code=stripped,
            )
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            total_bytes += len(stripped.encode("utf-8", errors="ignore"))
            languages[language] = languages.get(language, 0) + 1
            projects[project] = projects.get(project, 0) + 1

    return CodeCorpusReport(
        roots=tuple(str(root) for root in resolved_roots),
        output_path=str(output_path),
        scanned_files=scanned,
        written_examples=written,
        skipped_files=skipped,
        skipped_secret_like=skipped_secret_like,
        skipped_large=skipped_large,
        total_code_bytes=total_bytes,
        languages=dict(sorted(languages.items())),
        projects=dict(sorted(projects.items(), key=lambda item: (-item[1], item[0]))[:100]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local code corpus for GPU training.")
    parser.add_argument("--root", action="append", dest="roots", default=[])
    parser.add_argument("--output", default="artifacts/code-corpus/local-code-corpus.jsonl")
    parser.add_argument("--max-files", type=int, default=50_000)
    parser.add_argument("--max-file-bytes", type=int, default=512_000)
    parser.add_argument("--max-total-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--min-file-bytes", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(Path(root) for root in args.roots) if args.roots else default_roots()
    report = build_code_corpus(
        roots,
        Path(args.output),
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        max_total_bytes=args.max_total_bytes,
        min_file_bytes=args.min_file_bytes,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
