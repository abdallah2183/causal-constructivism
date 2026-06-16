from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for code language training. Install a CUDA-enabled torch build first."
        ) from exc
    return torch, nn, functional


@dataclass(frozen=True, slots=True)
class CodeTrainingCorpus:
    data: bytes
    examples: int
    bytes_count: int
    languages: dict[str, int]
    projects: dict[str, int]


def load_code_corpus(path: Path, max_bytes: int) -> CodeTrainingCorpus:
    chunks: list[bytes] = []
    examples = 0
    total = 0
    languages: dict[str, int] = {}
    projects: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            code = str(payload.get("code", "")).strip()
            if not code:
                continue
            language = str(payload.get("language", "unknown"))
            project = str(payload.get("project", "unknown"))
            source_path = str(payload.get("source_path", "unknown"))
            header = f"\n<file path=\"{source_path}\" language=\"{language}\" project=\"{project}\">\n"
            footer = "\n</file>\n"
            encoded = (header + code + footer).encode("utf-8", errors="ignore")
            if total + len(encoded) > max_bytes:
                remaining = max_bytes - total
                if remaining <= 0:
                    break
                encoded = encoded[:remaining]
            chunks.append(encoded)
            total += len(encoded)
            examples += 1
            languages[language] = languages.get(language, 0) + 1
            projects[project] = projects.get(project, 0) + 1
            if total >= max_bytes:
                break
    data = b"".join(chunks)
    if len(data) < 1024:
        raise ValueError(f"Corpus at {path} is too small for language training.")
    return CodeTrainingCorpus(
        data=data,
        examples=examples,
        bytes_count=len(data),
        languages=dict(sorted(languages.items())),
        projects=dict(sorted(projects.items(), key=lambda item: (-item[1], item[0]))[:100]),
    )


def make_model(nn, sequence_length: int, width: int, layers: int, heads: int):
    class ByteCodeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence_length = sequence_length
            self.byte_embedding = nn.Embedding(256, width)
            self.position_embedding = nn.Embedding(sequence_length, width)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=width * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, 256)

        def forward(self, tokens, causal_mask):
            positions = tokens.new_tensor(tuple(range(tokens.shape[1])))
            hidden = self.byte_embedding(tokens) + self.position_embedding(positions)
            encoded = self.encoder(hidden, mask=causal_mask)
            return self.head(self.norm(encoded))

    return ByteCodeTransformer()


def save_checkpoint(
    torch,
    output_dir: Path,
    model,
    step: int,
    loss: float,
    elapsed: float,
    device_name: str,
    corpus: CodeTrainingCorpus,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "step": step,
            "loss": loss,
            "elapsed_seconds": elapsed,
            "device": device_name,
            "sequence_length": args.sequence_length,
            "width": args.width,
            "layers": args.layers,
            "heads": args.heads,
            "vocab": "byte-0-255",
        },
        checkpoint,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "corpus_bytes": corpus.bytes_count,
        "device": device_name,
        "elapsed_seconds": elapsed,
        "examples": corpus.examples,
        "languages": corpus.languages,
        "loss": loss,
        "projects": corpus.projects,
        "sequence_length": args.sequence_length,
        "step": step,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint


def train(args: argparse.Namespace) -> None:
    torch, nn, functional = require_torch()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    corpus = load_code_corpus(Path(args.corpus), args.max_training_bytes)
    if corpus.bytes_count <= args.sequence_length + 1:
        raise ValueError("Corpus must be larger than sequence length.")

    data = torch.tensor(list(corpus.data), dtype=torch.uint8, device=device)
    model = make_model(
        nn,
        sequence_length=args.sequence_length,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    offsets = torch.arange(args.sequence_length + 1, device=device)
    causal_mask = torch.triu(
        torch.ones((args.sequence_length, args.sequence_length), dtype=torch.bool, device=device),
        diagonal=1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    start = time.time()
    last_checkpoint = start
    step = 0
    last_loss = math.nan
    sample_limit = corpus.bytes_count - args.sequence_length - 1

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps({
                "event": "start",
                "corpus": str(args.corpus),
                "corpus_bytes": corpus.bytes_count,
                "device": device_name,
                "examples": corpus.examples,
                "width": args.width,
                "layers": args.layers,
                "heads": args.heads,
                "sequence_length": args.sequence_length,
                "duration_seconds": args.duration_seconds,
            })
            + "\n"
        )
        while True:
            step += 1
            starts = torch.randint(0, sample_limit, (args.batch_size,), device=device)
            windows = data[(starts[:, None] + offsets[None, :])].long()
            x = windows[:, :-1]
            y = windows[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda" and args.amp,
                dtype=torch.float16,
            ):
                logits = model(x, causal_mask)
                loss = functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            last_loss = float(loss.detach().cpu())

            now = time.time()
            elapsed = now - start
            if step % args.log_every == 0:
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
                else:
                    allocated = reserved = 0
                log.write(
                    json.dumps({
                        "event": "step",
                        "elapsed_seconds": elapsed,
                        "loss": last_loss,
                        "memory_allocated_mib": allocated,
                        "memory_reserved_mib": reserved,
                        "step": step,
                    })
                    + "\n"
                )
                log.flush()

            if now - last_checkpoint >= args.checkpoint_every_seconds:
                save_checkpoint(torch, output_dir, model, step, last_loss, elapsed, device_name, corpus, args)
                last_checkpoint = now

            if args.max_steps and step >= args.max_steps:
                break
            if args.duration_seconds and elapsed >= args.duration_seconds:
                break

        checkpoint = save_checkpoint(
            torch,
            output_dir,
            model,
            step,
            last_loss,
            time.time() - start,
            device_name,
            corpus,
            args,
        )
        log.write(
            json.dumps({
                "event": "complete",
                "checkpoint": str(checkpoint),
                "loss": last_loss,
                "step": step,
            })
            + "\n"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a local byte-level code language model.")
    parser.add_argument("--corpus", default="artifacts/code-corpus/local-code-corpus.jsonl")
    parser.add_argument("--output-dir", default="artifacts/code-language-model")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-training-bytes", type=int, default=512_000_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-seconds", type=int, default=300)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
