from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for GPU training. Install a CUDA-enabled torch build first."
        ) from exc
    return torch, nn


@dataclass(frozen=True, slots=True)
class TraceExample:
    prompt: str
    labels: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    import re

    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())
        if len(token) > 1
    ]


def load_trace_examples(path: Path) -> list[TraceExample]:
    examples: list[TraceExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        labels = tuple(str(label) for label in payload.get("labels", []))
        prompt = str(payload.get("prompt", "")).strip()
        if prompt and labels:
            examples.append(TraceExample(prompt=prompt, labels=labels))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def augment_examples(examples: list[TraceExample], copies: int) -> list[TraceExample]:
    prefixes = (
        "please",
        "now",
        "locally",
        "verified",
        "carefully",
        "from one prompt",
        "without cloud",
        "using evidence",
    )
    suffixes = (
        "and verify it",
        "with tests",
        "with a manifest",
        "using local traces",
        "with memory",
        "for the current project",
        "and save the artifact",
    )
    augmented = list(examples)
    rng = random.Random(17)
    for example in examples:
        words = example.prompt.split()
        for _ in range(copies):
            mutated = list(words)
            if len(mutated) > 4 and rng.random() < 0.35:
                rng.shuffle(mutated)
            if rng.random() < 0.65:
                mutated.insert(0, rng.choice(prefixes))
            if rng.random() < 0.65:
                mutated.append(rng.choice(suffixes))
            augmented.append(TraceExample(
                prompt=" ".join(mutated),
                labels=example.labels,
            ))
    return augmented


def build_vocab(examples: list[TraceExample]) -> tuple[dict[str, int], dict[str, int]]:
    tokens = sorted({token for example in examples for token in tokenize(example.prompt)})
    labels = sorted({label for example in examples for label in example.labels})
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({token: index + 2 for index, token in enumerate(tokens)})
    label_vocab = {label: index for index, label in enumerate(labels)}
    return vocab, label_vocab


def encode_batch(
    torch,
    batch: list[TraceExample],
    vocab: dict[str, int],
    label_vocab: dict[str, int],
    sequence_length: int,
    device,
):
    x = torch.zeros((len(batch), sequence_length), dtype=torch.long, device=device)
    y = torch.zeros((len(batch), len(label_vocab)), dtype=torch.float32, device=device)
    for row, example in enumerate(batch):
        ids = [vocab.get(token, 1) for token in tokenize(example.prompt)[:sequence_length]]
        if ids:
            x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        for label in example.labels:
            y[row, label_vocab[label]] = 1.0
    return x, y


def encode_dataset(
    torch,
    examples: list[TraceExample],
    vocab: dict[str, int],
    label_vocab: dict[str, int],
    sequence_length: int,
    device,
):
    return encode_batch(torch, examples, vocab, label_vocab, sequence_length, device)


def make_model(nn, vocab_size: int, label_count: int, width: int, layers: int, heads: int):
    class TraceTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
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
            self.head = nn.Linear(width, label_count)

        def forward(self, tokens):
            mask = tokens == 0
            embedded = self.embedding(tokens)
            encoded = self.encoder(embedded, src_key_padding_mask=mask)
            valid = (~mask).unsqueeze(-1).float()
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            return self.head(self.norm(pooled))

    return TraceTransformer()


def save_checkpoint(
    torch,
    output_dir: Path,
    model,
    vocab: dict[str, int],
    label_vocab: dict[str, int],
    step: int,
    loss: float,
    elapsed: float,
    device_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "label_vocab": label_vocab,
            "step": step,
            "loss": loss,
            "elapsed_seconds": elapsed,
            "device": device_name,
        },
        checkpoint,
    )
    summary = output_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "device": device_name,
                "elapsed_seconds": elapsed,
                "labels": len(label_vocab),
                "loss": loss,
                "step": step,
                "vocab": len(vocab),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint


def train(args: argparse.Namespace) -> None:
    torch, nn = require_torch()
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    raw_examples = load_trace_examples(Path(args.dataset))
    examples = augment_examples(raw_examples, args.augment_copies)
    vocab, label_vocab = build_vocab(examples)
    model = make_model(
        nn,
        vocab_size=len(vocab),
        label_count=len(label_vocab),
        width=args.width,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    encoded_x, encoded_y = encode_dataset(
        torch,
        examples,
        vocab,
        label_vocab,
        args.sequence_length,
        device,
    )

    output_dir = Path(args.output_dir)
    log_path = output_dir / "training.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    last_checkpoint = start
    step = 0
    last_loss = math.nan

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps({
                "event": "start",
                "device": device_name,
                "examples": len(examples),
                "labels": len(label_vocab),
                "vocab": len(vocab),
                "width": args.width,
                "layers": args.layers,
                "heads": args.heads,
                "duration_seconds": args.duration_seconds,
            })
            + "\n"
        )
        while True:
            step += 1
            batch_indices = torch.randint(
                0,
                encoded_x.shape[0],
                (args.batch_size,),
                device=device,
            )
            x = encoded_x.index_select(0, batch_indices)
            y = encoded_y.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda" and args.amp,
                dtype=torch.float16,
            ):
                logits = model(x)
                loss = criterion(logits, y)
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
                        "step": step,
                        "loss": last_loss,
                        "elapsed_seconds": elapsed,
                        "memory_allocated_mib": allocated,
                        "memory_reserved_mib": reserved,
                    })
                    + "\n"
                )
                log.flush()

            if now - last_checkpoint >= args.checkpoint_every_seconds:
                save_checkpoint(
                    torch,
                    output_dir,
                    model,
                    vocab,
                    label_vocab,
                    step,
                    last_loss,
                    elapsed,
                    device_name,
                )
                last_checkpoint = now

            if args.max_steps and step >= args.max_steps:
                break
            if args.duration_seconds and elapsed >= args.duration_seconds:
                break

        checkpoint = save_checkpoint(
            torch,
            output_dir,
            model,
            vocab,
            label_vocab,
            step,
            last_loss,
            time.time() - start,
            device_name,
        )
        log.write(
            json.dumps({
                "event": "complete",
                "step": step,
                "loss": last_loss,
                "checkpoint": str(checkpoint),
            })
            + "\n"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the local GPU trace model.")
    parser.add_argument("--dataset", default="docs/training-data/local-trace-dataset.jsonl")
    parser.add_argument("--output-dir", default="models/gpu-trace-model")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--augment-copies", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=25)
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
