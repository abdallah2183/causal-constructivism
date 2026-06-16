from __future__ import annotations

import argparse
from pathlib import Path

from .code_language_training import make_model, require_torch


def sample_from_checkpoint(args: argparse.Namespace) -> str:
    torch, nn, _ = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = make_model(
        nn,
        sequence_length=int(checkpoint["sequence_length"]),
        width=int(checkpoint["width"]),
        layers=int(checkpoint["layers"]),
        heads=int(checkpoint["heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sequence_length = int(checkpoint["sequence_length"])
    generated = bytearray(args.prompt.encode("utf-8", errors="ignore"))
    causal_mask = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device),
        diagonal=1,
    )

    with torch.no_grad():
        for _ in range(args.max_bytes):
            context = bytes(generated[-sequence_length:])
            padded = bytes(max(0, sequence_length - len(context))) + context
            tokens = torch.tensor(list(padded), dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(tokens, causal_mask)[0, -1] / max(args.temperature, 0.05)
            if args.top_k > 0:
                values, indices = torch.topk(logits, k=min(args.top_k, logits.numel()))
                probabilities = torch.softmax(values, dim=-1)
                next_byte = int(indices[torch.multinomial(probabilities, 1)].item())
            else:
                probabilities = torch.softmax(logits, dim=-1)
                next_byte = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_byte)

    return generated.decode("utf-8", errors="ignore")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample text from a local code language checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/code-language-long/latest.pt")
    parser.add_argument("--prompt", default="<file path=\"index.html\" language=\"html\" project=\"generated\">\n")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = sample_from_checkpoint(args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
