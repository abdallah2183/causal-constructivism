# Local GPU Learning Path

The machine currently exposes an NVIDIA GeForce RTX 5060 Ti with about 16GB of
VRAM. That is useful for local coding-model inference and small fine-tuning
experiments, but it is not enough to train a frontier model from scratch.

## What Exists Now

Phase 17 adds a real local Programmer Core:

- AST-based Python project indexing.
- Source/test/module/symbol mapping.
- Programming-task file targeting.
- Local compile/test command planning.
- Verification command execution.
- Failure extraction from command output.
- NVIDIA GPU detection through `nvidia-smi`.
- JSONL memory recording for programming traces.

This is the foundation for learning from programming work. It is not yet a
neural training loop.

## What Must Exist Before GPU Training

Real local training needs:

1. A local code model compatible with the machine.
2. A dataset of programming traces:
   - task text,
   - indexed files,
   - selected target files,
   - attempted patch,
   - test result,
   - repair result,
   - final verified state.
3. A measurable benchmark:
   - pass rate,
   - number of repair attempts,
   - patch size,
   - unrelated-file changes,
   - regression count.
4. A training harness:
   - LoRA or QLoRA fine-tuning,
   - checkpoint output,
   - evaluation before/after training.

## First Learning Target

Do not train on random internet code first. Train on verified local traces.

The first dataset should come from Phase 17 memory:

```powershell
.\scripts\run-programmer.ps1 `
    -Task "improve graph error handling" `
    -Memory artifacts\programmer-memory.jsonl `
    -Json
```

When patch generation is added, each successful fix should append one trace to
that memory file.

## Honest Milestone

The first serious learning milestone is not "be smarter than LLMs." It is:

> On a suite of intentionally broken Python mini-projects, the system fixes more
> tasks after learning from its previous verified repair traces.

That is measurable. That is real.
