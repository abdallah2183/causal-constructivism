# Code Language Training

Phase 21 builds a local code corpus from project files and trains a byte-level
CUDA Transformer over that corpus.

## Build The Corpus

```powershell
.\scripts\build-code-corpus.ps1 `
    -Root C:\Users\abdal\OneDrive\Desktop `
    -Root C:\Users\abdal\Documents `
    -Output artifacts\code-corpus\local-code-corpus.jsonl
```

The corpus builder scans code-like files, skips common dependency/build folders,
and excludes files that look like secrets or credentials.

## Train

```powershell
.\scripts\run-code-language-training.ps1 `
    -Corpus artifacts\code-corpus\local-code-corpus.jsonl `
    -OutputDir artifacts\code-language-long `
    -DurationSeconds 86400 `
    -BatchSize 96 `
    -SequenceLength 384 `
    -Width 768 `
    -Layers 6 `
    -Heads 12
```

This trains a next-byte language model over local code. It samples random code
windows continuously, so a long run can process millions of training windows
even when the source corpus has thousands of files.

## Monitor

```powershell
nvidia-smi
Get-Content artifacts\code-language-long\training.log -Tail 20
Get-Content artifacts\code-language-long\summary.json
```

## Sample

After a checkpoint exists:

```powershell
.\scripts\sample-code-language.ps1 `
    -Checkpoint artifacts\code-language-long\latest.pt `
    -Prompt "<file path=`"index.html`" language=`"html`" project=`"generated`">`n" `
    -Output artifacts\generated-code\sample-index.html `
    -MaxBytes 4096
```

## Scope

This is real local neural training on local code. It is not the same as training
a frontier coding model from scratch. The output quality depends on corpus size,
training duration, model size, and evaluation quality.
