# GPU Training

Phase 20 trains a local CUDA PyTorch model over verified trace data.

## Short Run

```powershell
.\scripts\run-gpu-training.ps1 -DurationSeconds 300
```

## Long Run

```powershell
.\scripts\run-gpu-training.ps1 `
    -OutputDir artifacts\gpu-training-long `
    -DurationSeconds 21600 `
    -BatchSize 768 `
    -Width 768 `
    -Layers 6 `
    -Heads 12 `
    -AugmentCopies 4096
```

## Monitor

```powershell
nvidia-smi
Get-Content artifacts\gpu-training-long\training.log -Tail 20
Get-Content artifacts\gpu-training-long\summary.json
```

## Stop A Background Run

Find the Python process:

```powershell
Get-Process python,py
```

Stop it by process id:

```powershell
Stop-Process -Id <PID>
```

## Scope

This is real GPU neural training, but it is not foundation-model training. The
current dataset is small, so the training result is useful as a local mechanism
and checkpoint format, not as evidence of broad coding intelligence.
