param(
    [string]$Dataset = "docs/training-data/local-trace-dataset.jsonl",
    [string]$OutputDir = "models/gpu-trace-model",
    [int]$DurationSeconds = 3600,
    [int]$MaxSteps = 0,
    [int]$BatchSize = 512,
    [int]$Width = 512,
    [int]$Layers = 4,
    [int]$Heads = 8,
    [int]$AugmentCopies = 512
)

$ErrorActionPreference = "Stop"
$py = (Get-Command py -ErrorAction Stop).Source

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
$arguments = @(
    "-m", "causal_constructivism.gpu_training",
    "--dataset", $Dataset,
    "--output-dir", $OutputDir,
    "--duration-seconds", $DurationSeconds,
    "--batch-size", $BatchSize,
    "--width", $Width,
    "--layers", $Layers,
    "--heads", $Heads,
    "--augment-copies", $AugmentCopies,
    "--require-cuda"
)
if ($MaxSteps -gt 0) {
    $arguments += @("--max-steps", $MaxSteps)
}

& $py @arguments
exit $LASTEXITCODE
