param(
    [string]$Corpus = "artifacts/code-corpus/local-code-corpus.jsonl",
    [string]$OutputDir = "artifacts/code-language-model",
    [int]$DurationSeconds = 3600,
    [int]$MaxSteps = 0,
    [int]$BatchSize = 64,
    [int]$SequenceLength = 256,
    [int]$Width = 512,
    [int]$Layers = 6,
    [int]$Heads = 8,
    [int]$MaxTrainingBytes = 512000000
)

$ErrorActionPreference = "Stop"
$py = (Get-Command py -ErrorAction Stop).Source

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
$arguments = @(
    "-m", "causal_constructivism.code_language_training",
    "--corpus", $Corpus,
    "--output-dir", $OutputDir,
    "--duration-seconds", $DurationSeconds,
    "--batch-size", $BatchSize,
    "--sequence-length", $SequenceLength,
    "--width", $Width,
    "--layers", $Layers,
    "--heads", $Heads,
    "--max-training-bytes", $MaxTrainingBytes,
    "--require-cuda"
)
if ($MaxSteps -gt 0) {
    $arguments += @("--max-steps", $MaxSteps)
}

& $py @arguments
exit $LASTEXITCODE
