param(
    [string[]]$Trace = @(),
    [string]$Model = "models/local-trace-model.json",
    [string]$Dataset = "docs/training-data/local-trace-dataset.jsonl",
    [string]$EvalPrompt = "Build a website and update graph tests.",
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$workspacePython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$python = if (Test-Path -LiteralPath $workspacePython) {
    $workspacePython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
$arguments = @(
    "-m", "causal_constructivism",
    "--train-local",
    "--training-model", $Model,
    "--training-dataset", $Dataset,
    "--training-eval-prompt", $EvalPrompt
)
foreach ($tracePath in $Trace) {
    $arguments += @("--training-trace", $tracePath)
}
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
