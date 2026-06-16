param(
    [int]$HistoryLength = 6,
    [double]$TrueMass = 2.5,
    [double]$HiddenFriction = 0.25,
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
    "--historian",
    "--history-length", $HistoryLength,
    "--true-mass", $TrueMass,
    "--hidden-friction", $HiddenFriction
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
