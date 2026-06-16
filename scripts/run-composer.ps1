param(
    [double]$Friction = 0.25,
    [double]$Restitution = 0.65,
    [int]$CompoundCount = 8,
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
    "--composer",
    "--composer-friction", $Friction,
    "--composer-restitution", $Restitution,
    "--compound-count", $CompoundCount
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
