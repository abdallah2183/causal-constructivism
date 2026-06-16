param(
    [int]$LawObservations = 20,
    [double]$Gravity = 9.81,
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
    "--theorist",
    "--law-observations", $LawObservations,
    "--gravity", $Gravity
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
