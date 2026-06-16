param(
    [string]$InterventionVariable = "action.force",
    [double]$InterventionValue = 2.0,
    [string]$QueryVariable = "green.position",
    [int]$Particles = 200,
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
    "--twin",
    "--intervention-variable", $InterventionVariable,
    "--intervention-value", $InterventionValue,
    "--query-variable", $QueryVariable,
    "--particles", $Particles
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE

