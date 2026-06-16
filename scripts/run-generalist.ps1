param(
    [double]$SourceFriction = 0.30,
    [double]$TargetFriction = 0.05,
    [string]$ConceptLibrary = "artifacts/concept-library.json",
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
    "--generalist",
    "--source-friction", $SourceFriction,
    "--target-friction", $TargetFriction,
    "--concept-library", $ConceptLibrary
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
