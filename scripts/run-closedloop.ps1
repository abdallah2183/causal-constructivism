param(
    [int]$Experiments = 6,
    [string]$Database = "artifacts/closed-loop.db",
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
    "--closed-loop",
    "--experiments", $Experiments,
    "--database", $Database
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
