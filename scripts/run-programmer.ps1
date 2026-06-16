param(
    [string]$Task = "Add a verified programming capability to this Python project.",
    [switch]$RunTests,
    [string]$Memory = "",
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
    "--programmer",
    "--programmer-task", $Task
)
if ($RunTests) {
    $arguments += "--programmer-run-tests"
}
if ($Memory) {
    $arguments += @("--programmer-memory", $Memory)
}
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
