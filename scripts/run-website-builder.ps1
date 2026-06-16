param(
    [string]$Prompt = "Build a complete landing page for a local cognitive engine that programs, verifies, remembers, and runs on my RTX GPU.",
    [string]$Output = "docs/generated-websites",
    [string]$Slug = "",
    [string]$Trace = "",
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
    "--website-builder",
    "--website-prompt", $Prompt,
    "--website-output", $Output
)
if ($Slug) {
    $arguments += @("--website-slug", $Slug)
}
if ($Trace) {
    $arguments += @("--website-trace", $Trace)
}
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE
