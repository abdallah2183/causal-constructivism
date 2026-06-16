param(
    [string[]]$Root = @(),
    [string]$Output = "artifacts/code-corpus/local-code-corpus.jsonl",
    [int]$MaxFiles = 50000,
    [int]$MaxFileBytes = 512000,
    [int]$MaxTotalBytes = 1000000000
)

$ErrorActionPreference = "Stop"
$py = (Get-Command py -ErrorAction Stop).Source

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
$arguments = @(
    "-m", "causal_constructivism.code_corpus",
    "--output", $Output,
    "--max-files", $MaxFiles,
    "--max-file-bytes", $MaxFileBytes,
    "--max-total-bytes", $MaxTotalBytes
)
foreach ($path in $Root) {
    $arguments += @("--root", $path)
}

& $py @arguments
exit $LASTEXITCODE
