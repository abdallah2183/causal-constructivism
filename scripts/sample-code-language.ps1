param(
    [string]$Checkpoint = "artifacts/code-language-long/latest.pt",
    [string]$Prompt = "<file path=`"index.html`" language=`"html`" project=`"generated`">`n",
    [string]$Output = "",
    [int]$MaxBytes = 4096,
    [double]$Temperature = 0.8,
    [int]$TopK = 40,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$py = (Get-Command py -ErrorAction Stop).Source

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
$arguments = @(
    "-m", "causal_constructivism.code_language_sample",
    "--checkpoint", $Checkpoint,
    "--prompt", $Prompt,
    "--max-bytes", $MaxBytes,
    "--temperature", $Temperature,
    "--top-k", $TopK
)
if ($Output) {
    $arguments += @("--output", $Output)
}
if ($Cpu) {
    $arguments += "--cpu"
}

& $py @arguments
exit $LASTEXITCODE
