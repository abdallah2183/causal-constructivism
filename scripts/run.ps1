param(
    [int]$Steps = 5,
    [double]$TrueMass = 2.5,
    [double]$SensorNoise = 0.05,
    [int]$Seed = 7,
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
    "--steps", $Steps,
    "--true-mass", $TrueMass,
    "--sensor-noise", $SensorNoise,
    "--seed", $Seed
)
if ($Json) {
    $arguments += "--json"
}

& $python @arguments
exit $LASTEXITCODE

