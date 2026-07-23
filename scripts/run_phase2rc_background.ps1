param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$StdoutLog = "outputs/phase2rc_prospective_memory_audit/stdout.log",
    [string]$StderrLog = "outputs/phase2rc_prospective_memory_audit/stderr.log",
    [string]$ExitCodeFile = "outputs/phase2rc_prospective_memory_audit/exit_code.txt"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path $StdoutLog) | Out-Null
$env:PYTHONPATH = (Resolve-Path "src").Path
& python -m battery_fast_charge.phase2rc_cli --config configs/phase2rc_prospective_control_memory.yaml --project-root . 1> $StdoutLog 2> $StderrLog
$code = $LASTEXITCODE
Set-Content -LiteralPath $ExitCodeFile -Value $code -Encoding ascii
exit $code

