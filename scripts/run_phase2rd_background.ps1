param([string]$ProjectRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')),[string]$StdoutLog='outputs/phase2rd_final_discrimination/stdout.log',[string]$StderrLog='outputs/phase2rd_final_discrimination/stderr.log',[string]$ExitCodeFile='outputs/phase2rd_final_discrimination/exit_code.txt')
$ErrorActionPreference='Continue'; Set-Location -LiteralPath $ProjectRoot; New-Item -ItemType Directory -Force -Path (Split-Path $StdoutLog)|Out-Null; $env:PYTHONPATH=(Resolve-Path 'src').Path
$code=1
try { & python -m battery_fast_charge.phase2rd_cli --config configs/phase2rd_final_pure_dnn_discrimination.yaml --project-root . 1> $StdoutLog 2> $StderrLog; $code=$LASTEXITCODE } catch { $_ | Out-File -FilePath $StderrLog -Append -Encoding utf8; $code=1 } finally { Set-Content -LiteralPath $ExitCodeFile -Value $code -Encoding ascii }
exit $code

