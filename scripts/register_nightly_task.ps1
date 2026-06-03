# E2 Faz 8 - gecelik gorevi Windows Task Scheduler'a kaydet (tek seferlik).
# Gorev her gun 21:00'te scripts/nightly_serving.ps1 calistirir (BIST kapanis
# sonrasi, ayni gun verisi hazir). Saat parametre ile degistirilebilir.
#
# Kullanim (yonetici PowerShell onerilir):
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_nightly_task.ps1
#
# Kaldirma:
#     schtasks /Delete /TN "ts_forecasting_nightly" /F
# Sorgu / elle tetikleme:
#     schtasks /Query /TN "ts_forecasting_nightly" /V /FO LIST
#     schtasks /Run   /TN "ts_forecasting_nightly"

param([string]$Time = "21:00")

$ErrorActionPreference = "Stop"

$TaskName = "ts_forecasting_nightly"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Wrapper = Join-Path $RepoRoot "scripts\nightly_serving.ps1"

if (-not (Test-Path $Wrapper)) {
    Write-Error "Sarmalayici bulunamadi: $Wrapper"
    exit 2
}

$Action = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Wrapper`""

schtasks /Create /TN $TaskName /TR $Action /SC DAILY /ST $Time /F /RL LIMITED

if ($LASTEXITCODE -eq 0) {
    Write-Host "Gorev kaydedildi: $TaskName (her gun $Time)."
    Write-Host "Dogrula: schtasks /Query /TN $TaskName /V /FO LIST"
    Write-Host "Elle tetikle: schtasks /Run /TN $TaskName"
} else {
    Write-Error "schtasks /Create basarisiz (exit $LASTEXITCODE)."
}
exit $LASTEXITCODE
