# E2 Faz 8 - gecelik gorevi Windows Task Scheduler'a kaydet (tek seferlik).
# Gorev her gun 21:00'te scripts/nightly_serving.ps1 calistirir (BIST kapanis
# sonrasi, ayni gun verisi hazir). Saat parametre ile degistirilebilir.
#
# 2026-11: `schtasks /Create` varsayilanlari (pilde baslamaz, pile gecince durdurur,
# kacirilan baslangici calistirmaz) yuzunden Ag-Eyl 2026 gecelik kosulari sessizce
# kesildi. Kayit artik Register-ScheduledTask + ACIK ayarlarla yapilir:
#   - pilde BASLAR            (-AllowStartIfOnBatteries)
#   - pile gecince DURDURMAZ  (-DontStopIfGoingOnBatteries)
#   - kacirilan baslangici makine uygun olunca calistirir (-StartWhenAvailable)
#   - en fazla 2 saat calisir (-ExecutionTimeLimit)
#   - paralel ornek yok       (-MultipleInstances IgnoreNew)
#
# Kullanim (yonetici PowerShell onerilir):
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_nightly_task.ps1
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_nightly_task.ps1 -Time 21:30
#
# Kaldirma:
#     Unregister-ScheduledTask -TaskName "ts_forecasting_nightly" -Confirm:$false
# Sorgu / elle tetikleme:
#     Get-ScheduledTask     -TaskName "ts_forecasting_nightly" | Select-Object -ExpandProperty Settings
#     Get-ScheduledTaskInfo -TaskName "ts_forecasting_nightly"
#     Start-ScheduledTask   -TaskName "ts_forecasting_nightly"

param([string]$Time = "21:00")

$ErrorActionPreference = "Stop"

$TaskName = "ts_forecasting_nightly"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Wrapper = Join-Path $RepoRoot "scripts\nightly_serving.ps1"

if (-not (Test-Path $Wrapper)) {
    [Console]::Error.WriteLine("Sarmalayici bulunamadi: $Wrapper")
    exit 2
}

if ($Time -notmatch '^([01]\d|2[0-3]):[0-5]\d$') {
    [Console]::Error.WriteLine("Gecersiz -Time '$Time' (beklenen SS:DD, ornek 21:00).")
    exit 2
}
$At = [datetime]::ParseExact($Time, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)

try {
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Wrapper`""
    $Trigger = New-ScheduledTaskTrigger -Daily -At $At
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
        -MultipleInstances IgnoreNew
    # schtasks /RL LIMITED + mevcut kullanici (oturum acikken) ile ayni kimlik.
    $Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
        -Settings $Settings -Principal $Principal `
        -Description "ts_forecasting_lab gecelik serving (nightly_serving.ps1)" -Force | Out-Null
} catch {
    [Console]::Error.WriteLine("Register-ScheduledTask basarisiz: $($_.Exception.Message)")
    exit 1
}

Write-Host "Gorev kaydedildi: $TaskName (her gun $Time; pilde baslar, kacirilani calistirir, 2 sa sinir, paralel yok)."
Write-Host "Dogrula: Get-ScheduledTask -TaskName $TaskName | Select-Object -ExpandProperty Settings"
Write-Host "Elle tetikle: Start-ScheduledTask -TaskName $TaskName"
exit 0
