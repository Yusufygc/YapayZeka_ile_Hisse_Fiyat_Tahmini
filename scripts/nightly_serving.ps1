# E2 Faz 8 - gecelik serving pipeline sarmalayicisi (Windows Task Scheduler hedefi).
# Her gun 21:00'de calisir (BIST kapanis sonrasi): islem-gunu kapisi -> veri tazeleme -> skorlama -> PeerStore.
# Cikti logs/nightly_<yyyyMMdd>.log dosyasina akar; 14 gunden eski loglar budanir.

$ErrorActionPreference = "Stop"

# Repo koku = bu scriptin bir ust dizini (scripts/ -> repo).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONUNBUFFERED = "1"
$env:TQDM_DISABLE = "1"

$Py = "C:\Users\ysfygc\anaconda3\envs\dl_env\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "dl_env python bulunamadi: $Py"
    exit 2
}

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Stamp = Get-Date -Format "yyyyMMdd"
$Log = Join-Path $LogDir "nightly_$Stamp.log"

"==== nightly_serving baslangic $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" |
    Out-File -FilePath $Log -Append -Encoding utf8

# stdout + stderr (PS 5.1: *>>) log'a eklenir.
# --model ensemble (E2 Faz 9): LGB+MLP 2-bacak ensemble; ~4x gece hesap, IC +%18.
& $Py "tools/e2_nightly_pipeline.py" --db "data/serving_pool.db" --boost 400 --model ensemble *>> $Log
$rc = $LASTEXITCODE

"==== nightly_serving bitis exit=$rc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" |
    Out-File -FilePath $Log -Append -Encoding utf8

# 14 gunden eski nightly loglarini buda.
Get-ChildItem -Path $LogDir -Filter "nightly_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $rc
