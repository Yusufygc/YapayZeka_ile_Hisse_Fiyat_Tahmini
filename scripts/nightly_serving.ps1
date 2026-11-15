# E2 Faz 8 - gecelik serving pipeline sarmalayicisi (Windows Task Scheduler hedefi).
# Her gun 21:00'de calisir (BIST kapanis sonrasi): islem-gunu kapisi -> veri tazeleme -> skorlama -> PeerStore.
# Cikti logs/nightly_<yyyyMMdd>.log dosyasina akar; 14 gunden eski loglar budanir.

$ErrorActionPreference = "Stop"

# Repo koku = bu scriptin bir ust dizini (scripts/ -> repo).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONUNBUFFERED = "1"
$env:TQDM_DISABLE = "1"

# Python yorumlayici: AI_CORE_PYTHON env -> proje .venv -> PATH'teki python.
$Py = $null
if ($env:AI_CORE_PYTHON -and (Test-Path $env:AI_CORE_PYTHON)) {
    $Py = $env:AI_CORE_PYTHON
} else {
    $VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $VenvPy) {
        $Py = $VenvPy
    } else {
        $Cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($Cmd) { $Py = $Cmd.Source }
    }
}
if (-not $Py) {
    Write-Error "python bulunamadi (AI_CORE_PYTHON, .venv\Scripts\python.exe, PATH denendi)"
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
& $Py -m src.serving.nightly --db "data/serving_pool.db" --boost 400 --model ensemble --fallback-dir "data/yedek" *>> $Log
$rc = $LASTEXITCODE

"==== nightly_serving bitis exit=$rc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" |
    Out-File -FilePath $Log -Append -Encoding utf8

# 14 gunden eski nightly loglarini buda.
Get-ChildItem -Path $LogDir -Filter "nightly_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $rc
